import metrics
import os
import SimpleITK as sitk  
import csv
from concurrent.futures import ProcessPoolExecutor
import functools
from multiprocessing import Pool
from tqdm import tqdm
import json
import numpy as np

#root_folder = "/home/lukas-foerner/Projects/ALPS_PROSTATE/data/"
root_folder = "/home/lukas-foerner/Data/EANM_IMM_NIFTI"

def get_spacing_from_niftipath(path):
    spacing = sitk.ReadImage(path).GetSpacing()
    return spacing

def process_directory(dirpath, filenames):
    print(filenames)
    if "PETseg.nii.gz" not in filenames:
        print(f"Skipping: {dirpath}")
        return

    ptpath = os.path.join(dirpath, "SUV.nii.gz")
    gtpath = os.path.join(dirpath, "PETseg.nii.gz")
    pipath = os.path.join(dirpath, "patient_info.json")
    output_csv_path = os.path.join(dirpath, "patient_level_stats.csv")

    existing_metrics = {}
    if os.path.exists(output_csv_path):
        with open(output_csv_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header
            existing_metrics = {row[0]: row[1] for row in reader}

    print(f"Processing: {dirpath}")
    with open(pipath, 'r', encoding='utf-8') as f:
        patient_info = json.load(f)
    ptarray = metrics.get_3darray_from_niftipath(ptpath)
    gtarray = metrics.get_3darray_from_niftipath(gtpath)
    spacing = get_spacing_from_niftipath(gtpath)
    std_factor = None
    if patient_info.get('PatientWeight') and patient_info.get('PatientSize'):
        std_factor = np.sqrt((patient_info.get('PatientWeight') * (patient_info.get('PatientSize') * 100)) / 3600)

    # Define all metrics to calculate
    metrics_to_calculate = {
        "SUVmean": lambda: metrics.calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker='SUVmean'),
        "SUVmax": lambda: metrics.calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker='SUVmax'),
        "SUVpeak": lambda: metrics.calculate_suvpeak_median(ptarray, gtarray, spacing),
        "SUVstd": lambda: metrics.calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker='SUVstd'),
        "LesionCount": lambda: metrics.calculate_patient_level_lesion_count(gtarray),
        "TMTV": lambda: metrics.calculate_patient_level_tmtv(gtarray, spacing),
        "TLG": lambda: metrics.calculate_patient_level_tlg(ptarray, gtarray, spacing),
        "Dmax": lambda: metrics.calculate_patient_level_dissemination(gtarray, spacing),
        "SDmax": lambda: metrics.calculate_patient_level_dissemination(gtarray, spacing) / std_factor if std_factor else "NAN",
        "SurfaceArea": lambda: metrics.calculate_patient_level_surface_area(gtarray, spacing),
        "MTV2.5": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 2.5, spacing),
        "MTV3.0": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 3.0, spacing),
        "MTV3.5": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 3.5, spacing),
        "MTV4.0": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 4.0, spacing),
        "MTV30": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 0.3, spacing, rel_thres=True),
        "MTV40": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 0.4, spacing, rel_thres=True),
        "MTV41": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 0.41, spacing, rel_thres=True),
        "MTV50": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 0.5, spacing, rel_thres=True),
    }

    # Calculate missing metrics
    new_metrics = {}
    for metric, calc_func in metrics_to_calculate.items():
        if metric not in existing_metrics or existing_metrics[metric] in [None, '', '0']:
            try:
                new_metrics[metric] = calc_func()
            except Exception as e:
                print(f"Error calculating {metric}: {e}")
                new_metrics[metric] = None

    # Append new metrics to the CSV file
    if new_metrics:
        with open(output_csv_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not existing_metrics:  # Write header if file is empty
                writer.writerow(["Metric", "Value"])
            for metric, value in new_metrics.items():
                writer.writerow([metric, value])

    print(f"Updated results saved to {output_csv_path}")

def process_directory_wrapper(args):
    dirpath, filenames = args
    return process_directory(dirpath, filenames)

# Collect all directories and filenames
directories = [(dirpath, filenames) for dirpath, _, filenames in os.walk(root_folder)]
print(f"Found {len(directories)} directories to process.")

if __name__ == "__main__":
    # Use multiprocessing.Pool to parallelize the processing
    with Pool() as pool:
        list(tqdm(pool.imap_unordered(process_directory_wrapper, directories), total=len(directories)))
    #for dirpath, filenames in tqdm(directories):
    #    process_directory(dirpath, filenames)
