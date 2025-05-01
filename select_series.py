import os
import pathlib as plb
import pydicom
import csv
from collections import defaultdict
import nibabel as nib
import numpy as np
import pydicom
import sys
import shutil
import nilearn.image
from tqdm import tqdm
import subprocess
import tempfile
import dicom2nifti

FLAGGED_CSV = "flagged_studies.csv"
nii_out_path = "/home/lukas-foerner/Data/EANM_IMM_NIFTI"


## If dcm2niix fails can try with this library
def run_dicom2nifti(input_folder, output_folder):
    try:
        dicom2nifti.convert_directory(input_folder, str(output_folder), 
                                      compression=True, reorient=True)
        print(f"Converted {input_folder} to {output_folder}")
    except Exception as e:
        print(f"Error converting {input_folder}: {e}")

def run_dcm2niix(input_folder, output_folder,):
    try:
        # Construct the nnUNet predict command
        command = [
            "dcm2niix",
            "-z", "y",
            "-b", "y",
            "-ba", "n",
            "-o", output_folder,
            input_folder
        ]

        # Execute the command
        print("Running dcm2niix...")
        subprocess.run(command, check=True)
        print(f"dcm2niix completed. Results saved to {output_folder}.")
    
    except subprocess.CalledProcessError as e:
        print(f"Error during dcm2niix: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def dcm2nii_CT(CT_dcm_path, nii_out_path):
    # conversion of CT DICOM (in the CT_dcm_path) to nifti and save in nii_out_path
    with tempfile.TemporaryDirectory() as tmp: #convert CT
        tmp = plb.Path(str(tmp))
        # convert dicom directory to nifti
        # (store results in temp directory)
        run_dcm2niix(CT_dcm_path, str(tmp))
        print(os.listdir(tmp))
        if len(os.listdir(tmp)) == 2:
            nii = next(tmp.glob('*nii.gz'))
            # copy niftis to output folder with consistent naming
            shutil.copy(nii, nii_out_path+'/CT.nii.gz')
            nii = next(tmp.glob('*json'))
            # copy niftis to output folder with consistent naming
            shutil.copy(nii, nii_out_path+'/CT.json')
        elif len(os.listdir(tmp)) == 3:
            nii = next(tmp.glob('*Eq_1.nii.gz'))
            # copy niftis to output folder with consistent naming
            shutil.copy(nii, nii_out_path+'/CT.nii.gz')
            nii = next(tmp.glob('*json'))
            # copy niftis to output folder with consistent naming
            shutil.copy(nii, nii_out_path+'/CT.json')
        else:
            #raise ValueError("CT conversion failed")
            print("CT conversion failed")


def dcm2nii_PET(PET_dcm_path, nii_out_path):
    # conversion of PET DICOM (in the PET_dcm_path) to nifti (and SUV nifti) and save in nii_out_path
    files = os.listdir(PET_dcm_path)
    first_pt_dcm = files[0]
    suv_corr_factor = calculate_suv_factor(f"{PET_dcm_path}/{first_pt_dcm}")

    with tempfile.TemporaryDirectory() as tmp: #convert PET
        tmp = plb.Path(str(tmp))
        # convert dicom directory to nifti
        # (store results in temp directory)
        run_dcm2niix(PET_dcm_path, str(tmp))
        nii = next(tmp.glob('*nii.gz'))
        # copy nifti to output folder with consistent naming
        shutil.copy(nii, nii_out_path+'/PET.nii.gz')
        nii = next(tmp.glob('*json'))
        # copy nifti to output folder with consistent naming
        shutil.copy(nii, nii_out_path+'/PET.json')

        # convert pet images to quantitative suv images and save nifti file
        suv_pet_nii = convert_pet(nib.load(nii_out_path+'/PET.nii.gz'), suv_factor=suv_corr_factor)
        nib.save(suv_pet_nii, nii_out_path+'/SUV.nii.gz')


def conv_time(time_str):
    # function for time conversion in DICOM tag
    return (float(time_str[:2]) * 3600 + float(time_str[2:4]) * 60 + float(time_str[4:13]))


def calculate_suv_factor(dcm_path):
    # reads a PET dicom file and calculates the SUV conversion factor
    ds = pydicom.dcmread(str(dcm_path))
    total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
    start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
    half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
    acq_time = ds.AcquisitionTime
    weight = ds.PatientWeight
    time_diff = conv_time(acq_time) - conv_time(start_time)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)
    suv_factor = 1000 * weight / act_dose
    return suv_factor


def convert_pet(pet, suv_factor):
    # function for conversion of PET values to SUV (should work on Siemens PET/CT)
    affine = pet.affine
    pet_data = pet.get_fdata()
    pet_suv_data = (pet_data*suv_factor).astype(np.float32)
    pet_suv = nib.Nifti1Image(pet_suv_data, affine)
    return pet_suv


def dcm2nii_mask(mask_dcm_path, nii_out_path):
    # conversion of the mask dicom file to nifti (not directly possible with dicom2nifti)
    mask_dcm = list(mask_dcm_path.glob('*.dcm'))[0]
    mask = pydicom.read_file(str(mask_dcm))
    mask_array = mask.pixel_array
    
    # get mask array to correct orientation (this procedure is dataset specific)
    mask_array = np.transpose(mask_array,(2,1,0) )  
    mask_orientation = mask[0x5200, 0x9229][0].PlaneOrientationSequence[0].ImageOrientationPatient
    if mask_orientation[4] == 1:
        mask_array = np.flip(mask_array, 1 )
    
    # get affine matrix from the corresponding pet             
    pet = nib.load(str(nii_out_path+'/PET.nii.gz'))
    pet_affine = pet.affine
    
    # return mask as nifti object
    mask_out = nib.Nifti1Image(mask_array, pet_affine)
    nib.save(mask_out, nii_out_path+'/SEG.nii.gz')   
    

def resample_ct(nii_out_path):
    # resample CT to PET and mask resolution
    ct   = nib.load(nii_out_path+'/CT.nii.gz')
    pet  = nib.load(nii_out_path+'/PET.nii.gz')
    CTres = nilearn.image.resample_to_img(ct, pet, fill_value=-1024)
    nib.save(CTres, nii_out_path+'/CTres.nii.gz')

def handle_selected_series(selected_series):
    print("\n✅ Selected Series:")
    for s in selected_series:
        print(f"Patient ID: {s['PatientID']}, Date: {s['StudyDate']}, Modality: {s['Modality']}, Description: {s['SeriesDescription']}")
        out_path = os.path.join(nii_out_path, s['PatientID'], s['StudyDate'])
        print(out_path)
        os.makedirs(out_path, exist_ok=True)
        try:
            if s['Modality'] == "CT":
                dcm2nii_CT(s["Path"].rsplit('/', 1)[0], out_path)
            if s['Modality' ] == "PT":
                dcm2nii_PET(s["Path"].rsplit('/', 1)[0], out_path)
        except Exception as e:
            print(f"Error processing {s['Modality']} series: {e}")
            continue
    print("-" * 90)

def save_flagged_series_to_csv(patient_id, study_date):
    file_exists = os.path.isfile(FLAGGED_CSV)
    with open(FLAGGED_CSV, mode="a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["PatientID", "StudyDate"])
        if not file_exists:
            writer.writeheader()
            writer.writerow({
                "PatientID": patient_id,
                "StudyDate": study_date,
            })

def collect_series(study_dir):
    study_dir = plb.Path(study_dir)
    sub_dirs = [plb.Path(x[0]) for x in os.walk(study_dir)]

    grouped = defaultdict(list)

    for dir in sub_dirs:
        dicom_files = [f for f in dir.iterdir() if f.is_file() and f.name.lower() != "dicomdir"]
        if not dicom_files:
            continue

        first_file = dicom_files[0]

        try:
            ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)

            patient_id = getattr(ds, "PatientID", None)
            study_date = getattr(ds, "StudyDate", None)
            modality = getattr(ds, "Modality", None)
            series_desc = getattr(ds, "SeriesDescription", "").lower()
            study_desc = getattr(ds, "StudyDescription", "N/A")

            if not (patient_id and study_date and modality):
                continue

            if modality not in ("CT", "PT"):
                continue
            
            out_path_CT = os.path.join(nii_out_path, patient_id, study_date, "CT.nii.gz")
            out_path_PT = os.path.join(nii_out_path, patient_id, study_date, "PET.nii.gz")
            if os.path.isfile(out_path_CT) and os.path.isfile(out_path_PT):
                continue

            grouped[(patient_id, study_date)].append({
                "PatientID": patient_id,
                "StudyDate": study_date,
                "Modality": modality,
                "SeriesDescription": series_desc,
                "StudyDescription": study_desc,
                "Path": str(first_file)
            })

        except Exception as e:
            print(f"Failed to read DICOM: {first_file} — {e}")
            continue

    return grouped

def is_preselected(series):
    desc = series["SeriesDescription"]
    modality = series["Modality"]
    if modality == "CT" and "knochen" in desc:
        return True
    if modality == "PT" and any(x in desc for x in ["pet gk ctac", "qc fx"]):
        return True
    return False

def find_default_indices(series_list):
    preselected_indices = []
    has_knochen_ct = any(
        s["Modality"] == "CT" and "knochen" in s["SeriesDescription"] for s in series_list
    )

    if has_knochen_ct:
        for i, s in enumerate(series_list):
            desc = s["SeriesDescription"]
            modality = s["Modality"]
            if modality == "CT" and "knochen" in desc:
                preselected_indices.append(i)
            if modality == "PT" and any(x in desc for x in ["pet gk ctac", "qc fx"]):
                preselected_indices.append(i)
        return preselected_indices, False  # no flag

    # Fallback: look for weichteil CT
    for i, s in enumerate(series_list):
        desc = s["SeriesDescription"]
        modality = s["Modality"]
        if modality == "CT" and "weichteil" in desc:
            preselected_indices.append(i)
        if modality == "PT" and any(x in desc for x in ["pet gk ctac", "qc fx"]):
            preselected_indices.append(i)
    return preselected_indices, True  # weichteil selected, should flag

    #return [], False


def interactive_selection(grouped):
    sorted_keys = sorted(grouped.keys())
    total = len(sorted_keys)

    for idx, (patient_id, study_date) in enumerate(sorted_keys, start=1):
        series_list = grouped[(patient_id, study_date)]
        if not series_list:
            continue

        print(f"\n📚 Study {idx} of {total} — Patient ID: {patient_id} — Study Date: {study_date} - Study Desc: {series_list[0]['StudyDescription']}")
        print("Available Series:")

        preselected_indices, fallback_flag = find_default_indices(series_list)

        for i, s in enumerate(series_list):
            pre = i in preselected_indices
            mark = "[*]" if pre else "[ ]"
            print(f"{mark} [{i:2}] {s['Modality']:>3} | {s['SeriesDescription']}")

        if fallback_flag:
            print("⚠️ No 'knochen' CT found — defaulted to 'weichteil' CT and flagged study.")

        default_input = ",".join(str(i) for i in preselected_indices)
        user_input = input(f"Enter numbers to select (comma-separated), add 'x' to flag study [default: {default_input}]: ").strip().lower()

        # Parse selection
        flag_study = fallback_flag or 'x' in user_input
        input_parts = [part.strip() for part in user_input.split(",") if part.strip().isdigit()]
        if not user_input:
            indices = preselected_indices
        else:
            indices = [int(i) for i in input_parts if i.isdigit() and 0 <= int(i) < len(series_list)]

        selected_series = [series_list[i] for i in indices]
        handle_selected_series(selected_series)

        if flag_study:
            print("🚩 Study flagged and saved to CSV.")
            save_flagged_series_to_csv(patient_id, study_date)

if __name__ == "__main__":
    study_dir = "/data/IMM_reexport"  # Replace with your actual path
    grouped_series = collect_series(study_dir)
    interactive_selection(grouped_series)
