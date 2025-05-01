import os
import pathlib as plb
import pydicom
import csv

def collect_studies(study_dir):
    """
    Walks through the study directory, reads each DICOM file (skipping DICOMDIR),
    and groups series by (PatientID, StudyDate). Also records the relative folder path
    so that we can later map it to the output directory.
    """
    study_dir = plb.Path(study_dir)
    grouped = {}  # key: (PatientID, StudyDate) -> {"series":[], "rel_folder": relative_path}
    
    for root, dirs, files in os.walk(study_dir):
        # Build list of candidate DICOM files
        dicom_files = [plb.Path(root) / f for f in files if f.lower() != 'dicomdir']
        if not dicom_files:
            continue

        first_file = dicom_files[0]
        try:
            ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)
        except Exception as e:
            print(f"Error reading {first_file}: {e}")
            continue

        patient_id = getattr(ds, "PatientID", None)
        study_date = getattr(ds, "StudyDate", None)
        modality = getattr(ds, "Modality", None)
        # Lowercase the series description for case-insensitive matching.
        series_desc = getattr(ds, "SeriesDescription", "").lower()

        if not (patient_id and study_date and modality):
            continue

        key = (patient_id, study_date)
        if key not in grouped:
            # Save the folder relative to the input study_dir. This is used to look for the matching output.
            rel_folder = plb.Path(root).relative_to(study_dir)
            grouped[key] = {"series": [], "rel_folder": rel_folder}
        grouped[key]["series"].append({
            "PatientID": patient_id,
            "StudyDate": study_date,
            "Modality": modality,
            "SeriesDescription": series_desc,
            "Folder": plb.Path(root)
        })
    return grouped

def has_knochen_ct(series_list):
    """
    Returns True if at least one CT series has "knochen" in the SeriesDescription.
    """
    for s in series_list:
        if s["Modality"] == "CT" and "knochen" in s["SeriesDescription"]:
            return True
    return False

def get_modalities_in_dir(directory):
    """
    Scans the specified directory (and its subdirectories) for DICOM files and
    collects the modalities found. It stops searching early if both "CT" and "PT" are found.
    """
    modalities = set()
    directory = plb.Path(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = plb.Path(root) / file
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                mod = getattr(ds, "Modality", None)
                if mod:
                    modalities.add(mod)
                # If both CT and PT are found, we can stop early.
                if "CT" in modalities and "PT" in modalities:
                    return modalities
            except Exception:
                continue
    return modalities

def write_missing_studies(study_dir, output_dir, csv_file):
    """
    Looks at all studies in study_dir and for each study that either has no knochen CT
    OR whose corresponding output folder is missing CT or PET series, writes a row to csv_file.
    """
    grouped = collect_studies(study_dir)
    csv_rows = []
    output_dir = plb.Path(output_dir)

    for key, data in grouped.items():
        patient_id, study_date = key
        reasons = []
        
        # Check for a "knochen" CT in the input series.
        if not has_knochen_ct(data["series"]):
            reasons.append("no knochen CT in input")
        
        # Derive the corresponding output folder from the relative path.
        out_folder = output_dir / patient_id / study_date
        #print(out_folder)
        #exit()
        if not out_folder.exists():
            reasons.append("output folder missing")
        else:
            out_modalities = get_modalities_in_dir(out_folder)
            if not os.path.isfile(str(out_folder) +"/CT.nii.gz"):
                reasons.append("missing CT in output")
            if not os.path.isfile(str(out_folder) +"/PET.nii.gz"):
                reasons.append("missing PET in output")
        
        # If any issue is found, add a CSV entry.
        if reasons:
            csv_rows.append({
                "PatientID": patient_id,
                "StudyDate": study_date,
                "Reason": "; ".join(reasons)
            })

    # Write the CSV file.
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["PatientID", "StudyDate", "Reason"])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"CSV written to {csv_file} with {len(csv_rows)} flagged studies.")

if __name__ == "__main__":
    # Update these paths as needed
    study_dir = "/data/EANM_IMM"   # The directory with your original DICOM studies.
    output_dir = "/home/lukas-foerner/Data/EANM_IMM_NIFTI"  # The output directory that should contain processed CT and/or PET.
    csv_file = "missing_studies_imm.csv"     # Output CSV filename.
    
    write_missing_studies(study_dir, output_dir, csv_file)
