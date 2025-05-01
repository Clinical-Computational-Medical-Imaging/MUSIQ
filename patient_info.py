#!/usr/bin/env python3
import sys
import json
import pydicom
from pydicom.errors import InvalidDicomError
import pathlib as plb
from collections import defaultdict
import os

nii_out_path = "/home/lukas-foerner/Data/EANM_IMM_NIFTI" # <- Replace this with the actual path
study_dir = "/data/IMM_reexport"  # <- Replace this with the actual path

# List of standard patient info tags to extract
PATIENT_TAGS = {
    'PatientName':         ('0010', '0010'),
    'PatientID':           ('0010', '0020'),
    'PatientBirthDate':    ('0010', '0030'),
    'PatientSex':          ('0010', '0040'),
    'PatientAge':          ('0010', '1010'),
    'PatientSize':         ('0010', '1020'),  # in meters
    'PatientWeight':       ('0010', '1030'),  # in kilograms
}

def extract_patient_info(dcm):
    """Pulls patient info from a pydicom Dataset into a dict."""
    info = {}
    for key, (grp, elem) in PATIENT_TAGS.items():
        tag = (int(grp, 16), int(elem, 16))
        if tag in dcm:
            value = dcm.get(tag).value
            # Convert PersonName to string
            if hasattr(value, 'family_name') or hasattr(value, 'given_name'):
                value = str(value)
            info[key] = value
    return info

def main(dicom_path):
    try:
        dcm = pydicom.dcmread(dicom_path)
    except (InvalidDicomError, FileNotFoundError) as e:
        sys.stderr.write(f"Error reading DICOM file: {e}\n")
        #sys.exit(1)

    patient_id = getattr(dcm, "PatientID", None)
    study_date = getattr(dcm, "StudyDate", None)
    out_path_info = os.path.join(nii_out_path, patient_id, study_date, "patient_info.json")

    patient_info = extract_patient_info(dcm)

    with open(out_path_info, 'w', encoding='utf-8') as f:
        json.dump(patient_info, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(patient_info)} fields to {out_path_info}")

if __name__ == "__main__":
    study_dir = plb.Path(study_dir)
    sub_dirs = [plb.Path(x[0]) for x in os.walk(study_dir)]

    grouped = defaultdict(list)

    for dir in sub_dirs:
        dicom_files = [f for f in dir.iterdir() if f.is_file() and f.name.lower() != "dicomdir" and f.suffix.lower() != ".zip"]
        if not dicom_files:
            continue

        first_file = dicom_files[0]

        try:
            print(f"Reading {first_file}")
            main(str(first_file))
        except Exception as e:
            #sys.stderr.write(f"Error processing file {first_file}: {e}\n")
            continue




