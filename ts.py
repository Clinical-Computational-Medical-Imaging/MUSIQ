#!/usr/bin/env python3
import os
import json
import argparse

# Import the totalsegmentator function and the extended header loader from the TotalSegmentator package.
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from totalsegmentator.config import get_version

def process_folder(root_folder):
    """
    Recursively search the folder for CT.nii.gz files.
    For each found file, create a 'CTseg' subfolder, run segmentation using ml option,
    extract the label mapping from the segmentation output, and save a metadata JSON file.
    """
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "CT.nii.gz":
                input_file = os.path.join(dirpath, filename)
                output_file = os.path.join(dirpath, "CTseg.nii.gz")
                #output_folder = os.path.join(dirpath, "CTseg")
                #os.makedirs(output_folder, exist_ok=True)
                print(f"\nProcessing file:\n  Input: {input_file}\n  Output: {output_file}")

                if os.path.isfile(output_file):
                    continue

                # Run TotalSegmentator using the Python API with ml option and task "total".
                try:
                    totalsegmentator(input_file, output_file, ml=True, task="total", device="gpu:0", statistics=False, radiomics=False)
                    print("Segmentation successfully completed.")
                except Exception as e:
                    print(f"Error during segmentation for {input_file}:\n  {e}")
                    continue

                # Load the segmentation file to extract the label mapping from its extended header.
                try:
                    segmentation_img, label_map_dict = load_multilabel_nifti(output_file)
                    print("Label mapping successfully loaded from segmentation file.")
                except Exception as e:
                    print(f"Error loading segmentation file {output_file}: {e}")
                    label_map_dict = {}

                # Prepare metadata with settings, task/model info, and the label mapping obtained.
                metadata = {
                    "settings": {
                        "input_file": input_file,
                        "task": "total",
                        "ml": True
                    },
                    "model": "total",
                    "ts_version": get_version(),
                    "labels": label_map_dict
                }

                # Write out the metadata to CTseg.json in the same directory as the CT.nii.gz file.
                json_path = os.path.join(dirpath, "CTseg.json")
                try:
                    with open(json_path, "w") as fp:
                        json.dump(metadata, fp, indent=4)
                    print(f"Metadata saved to {json_path}")
                except Exception as e:
                    print(f"Error writing metadata for {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively run TotalSegmentator (ml option) on all CT.nii.gz files in a folder. "
                    "Extract label mapping from the segmentation output and save metadata as CTseg.json."
    )
    parser.add_argument("folder", type=str, help="Path to the input folder containing CT.nii.gz files")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory.")
        return

    process_folder(args.folder)

if __name__ == "__main__":
    main()
