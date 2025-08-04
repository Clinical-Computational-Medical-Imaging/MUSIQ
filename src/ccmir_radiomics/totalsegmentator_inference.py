import json
import logging
import os

from totalsegmentator.config import get_version
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from totalsegmentator.python_api import totalsegmentator

logger = logging.getLogger(__name__)


class TotalSegmentatorInference:
    def __init__(self, input_dirpath_processed: str | os.PathLike) -> None:
        """Class to handle TotalSegmentator inference on CT.nii.gz files in a specified folder.
        It processes each file, runs segmentation, extracts label mapping. Creates CTseg.nii.

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the CT.nii.gz files. Can be nested.
        """
        self.input_dirpath = input_dirpath_processed

    def run(self) -> None:
        """
        Recursively search the folder for CT.nii.gz files.
        For each found file, create a 'CTseg' subfolder, run segmentation using ml option,
        extract the label mapping from the segmentation output, and save a metadata JSON file.
        """
        if not os.path.isdir(self.input_dirpath):
            logger.error(f"Error: {self.input_dirpath} is not a valid directory.")
            return

        logger.info(f"Starting TotalSegmentator inference in {self.input_dirpath}")
        for dirpath, _, filenames in os.walk(self.input_dirpath):
            for filename in filenames:
                # Determine if this is a CT or MR file and set parameters accordingly
                is_ct = filename == "CT.nii.gz"
                is_mr = filename.endswith("nii.gz") and not filename.startswith(("CT", "SUV", "PET"))

                if not (is_ct or is_mr):
                    continue

                input_fpath = os.path.join(dirpath, filename)
                if is_ct:
                    output_fpath = os.path.join(dirpath, "CTseg.nii.gz")
                    task = "total"
                    modality = "CT"
                    metadata_key = "CTseg_metadata"
                    seg_path_key = "CTsegPath"
                else:
                    output_fpath = os.path.join(dirpath, f"{filename[:-7]}_seg.nii.gz")
                    task = "total_mr"
                    modality = "MR"
                    metadata_key = "MRseg_metadata"
                    seg_path_key = "MRsegPath"

                if os.path.isfile(output_fpath):
                    logger.info(f"Output file {output_fpath} already exists.")
                    continue

                patient_dirpath = os.path.dirname(dirpath)
                patient_info = None
                patient_info_path = os.path.join(patient_dirpath, "patient_info.json")
                if os.path.isfile(patient_info_path):
                    flag_json_exists = True
                    with open(patient_info_path) as json_file:
                        patient_info = json.load(json_file)
                else:
                    flag_json_exists = False
                    logger.error(f"Missing patient_info.json in {patient_dirpath}.")

                logger.info(f"Processing file: {input_fpath} to {output_fpath}")

                # Run TotalSegmentator using the Python API with ml option and appropriate task.
                try:
                    totalsegmentator(
                        input_fpath,
                        output_fpath,
                        ml=True,
                        task=task,
                        device="gpu:0",
                        statistics=False,
                        radiomics=False,
                    )
                    logger.info("Segmentation successfully completed.")
                except Exception as e:
                    logger.error(f"Error during segmentation for {input_fpath}:\n  {e}")
                    continue

                # Load the segmentation file to extract the label mapping from its extended header.
                try:
                    segmentation_img, label_map_dict = load_multilabel_nifti(output_fpath)
                    logger.info("Label mapping successfully loaded from segmentation file.")
                except Exception as e:
                    logger.error(f"Error loading segmentation file {output_fpath}: {e}")
                    label_map_dict = {}

                # Prepare metadata with settings, task/model info, and the label mapping obtained.
                seg_metadata = {
                    "settings": {"input_fpath": input_fpath, "task": task, "ml": True},
                    "model": "total",
                    "ts_version": get_version(),
                    "labels": label_map_dict,
                }
                if flag_json_exists and patient_info is not None:
                    study_date = dirpath.split(os.sep)[-1]
                    if modality == "CT":
                        series_index = 0
                    else:
                        mr_series = patient_info["Studies"][study_date]["Modalities"][modality]
                        # Find the index where the filename matches the MRPath value
                        series_index = None
                        for idx, serie in enumerate(mr_series):
                            for _serie_name, serie_data in serie.items():
                                if "MRPath" in serie_data and filename in os.path.basename(serie_data["MRPath"]):
                                    series_index = idx
                                    break
                            if series_index is not None:
                                break

                    if series_index is None:
                        logger.error(f"Could not find series index for {filename} in patient_info.json.")
                        continue

                    series_name = next(iter(patient_info["Studies"][study_date]["Modalities"][modality][series_index]))
                    patient_info["Studies"][study_date]["Modalities"][modality][series_index][series_name].update(
                        {seg_path_key: output_fpath}
                    )
                    patient_info["Studies"][study_date]["Modalities"][modality][series_index][series_name].update(
                        {metadata_key: seg_metadata}
                    )
                    with open(patient_info_path, "w") as f:
                        json.dump(patient_info, f)
                else:
                    with open(f"{filename[:-7]}_seg.json", "w") as f:
                        json.dump(seg_metadata, f)


def totalsegmentator_inference_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger()

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run TotalSegmentator (ml option) on all CT.nii.gz files in a folder. "
        "Extract label mapping from the segmentation output and save metadata as CTseg.json."
    )
    parser.add_argument(
        "--input-dirpath-processed", type=str, help="Path to the input folder containing CT.nii.gz files", required=True
    )
    args = parser.parse_args()

    TotalSegmentatorInference(
        input_dirpath_processed=args.input_dirpath_processed,
    ).run()


if __name__ == "__main__":
    totalsegmentator_inference_entrypoint()
