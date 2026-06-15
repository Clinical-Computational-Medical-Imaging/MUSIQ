import json
import logging
import os

import cc3d
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import metrics, utils

logger = logging.getLogger(__name__)


class TumorInfoExtraction:
    def __init__(self, input_dirpath_processed: str | os.PathLike) -> None:
        """Class to handle tumor information extraction from PETseg, SUV, and CTseg files in a specified folder.
        Creates CTsegres.nii.gz if it does not exist.
        The existing patient_info.json is required to extract and save data.
        Expects exactly one CT.nii.gz file per serie.

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the CT.nii.gz file. Can be nested.
        """
        self.input_dirpath = input_dirpath_processed

    def run(self) -> None:
        study_dirs = []

        for dirpath, dirnames, _filenames in os.walk(self.input_dirpath):
            for subdirname in dirnames:
                subdirpath = os.path.join(dirpath, subdirname)
                required_files = ["PETseg.nii.gz", "CTseg.nii.gz", "SUV.nii.gz"]
                if all(os.path.exists(os.path.join(subdirpath, f)) for f in required_files):
                    study_dirs.append(subdirpath)
                    break  # Stop after first matching subdirectory

        if not study_dirs:
            logger.info(
                "No complete studies found in the input directory. "
                "PETseg.nii.gz, CTseg.nii.gz, SUV.nii.gz are "
                "required in each patient/study directory."
            )
            return

        study_dirs = sorted(study_dirs, key=lambda x: os.path.basename(os.path.dirname(x)))
        for study_dirpath in tqdm(study_dirs):
            self.process_study(study_dirpath)

    @staticmethod
    def process_study(study_dirpath) -> None:
        """Process a single study directory."""
        logger.info(f"Extracting tumor metrics for {study_dirpath}")
        petseg_fpath = os.path.join(study_dirpath, "PETseg.nii.gz")
        ctsegres_fpath = os.path.join(study_dirpath, "CTsegres.nii.gz")
        suv_fpath = os.path.join(study_dirpath, "SUV.nii.gz")

        patient_dirpath = os.path.dirname(study_dirpath)

        if not os.path.isfile(os.path.join(patient_dirpath, "patient_info.json")):
            json_exists = False
            logger.error(f"Missing patient_info.json in {patient_dirpath}. Radiomics are extracted to csv.")
        else:
            json_exists = True
            with open(os.path.join(patient_dirpath, "patient_info.json")) as json_file:
                patient_info = json.load(json_file)

        if not os.path.exists(ctsegres_fpath):
            utils.resample_image(
                source_img=os.path.join(study_dirpath, "CTseg.nii.gz"),
                target_img=os.path.join(study_dirpath, "PET.nii.gz"),
                nii_output_dirpath=study_dirpath,
                interpolation="nearest",
                fill_value=0,
                output_fname="CTsegres.nii.gz",
            )

        petseg_array = metrics.get_3darray_from_niftipath(petseg_fpath)
        ctsegres_array = metrics.get_3darray_from_niftipath(ctsegres_fpath)
        suv_array = metrics.get_3darray_from_niftipath(suv_fpath)
        spacing = utils.get_spacing_from_niftipath(suv_fpath)
        voxel_volume_cc = np.prod(spacing) / 1000

        study_date = study_dirpath.split(os.sep)[-1]

        # expects exactly one CT per serie
        if json_exists:
            series_name = next(iter(patient_info["Studies"][study_date]["Modalities"]["CT"][0]))
            try:
                organ_labels = patient_info["Studies"][study_date]["Modalities"]["CT"][0][series_name][
                    "CTseg_metadata"
                ]["labels"]
            except KeyError:
                logger.warning(
                    f"CTseg_metadata not found in {patient_dirpath}/patient_info.json for {study_date}. "
                    "Organ overlap metrics will not be computed."
                )
                organ_labels = {}
        else:
            logger.warning(
                f"No patient_info.json for {patient_dirpath}/{study_date}. Organ overlap metrics will not be computed."
            )
            organ_labels = {}

        # Perform connected component analysis
        labeled_tumors, num_lesions = cc3d.connected_components(petseg_array, connectivity=26, return_N=True)

        # Process each tumor
        results = []
        for i in range(1, num_lesions + 1):
            tumor_mask = np.zeros_like(labeled_tumors)
            tumor_mask[labeled_tumors == i] = 1

            pet_metrics = utils.compute_pet_metrics(tumor_mask, suv_array, spacing)
            if organ_labels:
                organ_overlap = utils.compute_tumor_organ_overlap(tumor_mask, ctsegres_array, organ_labels)
            num_nonzero_voxels = len(np.nonzero(tumor_mask)[0])

            results.append(
                {
                    "TumorID": i,
                    "Volume_cm3": num_nonzero_voxels * voxel_volume_cc,
                    "PETMetrics": pet_metrics,
                    "OrganOverlap": organ_overlap if organ_labels else {},
                }
            )
        if json_exists:
            patient_info["Studies"][study_date]["TumorStats"].update({"Tumors": results})
            with open(os.path.join(patient_dirpath, "patient_info.json"), "w") as f:
                json.dump(patient_info, f)
        else:
            pd.json_normalize(results).to_csv(
                os.path.join(patient_dirpath, "tumor_statistics.csv"),
                index=False,
            )


def tumor_info_extraction_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.tumor_info_extraction")

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run tumor information extraction on all studies in a folder."
    )
    parser.add_argument(
        "--input-dirpath-processed",
        type=str,
        help="Path to the input folder containing PETseg.nii.gz files",
        required=True,
    )
    args = parser.parse_args()

    TumorInfoExtraction(
        input_dirpath_processed=args.input_dirpath_processed,
    ).run()


if __name__ == "__main__":
    tumor_info_extraction_entrypoint()
