import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import metrics
from .utils import get_spacing_from_niftipath, make_json_safe

logger = logging.getLogger(__name__)


class RadiomicsExtractor:
    def __init__(self, input_dirpath_processed: str | os.PathLike) -> None:
        """Calculates patient-level statistics from PET and segmentation images.
        Expects exactly one PT and matching CT series per study date

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the SUV, PETseg and PET.json files.
            Can be nested.
        """
        self.input_dirpath = input_dirpath_processed
        self.multiprocessing = False

    def run(self) -> None:
        sub_dirs = [dirpath for dirpath, _, filenames in os.walk(self.input_dirpath) if "PETseg.nii.gz" in filenames]
        if not sub_dirs:
            logger.info("No directories found with PETseg.nii.gz files.")
            return

        # paralellize
        if self.multiprocessing:
            with ProcessPoolExecutor() as executor:
                list(executor.map(self.process_directory_wrapper, sub_dirs))

        for dirpath in sub_dirs:
            self.compute_patient_radiomics(dirpath)

    def compute_patient_radiomics(self, dirpath: str | os.PathLike) -> None:
        """Computes patient-level radiomics metrics from SUV and PETseg files.

        Args:
            dirpath (str | os.PathLike): Path to patient sub directory containing
            SUV.nii.gz, PETseg.nii.gz, and series_patient_info.json.
        """
        ct_fpath = os.path.join(dirpath, "CT.nii.gz")
        ctseg_fpath = os.path.join(dirpath, "CTseg.nii.gz")
        suv_fpath = os.path.join(dirpath, "SUV.nii.gz")
        petseg_fpath = os.path.join(dirpath, "PETseg.nii.gz")
        patient_dirpath = os.path.dirname(dirpath)
        patient_info_path = os.path.join(patient_dirpath, "patient_info.json")

        if not os.path.isfile(patient_info_path):
            logger.error(f"Missing patient_info.json in {patient_dirpath}. Radiomics extraction not possible.")
            return
        else:
            logger.info(f"Processing: {dirpath}")
            with open(patient_info_path) as json_file:
                patient_info = json.loads(json_file.read())

            study_date = os.path.join(dirpath).split(os.sep)[-1]
            study = patient_info["Studies"].get(study_date, {})
            existing_metrics = study.get("TumorStats", {})
            if study.get("PatientWeight") and study.get("PatientSize"):
                std_factor = np.sqrt(
                    (float(study.get("PatientWeight")) * (float(study.get("PatientSize")) * 100)) / 3600
                )
            else:
                std_factor = None

        ptarray = metrics.get_3darray_from_niftipath(suv_fpath)
        gtarray = metrics.get_3darray_from_niftipath(petseg_fpath)
        spacing = get_spacing_from_niftipath(petseg_fpath)

        # Define all metrics to calculate
        metrics_to_calculate = {
            "SUVmean": lambda: metrics.calculate_patient_level_lesion_suvmean_suvmax(
                ptarray, gtarray, marker="SUVmean"
            ),
            "SUVmax": lambda: metrics.calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker="SUVmax"),
            "SUVpeak": lambda: metrics.calculate_suvpeak_median(ptarray, gtarray, spacing),
            "SUVstd": lambda: metrics.calculate_patient_level_lesion_suvmean_suvmax(ptarray, gtarray, marker="SUVstd"),
            "LesionCount": lambda: metrics.calculate_patient_level_lesion_count(gtarray),
            "TMTV": lambda: metrics.calculate_patient_level_tmtv(gtarray, spacing),
            "TLG": lambda: metrics.calculate_patient_level_tlg(ptarray, gtarray, spacing),
            "Dmax": lambda: metrics.calculate_patient_level_dissemination(gtarray, spacing),
            "SDmax": lambda: metrics.calculate_patient_level_dissemination(gtarray, spacing) / std_factor
            if std_factor
            else "NAN",
            "SurfaceArea": lambda: metrics.calculate_patient_level_surface_area(gtarray, spacing),
            "MTV2.5": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 2.5, spacing),
            "MTV3.0": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 3.0, spacing),
            "MTV3.5": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 3.5, spacing),
            "MTV4.0": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 4.0, spacing),
            "MTV30": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 0.3, spacing, rel_thres=True),
            "MTV40": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 0.4, spacing, rel_thres=True),
            "MTV41": lambda: metrics.calculate_patient_level_mtv_psmatv(
                ptarray, gtarray, 0.41, spacing, rel_thres=True
            ),
            "MTV50": lambda: metrics.calculate_patient_level_mtv_psmatv(ptarray, gtarray, 0.5, spacing, rel_thres=True),
            # dirty workaround to use the function for HU multi-metrics
            "HU Mean": lambda: metrics.calculate_hu_statistics(ct_fpath, ctseg_fpath),
        }

        # Calculate missing metrics
        new_metrics = {}
        for metric, calc_func in metrics_to_calculate.items():
            if metric not in existing_metrics or existing_metrics[metric] in [None, "", "0"]:
                try:
                    if metric == "HU Mean":  # dirty workaround continued
                        new_metrics.update(**calc_func())
                    else:
                        new_metrics[metric] = calc_func()
                except Exception as e:
                    logger.info(f"Error calculating {metric}: {e}")
                    new_metrics[metric] = None

        # Append new metrics to the CSV file
        if new_metrics:
            patient_info["Studies"][study_date]["TumorStats"] = {**existing_metrics, **new_metrics}
            with open(os.path.join(patient_dirpath, "patient_info.json"), "w") as f:
                json.dump(make_json_safe(patient_info), f)

    def process_directory_wrapper(self, args) -> None:
        try:
            dirpath, _ = args
            return self.compute_patient_radiomics(dirpath)
        except ValueError:
            return


def radiomics_extraction_entrypoint() -> None:
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

    RadiomicsExtractor(
        input_dirpath_processed=args.input_dirpath_processed,
    ).run()


if __name__ == "__main__":
    radiomics_extraction_entrypoint()
