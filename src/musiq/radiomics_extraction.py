import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import metrics
from .utils import get_spacing_from_niftipath, make_json_safe

logger = logging.getLogger(__name__)


class RadiomicsExtractor:
    def __init__(self, input_dirpath_processed: str | os.PathLike, pet_metric: str | list[str] = ["SUV", "SUL"]) -> None:
        """Calculates patient-level statistics from PET and segmentation images.
        Expects exactly one PT and matching CT series per study date

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the SUV/SUL, PETseg and PET.json files.
            Can be nested.
            pet_metric (str | list[str]): PET metric(s) to use as input. "SUV", "SUL", or both (default: "SUV SUL").
        """
        pet_metrics = [pet_metric] if isinstance(pet_metric, str) else list(pet_metric)
        for m in pet_metrics:
            if m not in ("SUV", "SUL"):
                raise ValueError(f"pet_metric must be 'SUV' or 'SUL', got '{m}'")
        self.input_dirpath = input_dirpath_processed
        self.pet_metrics = pet_metrics
        self.multiprocessing = False

    def run(self) -> None:
        for metric in self.pet_metrics:
            petseg_fname = "PETseg.nii.gz" if metric == "SUV" else "PETsegSUL.nii.gz"
            necessary_files = [petseg_fname, f"{metric}.nii.gz", "CTseg.nii.gz", "CT.nii.gz"]
            sub_dirs = [
                dirpath
                for dirpath, _, filenames in os.walk(self.input_dirpath)
                if all(f in filenames for f in necessary_files)
            ]
            if not sub_dirs:
                msg = f"No directories found with necessary files for {metric}: {necessary_files}."
                if metric == "SUL":
                    msg += " SUL.nii.gz and PETsegSUL.nii.gz are created by the muscle-fat and autopet tasks — make sure both have been run first."
                logger.warning(msg)
                continue

            sub_dirs = sorted(sub_dirs, key=lambda x: os.path.basename(os.path.dirname(x)))
            logger.info("Running radiomics extraction for %s on %d directories.", metric, len(sub_dirs))
            # paralellize
            if self.multiprocessing:
                with ProcessPoolExecutor(max_workers=30) as executor:
                    list(executor.map(self.process_directory_wrapper, [(d, metric) for d in sub_dirs]))
            else:
                for dirpath in sub_dirs:
                    self.compute_patient_radiomics(dirpath, metric)

    def compute_patient_radiomics(self, dirpath: str | os.PathLike, pet_metric: str) -> None:
        """Computes patient-level radiomics metrics from SUV/SUL and PETseg files.

        Args:
            dirpath (str | os.PathLike): Path to patient sub directory containing
            SUV.nii.gz or SUL.nii.gz, PETseg.nii.gz or PETsegSUL.nii.gz, and patient_info.json.
            pet_metric (str): PET metric to use ("SUV" or "SUL").
        """
        petseg_fname = "PETseg.nii.gz" if pet_metric == "SUV" else "PETsegSUL.nii.gz"
        tumor_stats_key = "TumorStats" if pet_metric == "SUV" else "TumorStatsSUL"

        ct_fpath = os.path.join(dirpath, "CT.nii.gz")
        ctseg_fpath = os.path.join(dirpath, "CTseg.nii.gz")
        suv_fpath = os.path.join(dirpath, f"{pet_metric}.nii.gz")
        petseg_fpath = os.path.join(dirpath, petseg_fname)
        patient_dirpath = os.path.dirname(dirpath)
        patient_info_path = os.path.join(patient_dirpath, "patient_info.json")

        if not os.path.isfile(patient_info_path):
            logger.error(f"Missing patient_info.json in {patient_dirpath}. Radiomics are extracted to csv.")
            std_factor = None
            existing_metrics = {}
        else:
            logger.info(f"Processing: {dirpath}")
            with open(patient_info_path) as json_file:
                patient_info = json.loads(json_file.read())

            study_date = os.path.join(dirpath).split(os.sep)[-1]
            study = patient_info["Studies"].get(study_date, {})
            existing_metrics = study.get(tumor_stats_key, {})
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

        # Append new metrics to the json file
        if new_metrics and os.path.isfile(patient_info_path):
            patient_info["Studies"][study_date][tumor_stats_key] = {**existing_metrics, **new_metrics}
            with open(os.path.join(patient_dirpath, "patient_info.json"), "w") as f:
                json.dump(make_json_safe(patient_info), f)
        elif new_metrics and not os.path.isfile(patient_info_path):
            pd.DataFrame([new_metrics]).to_csv(
                os.path.join(patient_dirpath, "patient_radiomics.csv"),
                index=False,
            )

    def process_directory_wrapper(self, args: tuple) -> None:
        dirpath, pet_metric = args
        try:
            return self.compute_patient_radiomics(dirpath, pet_metric)
        except ValueError:
            return


def radiomics_extraction_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.radiomics_extraction")

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run Radiomics computation."
    )
    parser.add_argument(
        "--input-dirpath-processed", type=str, help="Path to the input folder containing .nii.gz files", required=True
    )
    parser.add_argument(
        "--pet-metric",
        type=str,
        nargs="+",
        choices=["SUV", "SUL"],
        default=["SUV", "SUL"],
        help="PET metric(s) to use as input. Pass one or both: --pet-metric SUV SUL (default: SUV SUL)",
    )
    args = parser.parse_args()

    RadiomicsExtractor(
        input_dirpath_processed=args.input_dirpath_processed,
        pet_metric=args.pet_metric,
    ).run()


if __name__ == "__main__":
    radiomics_extraction_entrypoint()
