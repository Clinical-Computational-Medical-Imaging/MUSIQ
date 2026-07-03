import json
import logging
import os

import cc3d
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import metrics, utils
from .radiomics_extraction import DEFAULT_LABEL_DIRPATH, resolve_tumor_label

logger = logging.getLogger(__name__)


class TumorInfoExtraction:
    def __init__(
        self,
        input_dirpath_processed: str | os.PathLike,
        pet_metric: str | list[str] | None = None,
        label_dirpath: str | os.PathLike = DEFAULT_LABEL_DIRPATH,
    ) -> None:
        """Class to handle tumor information extraction from PETseg, SUV/SUL, and CTseg files in a specified folder.
        Creates CTsegres.nii.gz if it does not exist.
        The existing patient_info.json is required to extract and save data.
        Expects exactly one CT.nii.gz file per serie.

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the CT.nii.gz file. Can be nested.
            pet_metric (str | list[str] | None): PET metric(s) to use as input.
                Accepts "SUV", "SUL", or both. Defaults to ["SUV", "SUL"].
            label_dirpath (str | os.PathLike): TEMP root of the physician Tumor segmentations
                (<label_dirpath>/<PatientID>/<PatientID> segmentation_Tumor.nii).
        """
        if pet_metric is None:
            pet_metric = ["SUV", "SUL"]
        pet_metrics = [pet_metric] if isinstance(pet_metric, str) else list(pet_metric)
        for m in pet_metrics:
            if m not in ("SUV", "SUL"):
                raise ValueError(f"pet_metric must be 'SUV' or 'SUL', got '{m}'")
        self.input_dirpath = input_dirpath_processed
        self.pet_metrics = pet_metrics
        self.label_dirpath = label_dirpath
        self.skipped: list[dict] = []  # TEMP: studies skipped due to a grid mismatch with the SUV/SUL image

    def run(self) -> None:
        for metric in self.pet_metrics:
            study_dirs = []

            # TEMP: run off the physician Tumor segmentation from label_dirpath (single SUV/PET-space
            # mask used for both metrics).
            required_files = ["CTseg.nii.gz", f"{metric}.nii.gz"]

            for dirpath, dirnames, _filenames in os.walk(self.input_dirpath):
                # Don't descend into non-patient dirs (e.g. cads_staging intermediates).
                dirnames[:] = [d for d in dirnames if d not in utils.RESERVED_PROCESSED_DIRS]
                for subdirname in dirnames:
                    subdirpath = os.path.join(dirpath, subdirname)
                    if all(os.path.exists(os.path.join(subdirpath, f)) for f in required_files) and resolve_tumor_label(
                        subdirpath, self.label_dirpath
                    ):
                        study_dirs.append(subdirpath)
                        break  # Stop after first matching subdirectory

            if not study_dirs:
                msg = (
                    f"No complete studies found for {metric}. "
                    f"{', '.join(required_files)} plus a Tumor segmentation under {self.label_dirpath} "
                    "are required in each patient/study directory."
                )
                if metric == "SUL":
                    msg += " SUL.nii.gz and PETsegSUL.nii.gz are created by the muscle-fat and autopet tasks — "
                    "make sure both have been run first."
                logger.warning(msg)
                continue

            logger.info("Running tumor info extraction for %s on %d studies.", metric, len(study_dirs))
            study_dirs = sorted(study_dirs, key=lambda x: os.path.basename(os.path.dirname(x)))
            for study_dirpath in tqdm(study_dirs, desc=metric):
                self.process_study(study_dirpath, metric)

        # TEMP: report studies skipped because the Tumor label grid did not match the SUV/SUL image.
        if self.skipped:
            report_path = os.path.join(self.input_dirpath, "tumor_seg_tumorinfo_skipped.csv")
            pd.DataFrame(self.skipped).drop_duplicates().to_csv(report_path, index=False)
            logger.warning(
                "Skipped %d study/metric pair(s) on grid mismatch; report: %s", len(self.skipped), report_path
            )

    def process_study(self, study_dirpath, pet_metric: str = "SUV") -> None:
        """Process a single study directory."""
        logger.info(f"Extracting tumor metrics for {study_dirpath}")
        # TEMP: read the physician Tumor segmentation, write per-lesion results to a separate key.
        tumor_stats_key = "TumorStatsRevised" if pet_metric == "SUV" else "TumorStatsRevisedSUL"

        petseg_fpath = resolve_tumor_label(study_dirpath, self.label_dirpath)
        if petseg_fpath is None:
            logger.warning("No Tumor segmentation found for %s under %s; skipping.", study_dirpath, self.label_dirpath)
            return
        ctsegres_fpath = os.path.join(study_dirpath, "CTsegres.nii.gz")
        suv_fpath = os.path.join(study_dirpath, f"{pet_metric}.nii.gz")

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
        suv_array = metrics.get_3darray_from_niftipath(suv_fpath)
        # TEMP: the Tumor label must sit on the same grid as the SUV/SUL image. When the physician
        # segmented a different reconstruction the grids differ — skip and report rather than
        # silently producing misaligned metrics.
        if petseg_array.shape != suv_array.shape:
            logger.warning(
                "Grid mismatch for %s: Tumor label %s vs %s %s. Skipping.",
                study_dirpath,
                petseg_array.shape,
                pet_metric,
                suv_array.shape,
            )
            self.skipped.append(
                {
                    "study_dirpath": study_dirpath,
                    "pet_metric": pet_metric,
                    "label_path": petseg_fpath,
                    "label_shape": str(petseg_array.shape),
                    "image_shape": str(suv_array.shape),
                }
            )
            return
        ctsegres_array = metrics.get_3darray_from_niftipath(ctsegres_fpath)
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
            patient_info["Studies"][study_date].setdefault(tumor_stats_key, {}).update({"Tumors": results})
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
    parser.add_argument(
        "--pet-metric",
        type=str,
        nargs="+",
        choices=["SUV", "SUL"],
        default=["SUV", "SUL"],
        help="PET metric(s) to use as input. Pass one or both: --pet-metric SUV SUL (default: SUV SUL)",
    )
    parser.add_argument(
        "--label-dirpath",
        type=str,
        default=DEFAULT_LABEL_DIRPATH,
        help=(
            "TEMP: root of the physician Tumor segmentations "
            "(<label_dirpath>/<PatientID>/<PatientID> segmentation_Tumor.nii). "
            f"Default: {DEFAULT_LABEL_DIRPATH}"
        ),
    )
    args = parser.parse_args()

    TumorInfoExtraction(
        input_dirpath_processed=args.input_dirpath_processed,
        pet_metric=args.pet_metric,
        label_dirpath=args.label_dirpath,
    ).run()


if __name__ == "__main__":
    tumor_info_extraction_entrypoint()
