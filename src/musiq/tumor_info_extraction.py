import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import cc3d
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from . import metrics, utils
from .radiomics_extraction import (
    DEFAULT_LABEL_GLOB,
    MASK_SOURCES,
    resample_label_to_image_grid,
    resolve_mask,
)

logger = logging.getLogger(__name__)


def _suvpeak_half_kernel(spacing_mm: float) -> int:
    """Half-width (in voxels) of the ~1 cm^3 SUVpeak kernel along one axis.

    Mirrors the kernel sizing in metrics.calculate_suvpeak_median so per-lesion crops can be padded
    enough that the peak neighbourhood is fully contained and SUVpeak matches a full-volume computation.
    """
    num = max(1, int(round(10.0 / spacing_mm)))  # 10 mm = 1 cm
    if num % 2 == 0:
        num += 1
    return num // 2


class TumorInfoExtraction:
    def __init__(
        self,
        input_dirpath_processed: str | os.PathLike,
        pet_metric: str | list[str] | None = None,
        mask_source: str = "auto",
        label_dirpath: str | os.PathLike | None = None,
        label_glob: str = DEFAULT_LABEL_GLOB,
        workers: int = 1,
    ) -> None:
        """Class to handle tumor information extraction from a mask, SUV/SUL, and CTseg files in a specified folder.
        Creates CTsegres.nii.gz if it does not exist.
        The existing patient_info.json is required to extract and save data.
        Expects exactly one CT.nii.gz file per serie.

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the CT.nii.gz file. Can be nested.
            pet_metric (str | list[str] | None): PET metric(s) to use as input.
                Accepts "SUV", "SUL", or both. Defaults to ["SUV", "SUL"].
            mask_source (str): "auto" (PETseg/PETsegSUL -> TumorStats/TumorStatsSUL) or
                "revised" (physician Tumor label -> TumorStatsRevised/TumorStatsRevisedSUL).
            label_dirpath (str | os.PathLike | None): used when mask_source="revised". None looks for the
                label inside each study dir; a path looks under <label_dirpath>/<PatientID>/.
            label_glob (str): filename pattern of the revised label (may contain wildcards), e.g.
                "PETseg_revised.nii" or "*segmentation_Tumor.nii".
            workers (int): number of parallel worker processes. 1 (default) runs serially. Patients are
                the unit of parallelism, so each patient_info.json is only ever written by one worker.
        """
        if pet_metric is None:
            pet_metric = ["SUV", "SUL"]
        pet_metrics = [pet_metric] if isinstance(pet_metric, str) else list(pet_metric)
        for m in pet_metrics:
            if m not in ("SUV", "SUL"):
                raise ValueError(f"pet_metric must be 'SUV' or 'SUL', got '{m}'")
        if mask_source not in MASK_SOURCES:
            raise ValueError(f"mask_source must be one of {MASK_SOURCES}, got '{mask_source}'")
        self.input_dirpath = input_dirpath_processed
        self.pet_metrics = pet_metrics
        self.mask_source = mask_source
        self.label_dirpath = label_dirpath
        self.label_glob = label_glob
        self.workers = max(1, int(workers))
        self.skipped: list[dict] = []  # studies skipped on grid mismatch

    def run(self) -> None:
        for metric in self.pet_metrics:
            study_dirs = []
            required_files = ["CTseg.nii.gz", f"{metric}.nii.gz"]

            for dirpath, dirnames, _filenames in os.walk(self.input_dirpath):
                # Don't descend into non-patient dirs (e.g. cads_staging intermediates).
                dirnames[:] = [d for d in dirnames if d not in utils.RESERVED_PROCESSED_DIRS]
                for subdirname in dirnames:
                    subdirpath = os.path.join(dirpath, subdirname)
                    if (
                        all(os.path.exists(os.path.join(subdirpath, f)) for f in required_files)
                        and resolve_mask(subdirpath, metric, self.mask_source, self.label_dirpath, self.label_glob)[0]
                    ):
                        study_dirs.append(subdirpath)
                        break  # Stop after first matching subdirectory

            if not study_dirs:
                label_loc = "the study dir" if not self.label_dirpath else self.label_dirpath
                mask_desc = (
                    f"a '{self.label_glob}' label in {label_loc}"
                    if self.mask_source == "revised"
                    else ("PETseg.nii.gz" if metric == "SUV" else "PETsegSUL.nii.gz")
                )
                msg = (
                    f"No complete studies found for {metric}. "
                    f"{', '.join(required_files)} plus {mask_desc} are required in each patient/study directory."
                )
                if metric == "SUL":
                    msg += " SUL.nii.gz and PETsegSUL.nii.gz are created by the muscle-fat and autopet tasks — "
                    "make sure both have been run first."
                logger.warning(msg)
                continue

            study_dirs = sorted(study_dirs, key=lambda x: os.path.basename(os.path.dirname(x)))
            # Group by patient dir so each patient_info.json has a single writer
            patient_groups: dict[str, list[str]] = {}
            for d in study_dirs:
                patient_groups.setdefault(os.path.dirname(d), []).append(d)
            group_items = [(dirs, metric) for dirs in patient_groups.values()]
            logger.info(
                "Running %s tumor info for %s on %d studies (%d patients, %d workers).",
                self.mask_source,
                metric,
                len(study_dirs),
                len(group_items),
                self.workers,
            )
            if self.workers > 1:
                with ProcessPoolExecutor(max_workers=self.workers) as executor:
                    for skips in executor.map(self._process_patient_group, group_items):
                        self.skipped.extend(skips)
            else:
                for item in tqdm(group_items, desc=metric):
                    self.skipped.extend(self._process_patient_group(item))

        # Report studies skipped because the Tumor label grid did not match the SUV/SUL image (revised only).
        if self.skipped:
            report_path = os.path.join(self.input_dirpath, "tumor_seg_tumorinfo_skipped.csv")
            pd.DataFrame(self.skipped).drop_duplicates().to_csv(report_path, index=False)
            logger.warning(
                "Skipped %d study/metric pair(s) on grid mismatch; report: %s", len(self.skipped), report_path
            )

    def _process_patient_group(self, args: tuple) -> list[dict]:
        """Process every study dir of one patient serially and return any skip records.

        Grouping by patient keeps each patient_info.json single-writer, so groups can run in parallel.
        """
        study_dirs, pet_metric = args
        skips: list[dict] = []
        for study_dirpath in study_dirs:
            try:
                skip = self.process_study(study_dirpath, pet_metric)
            except Exception as e:  # never let one study kill the whole pool
                logger.error("Error processing %s (%s): %s", study_dirpath, pet_metric, e)
                continue
            if skip:
                skips.append(skip)
        return skips

    def process_study(self, study_dirpath, pet_metric: str = "SUV") -> dict | None:
        """Process a single study directory. Returns a skip record on grid mismatch, else None."""
        logger.info(f"Extracting tumor metrics for {study_dirpath}")
        tumor_stats_key: str
        petseg_fpath, tumor_stats_key = resolve_mask(
            study_dirpath, pet_metric, self.mask_source, self.label_dirpath, self.label_glob
        )
        if petseg_fpath is None:
            logger.warning("No %s mask found for %s; skipping.", self.mask_source, study_dirpath)
            return None
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
        # Revised label must align in world-space; resample if shape differs, skip if disjoint
        if self.mask_source == "revised" and petseg_array.shape != suv_array.shape:
            resampled = resample_label_to_image_grid(petseg_fpath, suv_fpath, study_dirpath)
            if resampled is None or ((petseg_array > 0).any() and not (resampled > 0).any()):
                logger.warning(
                    "Grid mismatch (disjoint world space) for %s: Tumor label %s vs %s %s. Skipping.",
                    study_dirpath,
                    petseg_array.shape,
                    pet_metric,
                    suv_array.shape,
                )
                return {
                    "study_dirpath": study_dirpath,
                    "pet_metric": pet_metric,
                    "label_path": petseg_fpath,
                    "label_shape": str(petseg_array.shape),
                    "image_shape": str(suv_array.shape),
                }
            logger.info(
                "Revised label for %s resampled onto the %s grid (%s -> %s); tumor preserved.",
                study_dirpath,
                pet_metric,
                petseg_array.shape,
                suv_array.shape,
            )
            petseg_array = resampled
        ctsegres_array = metrics.get_3darray_from_niftipath(ctsegres_fpath)
        spacing = utils.get_spacing_from_niftipath(suv_fpath)
        voxel_volume_cc = np.prod(spacing) / 1000

        study_date = study_dirpath.split(os.sep)[-1]

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

        # Per-lesion bbox (cost scales with lesion size); pad >= SUVpeak kernel width + >=1 voxel for marching cubes
        pad = [max(1, _suvpeak_half_kernel(sp)) for sp in spacing]
        bboxes = ndimage.find_objects(labeled_tumors)

        results = []
        for i in range(1, num_lesions + 1):
            bbox = bboxes[i - 1]
            if bbox is None:  # cc3d labels are contiguous 1..N, but stay safe
                continue
            crop = tuple(
                slice(max(0, s.start - p), min(dim, s.stop + p))
                for s, p, dim in zip(bbox, pad, labeled_tumors.shape, strict=True)
            )
            tumor_mask = (labeled_tumors[crop] == i).astype(np.uint8)

            pet_metrics = utils.compute_pet_metrics(tumor_mask, suv_array[crop], spacing)
            if organ_labels:
                organ_overlap = utils.compute_tumor_organ_overlap(tumor_mask, ctsegres_array[crop], organ_labels)
            num_nonzero_voxels = int(tumor_mask.sum())

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
        "--mask-source",
        type=str,
        choices=list(MASK_SOURCES),
        default="auto",
        help="Mask to compute on: 'auto' (PETseg -> TumorStats) or 'revised' (physician label -> "
        "TumorStatsRevised). Default: auto.",
    )
    parser.add_argument(
        "--label-dirpath",
        type=str,
        default=None,
        help="Used with --mask-source revised. Omit to look for the label inside each study dir "
        "(e.g. MULTIPRO PETseg_revised.nii); set to a parallel labels root to look under "
        "<label_dirpath>/<PatientID>/ (e.g. Scheurer labels tree).",
    )
    parser.add_argument(
        "--label-glob",
        type=str,
        default=DEFAULT_LABEL_GLOB,
        help="Filename pattern of the revised label (wildcards allowed), used with --mask-source revised. "
        f"Default: '{DEFAULT_LABEL_GLOB}'. Scheurer uses '*segmentation_Tumor.nii'.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes (patients run in parallel). Default: 1 (serial).",
    )
    args = parser.parse_args()

    TumorInfoExtraction(
        input_dirpath_processed=args.input_dirpath_processed,
        pet_metric=args.pet_metric,
        mask_source=args.mask_source,
        label_dirpath=args.label_dirpath,
        label_glob=args.label_glob,
        workers=args.workers,
    ).run()


if __name__ == "__main__":
    tumor_info_extraction_entrypoint()
