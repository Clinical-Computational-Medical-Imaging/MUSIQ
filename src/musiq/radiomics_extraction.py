import glob
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from . import metrics
from .utils import RESERVED_PROCESSED_DIRS, get_spacing_from_niftipath, make_json_safe

logger = logging.getLogger(__name__)

# Radiomics can run off two mask sources, selected via ``mask_source``:
#   "auto"    - the pipeline's automated PETseg.nii.gz / PETsegSUL.nii.gz mask inside the study dir
#               (results -> TumorStats / TumorStatsSUL).
#   "revised" - a physician Tumor segmentation (results -> TumorStatsRevised / TumorStatsRevisedSUL),
#               which must share the SUV/PET grid. Its filename varies by cohort (label_glob) and it
#               lives either inside the study dir (label_dirpath=None, e.g. MULTIPRO's PETseg_revised.nii)
#               or in a parallel tree keyed by PatientID (label_dirpath set, e.g. Scheurer's
#               <label_dirpath>/<PatientID>/<PatientID> segmentation_Tumor.nii).
MASK_SOURCES = ("auto", "revised")
DEFAULT_LABEL_GLOB = "PETseg_revised.nii"


def resolve_mask(
    study_dirpath: str | os.PathLike,
    metric: str,
    mask_source: str,
    label_dirpath: str | os.PathLike | None,
    label_glob: str = DEFAULT_LABEL_GLOB,
) -> tuple[str | None, str]:
    """Return (mask_path, tumor_stats_key) for a study dir given the mask source and PET metric.

    mask_path is None when a "revised" label cannot be found for the study.
    """
    if mask_source == "revised":
        key = "TumorStatsRevised" if metric == "SUV" else "TumorStatsRevisedSUL"
        return resolve_tumor_label(study_dirpath, label_dirpath, label_glob), key
    fname = "PETseg.nii.gz" if metric == "SUV" else "PETsegSUL.nii.gz"
    key = "TumorStats" if metric == "SUV" else "TumorStatsSUL"
    mask_path = os.path.join(study_dirpath, fname)
    return (mask_path if os.path.exists(mask_path) else None), key


def resolve_tumor_label(
    study_dirpath: str | os.PathLike,
    label_dirpath: str | os.PathLike | None,
    label_glob: str = DEFAULT_LABEL_GLOB,
) -> str | None:
    """Locate the physician Tumor segmentation for a processed study directory. Returns None if absent.

    When ``label_dirpath`` is None the label is looked up inside the study dir itself; otherwise it is
    looked up under ``<label_dirpath>/<PatientID>/`` (PatientID = the study dir's parent). ``label_glob``
    is the filename pattern (may contain wildcards); the first match wins.
    """
    if label_dirpath:
        patient_id = os.path.basename(os.path.dirname(study_dirpath))
        search_dir = os.path.join(label_dirpath, patient_id)
    else:
        search_dir = study_dirpath
    matches = sorted(glob.glob(os.path.join(search_dir, label_glob)))
    return matches[0] if matches else None


class RadiomicsExtractor:
    def __init__(
        self,
        input_dirpath_processed: str | os.PathLike,
        pet_metric: str | list[str] | None = None,
        mask_source: str = "auto",
        label_dirpath: str | os.PathLike | None = None,
        label_glob: str = DEFAULT_LABEL_GLOB,
        workers: int = 1,
    ) -> None:
        """Calculates patient-level statistics from PET and segmentation images.
        Expects exactly one PT and matching CT series per study date

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the SUV/SUL, PETseg and PET.json files.
            Can be nested.
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
        self.skipped: list[dict] = []  # studies skipped due to a grid mismatch with the SUV/SUL image

    def run(self) -> None:
        for metric in self.pet_metrics:
            necessary_files = [f"{metric}.nii.gz", "CTseg.nii.gz", "CT.nii.gz"]
            sub_dirs = []
            for dirpath, dirnames, filenames in os.walk(self.input_dirpath):
                # Don't descend into non-patient dirs (e.g. cads_staging intermediates).
                dirnames[:] = [d for d in dirnames if d not in RESERVED_PROCESSED_DIRS]
                if (
                    all(f in filenames for f in necessary_files)
                    and resolve_mask(dirpath, metric, self.mask_source, self.label_dirpath, self.label_glob)[0]
                ):
                    sub_dirs.append(dirpath)
            if not sub_dirs:
                label_loc = "the study dir" if not self.label_dirpath else self.label_dirpath
                mask_desc = (
                    f"a '{self.label_glob}' label in {label_loc}"
                    if self.mask_source == "revised"
                    else ("PETseg.nii.gz" if metric == "SUV" else "PETsegSUL.nii.gz")
                )
                msg = f"No directories found with necessary files for {metric}: {necessary_files} plus {mask_desc}."
                if metric == "SUL":
                    msg += (
                        " SUL.nii.gz and PETsegSUL.nii.gz are created by the muscle-fat and autopet tasks"
                        " — make sure both have been run first."
                    )
                logger.warning(msg)
                continue

            sub_dirs = sorted(sub_dirs, key=lambda x: os.path.basename(os.path.dirname(x)))
            # Group study dirs by patient dir so each patient_info.json is only ever touched by one
            # worker — this is what makes multiprocessing safe.
            patient_groups: dict[str, list[str]] = {}
            for d in sub_dirs:
                patient_groups.setdefault(os.path.dirname(d), []).append(d)
            group_items = [(dirs, metric) for dirs in patient_groups.values()]
            logger.info(
                "Running %s radiomics for %s on %d studies (%d patients, %d workers).",
                self.mask_source,
                metric,
                len(sub_dirs),
                len(group_items),
                self.workers,
            )
            if self.workers > 1:
                with ProcessPoolExecutor(max_workers=self.workers) as executor:
                    for skips in executor.map(self._process_patient_group, group_items):
                        self.skipped.extend(skips)
            else:
                for item in group_items:
                    self.skipped.extend(self._process_patient_group(item))

        # Report studies skipped because the Tumor label grid did not match the SUV/SUL image (revised only).
        if self.skipped:
            report_path = os.path.join(self.input_dirpath, "tumor_seg_radiomics_skipped.csv")
            pd.DataFrame(self.skipped).drop_duplicates().to_csv(report_path, index=False)
            logger.warning(
                "Skipped %d study/metric pair(s) on grid mismatch; report: %s", len(self.skipped), report_path
            )

    def compute_patient_radiomics(self, dirpath: str | os.PathLike, pet_metric: str) -> dict | None:
        """Computes patient-level radiomics metrics from SUV/SUL and the selected mask.

        Args:
            dirpath (str | os.PathLike): Path to patient sub directory containing
            SUV.nii.gz or SUL.nii.gz, the mask (PETseg / label), and patient_info.json.
            pet_metric (str): PET metric to use ("SUV" or "SUL").

        Returns:
            dict | None: a skip record when the study is skipped on a grid mismatch, else None.
        """
        tumor_stats_key: str
        petseg_fpath, tumor_stats_key = resolve_mask(
            dirpath, pet_metric, self.mask_source, self.label_dirpath, self.label_glob
        )

        ct_fpath = os.path.join(dirpath, "CT.nii.gz")
        ctseg_fpath = os.path.join(dirpath, "CTseg.nii.gz")
        suv_fpath = os.path.join(dirpath, f"{pet_metric}.nii.gz")
        if petseg_fpath is None:
            logger.warning("No %s mask found for %s; skipping.", self.mask_source, dirpath)
            return None
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
        # A revised (physician) label must sit on the same grid as the SUV/SUL image. When the physician
        # segmented a different reconstruction the grids differ — skip and report rather than silently
        # producing misaligned radiomics. (Automated PETseg masks share the grid by construction.)
        if self.mask_source == "revised" and gtarray.shape != ptarray.shape:
            logger.warning(
                "Grid mismatch for %s: Tumor label %s vs %s %s. Skipping.",
                dirpath,
                gtarray.shape,
                pet_metric,
                ptarray.shape,
            )
            return {
                "study_dirpath": dirpath,
                "pet_metric": pet_metric,
                "label_path": petseg_fpath,
                "label_shape": str(gtarray.shape),
                "image_shape": str(ptarray.shape),
            }
        spacing = get_spacing_from_niftipath(suv_fpath)

        # Dmax and SDmax share the same (dissemination) computation, so memoize it to compute it at
        # most once per study even when both metrics are missing, while keeping each metric's own skip logic.
        _dmax_cache: dict = {}

        def _dmax() -> np.float64:
            if "v" not in _dmax_cache:
                _dmax_cache["v"] = metrics.calculate_patient_level_dissemination(gtarray, spacing)
            return _dmax_cache["v"]

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
            "Dmax": lambda: _dmax(),
            "SDmax": lambda: _dmax() / std_factor if std_factor else "NAN",
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

    def _process_patient_group(self, args: tuple) -> list[dict]:
        """Process every study dir of one patient serially and return any skip records.

        Grouping by patient keeps each patient_info.json single-writer, so groups can run in parallel.
        """
        study_dirs, pet_metric = args
        skips: list[dict] = []
        for dirpath in study_dirs:
            try:
                skip = self.compute_patient_radiomics(dirpath, pet_metric)
            except Exception as e:  # never let one study kill the whole pool
                logger.error("Error processing %s (%s): %s", dirpath, pet_metric, e)
                continue
            if skip:
                skips.append(skip)
        return skips


def radiomics_extraction_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.radiomics_extraction")

    import argparse

    parser = argparse.ArgumentParser(description="Recursively run Radiomics computation.")
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

    RadiomicsExtractor(
        input_dirpath_processed=args.input_dirpath_processed,
        pet_metric=args.pet_metric,
        mask_source=args.mask_source,
        label_dirpath=args.label_dirpath,
        label_glob=args.label_glob,
        workers=args.workers,
    ).run()


if __name__ == "__main__":
    radiomics_extraction_entrypoint()
