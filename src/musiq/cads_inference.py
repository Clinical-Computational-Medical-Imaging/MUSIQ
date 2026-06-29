import json
import logging
import os
import pathlib as plb
import pickle
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CADS"))

import nibabel as nib
import numpy as np
from cads.dataset_utils.bodyparts_labelmaps import (
    except_labels_combine,
    labelmap_all_structure,
    map_taskid_to_labelmaps,
)
from cads.dataset_utils.preprocessing import preprocess_nifti_and_save_to_dir, restore_seg_in_orig_format
from cads.utils.inference import predict_preprocessed_images
from cads.utils.libs import check_or_download_model_weights, get_model_weights_dir, setup_nnunet_env

from .utils import natural_key

setup_nnunet_env()
logger = logging.getLogger(__name__)

VALID_TASK_IDS = {551, 552, 553, 554, 555, 556, 557, 558, 559}


def _expand_task_deps(task_ids: list[int]) -> list[int]:
    """Add the dependency tasks CADS needs internally (mirrors predict_preprocessed_images /
    predict): groups 557/558 need 553 (brain) and 558 also needs 552 (spine). The combined
    output is built over this expanded set, exactly as the monolithic predict() does."""
    ids = sorted(task_ids)
    if any(t in ids for t in (557, 558)) and 553 not in ids:
        ids.append(553)
    if 558 in ids and 552 not in ids:
        ids.append(552)
    return sorted(ids)


class CadsInference:
    """Staged CADS pipeline over the MUSIQ processed tree.

    CADS is split into three standalone stages so CPU and GPU work can run as separate
    jobs (see https://github.com/murong-xu/CADS Option 2). Intermediate artifacts are kept
    in a staging directory (default ``<processed>/cads_staging``) keyed by a per-study case
    id ``<patient_id>__<study_date>`` and auto-removed once a study's ``CTcads.nii.gz`` is
    produced:

    1. :meth:`preprocess` (CPU)  — reorient/resample each ``CT.nii.gz`` to 1.5 mm RAS and write
       a ``<case>.nii.gz`` plus ``<case>_metadata.pkl``.
    2. :meth:`inference` (GPU)   — run the CADS models on the preprocessed images, producing
       per-task ``<case>_part_55X.nii.gz`` masks in preprocessed space.
    3. :meth:`restore` (CPU)     — restore each part to the original geometry, combine them into
       a single ``CTcads.nii.gz`` (labelmap_all_structure labels), update ``patient_info.json``
       and clean up the case's intermediates.

    :meth:`run` executes all three in order (used by the unified ``cads`` workflow task).
    """

    def __init__(
        self,
        input_dirpath_processed: str | os.PathLike,
        tasks: list[str] | None = None,
        work_dir: str | os.PathLike | None = None,
        use_cpu: bool = False,
        num_threads_preprocessing: int = 4,
        nr_threads_saving: int = 4,
    ) -> None:
        """
        Args:
            input_dirpath_processed: Processed output tree (processed/<patient>/<study_date>/).
            tasks: CADS task ids as strings, or "all". Defaults to all 551-559.
            work_dir: Staging directory for intermediates. Defaults to <processed>/cads_staging.
                Must be identical across the three stages when run as separate jobs (the default
                is derived from input_dirpath_processed, so separate jobs agree automatically).
            use_cpu: Run inference on CPU instead of GPU.
            num_threads_preprocessing: Threads for (pre)processing/restoration.
            nr_threads_saving: Threads for saving segmentations.
        """
        self.input_dirpath = input_dirpath_processed

        tasks = tasks or []
        valid_str_tasks = {str(t) for t in VALID_TASK_IDS} | {"all", ""}
        if any(t not in valid_str_tasks for t in tasks):
            logger.error(
                "Wrong input tasks for CADS model. Please use one/ some of "
                "the following separated by spaces: all 551 552 553 554 555 556 557 558 559"
            )
            sys.exit(1)
        if not tasks or "all" in tasks:
            self.task_ids = sorted(VALID_TASK_IDS)
        else:
            self.task_ids = sorted(int(t) for t in tasks if t)

        self.work_dir = os.fspath(work_dir) if work_dir else os.path.join(self.input_dirpath, "cads_staging")
        self.preprocessed_dir = os.path.join(self.work_dir, "preprocessed")
        self.metadata_dir = os.path.join(self.work_dir, "metadata")
        self.seg_dir = os.path.join(self.work_dir, "seg_preprocessed")
        self.use_cpu = use_cpu
        self.num_threads_preprocessing = num_threads_preprocessing
        self.nr_threads_saving = nr_threads_saving

    @staticmethod
    def _case_id(patient_id: str, study_date: str) -> str:
        return f"{patient_id}__{study_date}"

    def _iter_ct_studies(self):
        """Yield (study_dirpath, patient_id, study_date) for every CT.nii.gz in the tree."""
        if not os.path.isdir(self.input_dirpath):
            logger.error(f"Error: {self.input_dirpath} is not a valid directory.")
            return
        top_dirs = [d for d in os.listdir(self.input_dirpath) if os.path.isdir(os.path.join(self.input_dirpath, d))]
        top_dirs.sort(key=natural_key)
        for top_dir in top_dirs:
            top_dir_path = os.path.join(self.input_dirpath, top_dir)
            for dirpath, dirnames, filenames in os.walk(top_dir_path):
                rel_parts = plb.Path(os.path.relpath(dirpath, self.input_dirpath)).parts
                if len(rel_parts) != 2:
                    continue
                dirnames.clear()
                patient_id, study_date = rel_parts
                if "CT.nii.gz" in filenames:
                    yield dirpath, patient_id, study_date

    def _case_to_study(self) -> dict[str, tuple[str, str, str]]:
        """Map case id -> (study_dirpath, patient_id, study_date) for restoration/placement."""
        return {
            self._case_id(patient_id, study_date): (dirpath, patient_id, study_date)
            for dirpath, patient_id, study_date in self._iter_ct_studies()
        }

    # ------------------------------------------------------------------ stage 1: preprocess (CPU)
    def preprocess(self) -> None:
        logger.info(f"[CADS preprocess] Preprocessing CTs in {self.input_dirpath}")
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)

        for dirpath, patient_id, study_date in self._iter_ct_studies():
            case = self._case_id(patient_id, study_date)
            if os.path.isfile(os.path.join(dirpath, "CTcads.nii.gz")):
                logger.info(f"CTcads.nii.gz already exists for {case}, skipping preprocessing.")
                continue
            pre_img = os.path.join(self.preprocessed_dir, f"{case}.nii.gz")
            if os.path.isfile(pre_img):
                logger.info(f"Preprocessed image already exists for {case}, skipping.")
                continue

            ct_path = os.path.join(dirpath, "CT.nii.gz")
            logger.info(f"Preprocessing {case}.")
            preprocess_nifti_and_save_to_dir(
                ct_path,
                self.preprocessed_dir,
                self.metadata_dir,
                case,
                spacing=1.5,
                num_threads_preprocessing=self.num_threads_preprocessing,
            )
            if not os.path.isfile(pre_img):
                # preprocess_nifti_and_save_to_dir skips writing when the CT is already at
                # 1.5 mm RAS with zero origin (very rare for clinical CT). Without it the later
                # stages have no input/metadata, so flag the case rather than silently dropping it.
                logger.warning(
                    f"No preprocessed image written for {case} (CT may already be 1.5 mm RAS); "
                    "this case will be skipped by inference/restore."
                )

    # ------------------------------------------------------------------ stage 2: inference (GPU)
    def inference(self) -> None:
        logger.info(f"[CADS inference] Running CADS models (use_cpu={self.use_cpu})")
        if not os.path.isdir(self.preprocessed_dir):
            logger.error(f"Preprocessed dir {self.preprocessed_dir} missing — run the preprocess stage first.")
            return
        os.makedirs(self.seg_dir, exist_ok=True)

        case_to_study = self._case_to_study()
        to_run = []
        for fname in sorted(os.listdir(self.preprocessed_dir)):
            if not fname.endswith(".nii.gz"):
                continue
            case = fname[:-7]
            study = case_to_study.get(case)
            if study and os.path.isfile(os.path.join(study[0], "CTcads.nii.gz")):
                logger.info(f"CTcads.nii.gz already exists for {case}, skipping inference.")
                continue
            if self._case_segmentation_complete(case):
                logger.info(f"Segmentations already exist for {case}, skipping inference.")
                continue
            to_run.append(os.path.join(self.preprocessed_dir, fname))

        if not to_run:
            logger.info("[CADS inference] Nothing to do.")
            return

        model_folder = get_model_weights_dir()
        for task_id in _expand_task_deps(self.task_ids):
            check_or_download_model_weights(task_id)

        predict_preprocessed_images(
            to_run,
            self.seg_dir,
            model_folder,
            self.task_ids,
            folds="all",
            use_cpu=self.use_cpu,
            postprocess_cads=True,
            num_threads_preprocessing=self.num_threads_preprocessing,
            nr_threads_saving=self.nr_threads_saving,
            mode="auto",
        )

    def _case_segmentation_complete(self, case: str) -> bool:
        """True if every expected part file for a case already exists (idempotent re-runs)."""
        case_seg_dir = os.path.join(self.seg_dir, case)
        if not os.path.isdir(case_seg_dir):
            return False
        return all(
            os.path.isfile(os.path.join(case_seg_dir, f"{case}_part_{task_id}.nii.gz"))
            for task_id in _expand_task_deps(self.task_ids)
        )

    # ------------------------------------------------------------------ stage 3: restore + combine (CPU)
    def restore(self) -> None:
        logger.info("[CADS restore] Restoring to original geometry and combining segmentations.")
        labelmap_all_structure_inv = {v: k for k, v in labelmap_all_structure.items()}
        expanded_tasks = _expand_task_deps(self.task_ids)

        for case, (dirpath, _patient_id, study_date) in self._case_to_study().items():
            output_fpath = os.path.join(dirpath, "CTcads.nii.gz")
            if os.path.isfile(output_fpath):
                logger.info(f"CTcads.nii.gz already exists for {case}, skipping restore.")
                continue

            case_seg_dir = os.path.join(self.seg_dir, case)
            metadata_path = os.path.join(self.metadata_dir, f"{case}_metadata.pkl")
            if not os.path.isdir(case_seg_dir):
                logger.warning(f"No segmentation dir for {case}; run inference first. Skipping.")
                continue
            if not os.path.isfile(metadata_path):
                logger.warning(f"No metadata for {case} at {metadata_path}; cannot restore. Skipping.")
                continue

            with open(metadata_path, "rb") as f:
                metadata_orig = pickle.load(f)

            # Restore every part file to the original image geometry (in place).
            for seg_file in os.listdir(case_seg_dir):
                if not seg_file.endswith(".nii.gz"):
                    continue  # skip *_ERROR.log etc.
                seg_path = os.path.join(case_seg_dir, seg_file)
                restore_seg_in_orig_format(
                    seg_path, seg_path, metadata_orig, num_threads_preprocessing=self.num_threads_preprocessing
                )

            # Combine all task parts into one labelmap_all_structure segmentation.
            seg_combined = None
            affine = None
            for task_id in expanded_tasks:
                part_path = os.path.join(case_seg_dir, f"{case}_part_{task_id}.nii.gz")
                if not os.path.isfile(part_path):
                    logger.warning(f"Missing part file {part_path} for {case}; skipping that task in combine.")
                    continue
                part_img = nib.load(part_path)
                seg = part_img.get_fdata()
                if seg_combined is None:
                    seg_combined = np.zeros(seg.shape, dtype=np.uint8)
                    affine = part_img.affine
                for class_index, class_name in map_taskid_to_labelmaps[task_id].items():
                    if class_name in except_labels_combine:
                        continue
                    seg_combined[seg == class_index] = labelmap_all_structure_inv[class_name]

            if seg_combined is None:
                logger.error(f"No part files could be combined for {case}; CTcads.nii.gz not written.")
                continue

            nib.save(nib.Nifti1Image(seg_combined, affine), output_fpath)
            logger.info(f"Wrote {output_fpath}.")

            self._update_patient_info(dirpath, study_date, output_fpath)
            self._cleanup_case(case, case_seg_dir, metadata_path)

    def _update_patient_info(self, dirpath: str, study_date: str, output_fpath: str) -> None:
        patient_dirpath = os.path.dirname(dirpath)
        patient_info_path = os.path.join(patient_dirpath, "patient_info.json")
        seg_metadata = {
            "settings": {"out": output_fpath, "task": self.task_ids, "staged": True},
            "model": "cads v1.0.0",
        }
        if not os.path.isfile(patient_info_path):
            logger.error(f"Missing patient_info.json in {patient_dirpath}.")
            with open(f"{output_fpath[:-7]}_seg.json", "w") as f:
                json.dump(seg_metadata, f)
            return
        with open(patient_info_path) as json_file:
            patient_info = json.load(json_file)
        try:
            series_name = next(iter(patient_info["Studies"][study_date]["Modalities"]["CT"][0]))
            ct_entry = patient_info["Studies"][study_date]["Modalities"]["CT"][0][series_name]
        except (KeyError, IndexError, StopIteration):
            logger.error(f"Could not locate CT series slot for {study_date} in {patient_info_path}.")
            return
        ct_entry.update({"CTcadsPath": output_fpath, "CTcads_metadata": seg_metadata})
        with open(patient_info_path, "w") as f:
            json.dump(patient_info, f)

    def _cleanup_case(self, case: str, case_seg_dir: str, metadata_path: str) -> None:
        """Remove a case's intermediates once its CTcads.nii.gz exists (disk-saving auto-clean)."""
        for path in (
            os.path.join(self.preprocessed_dir, f"{case}.nii.gz"),
            metadata_path,
        ):
            if os.path.isfile(path):
                os.remove(path)
        shutil.rmtree(case_seg_dir, ignore_errors=True)

    def run(self) -> None:
        """Run all three stages in sequence (single-node path used by the `cads` workflow task)."""
        self.preprocess()
        self.inference()
        self.restore()


def _add_common_args(parser) -> None:
    parser.add_argument(
        "--input-dirpath-processed", type=str, required=True, help="Path to the processed folder with CT.nii.gz files."
    )
    parser.add_argument(
        "--cads-tasks",
        nargs="+",
        default=None,
        help="CADS tasks (space-separated): all, 551, 552, 553, 554, 555, 556, 557, 558, 559.",
    )
    parser.add_argument(
        "--cads-work-dir",
        type=str,
        default=None,
        help="Staging dir for intermediates (must match across stages). Default: <processed>/cads_staging.",
    )


def cads_preprocess_entrypoint() -> None:
    """Standalone CPU stage: preprocess CTs for CADS."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.cads_preprocess")

    import argparse

    parser = argparse.ArgumentParser(description="CADS staged pipeline — stage 1: preprocess CTs (CPU).")
    _add_common_args(parser)
    args = parser.parse_args()
    CadsInference(args.input_dirpath_processed, tasks=args.cads_tasks, work_dir=args.cads_work_dir).preprocess()


def cads_inference_entrypoint() -> None:
    """Standalone GPU stage: run CADS models on preprocessed images."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.cads_inference")

    import argparse

    parser = argparse.ArgumentParser(
        description="CADS staged pipeline — stage 2: model inference on preprocessed CTs (GPU recommended). "
        "Run stage 1 (musiq_cads_preprocess) first."
    )
    _add_common_args(parser)
    parser.add_argument("--cpu", action="store_true", help="Run inference on CPU instead of GPU.")
    args = parser.parse_args()
    CadsInference(
        args.input_dirpath_processed, tasks=args.cads_tasks, work_dir=args.cads_work_dir, use_cpu=args.cpu
    ).inference()


def cads_restore_entrypoint() -> None:
    """Standalone CPU stage: restore to original geometry, combine into CTcads.nii.gz."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.cads_restore")

    import argparse

    parser = argparse.ArgumentParser(
        description="CADS staged pipeline — stage 3: restore segmentations to original geometry and "
        "combine into CTcads.nii.gz (CPU). Run stages 1 and 2 first."
    )
    _add_common_args(parser)
    args = parser.parse_args()
    CadsInference(args.input_dirpath_processed, tasks=args.cads_tasks, work_dir=args.cads_work_dir).restore()


if __name__ == "__main__":
    cads_inference_entrypoint()
