import json
import logging
import os
import pathlib as plb
import pickle
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

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

from .utils import list_patient_dirs

setup_nnunet_env()
logger = logging.getLogger(__name__)

VALID_TASK_IDS = {551, 552, 553, 554, 555, 556, 557, 558, 559}


def _expand_task_deps(task_ids: list[int]) -> list[int]:
    """Expand task_ids with required dependencies (557/558 need 553; 558 also needs 552)."""
    ids = sorted(task_ids)
    if any(t in ids for t in (557, 558)) and 553 not in ids:
        ids.append(553)
    if 558 in ids and 552 not in ids:
        ids.append(552)
    return sorted(ids)


# Combined-labelmap inverse (class name -> global label id)
_LABELMAP_INV = {v: k for k, v in labelmap_all_structure.items()}


def _restore_one_case(job: dict) -> dict:
    """Restore + combine a single CADS case into its ``CTcads.nii.gz``.

    Module-level (hence picklable) so it can run in a ProcessPoolExecutor worker. Returns a result
    dict with a ``status`` in {exists, no_seg, no_metadata, no_parts, error, wrote}.
    """
    case = job["case"]
    dirpath = job["dirpath"]
    seg_dir = job["seg_dir"]
    metadata_dir = job["metadata_dir"]
    preprocessed_dir = job["preprocessed_dir"]
    expanded_tasks = job["expanded_tasks"]
    num_threads = job["num_threads"]

    output_fpath = os.path.join(dirpath, "CTcads.nii.gz")
    result = {"case": case, "dirpath": dirpath, "study_date": job["study_date"], "output_fpath": output_fpath}
    if os.path.isfile(output_fpath):
        return {**result, "status": "exists"}

    case_seg_dir = os.path.join(seg_dir, case)
    metadata_path = os.path.join(metadata_dir, f"{case}_metadata.pkl")
    if not os.path.isdir(case_seg_dir):
        return {**result, "status": "no_seg"}
    if not os.path.isfile(metadata_path):
        return {**result, "status": "no_metadata"}

    try:
        with open(metadata_path, "rb") as f:
            metadata_orig = pickle.load(f)

        for seg_file in os.listdir(case_seg_dir):
            if not seg_file.endswith(".nii.gz"):
                continue  # skip *_ERROR.log etc.
            seg_path = os.path.join(case_seg_dir, seg_file)
            restore_seg_in_orig_format(seg_path, seg_path, metadata_orig, num_threads_preprocessing=num_threads)

        # Combine parts into labelmap_all_structure; read via dataobj (native int) not get_fdata (float64)
        seg_combined = None
        affine = None
        for task_id in expanded_tasks:
            part_path = os.path.join(case_seg_dir, f"{case}_part_{task_id}.nii.gz")
            if not os.path.isfile(part_path):
                continue
            part_img = nib.load(part_path)
            seg = np.asanyarray(part_img.dataobj)
            if seg_combined is None:
                seg_combined = np.zeros(seg.shape, dtype=np.uint8)
                affine = part_img.affine
            for class_index, class_name in map_taskid_to_labelmaps[task_id].items():
                if class_name in except_labels_combine:
                    continue
                seg_combined[seg == class_index] = _LABELMAP_INV[class_name]

        if seg_combined is None:
            return {**result, "status": "no_parts"}

        nib.save(nib.Nifti1Image(seg_combined, affine), output_fpath)
    except Exception as e:  # don't let one bad case kill the pool
        return {**result, "status": "error", "error": repr(e)}

    # Clean up this case's intermediates
    for path in (os.path.join(preprocessed_dir, f"{case}.nii.gz"), metadata_path):
        if os.path.isfile(path):
            os.remove(path)
    shutil.rmtree(case_seg_dir, ignore_errors=True)
    return {**result, "status": "wrote"}


class CadsInference:
    """Staged CADS pipeline (preprocess CT -> run models -> restore+combine into CTcads.nii.gz),
    split so CPU/GPU stages run separately.
    """

    def __init__(
        self,
        input_dirpath_processed: str | os.PathLike,
        tasks: list[str] | None = None,
        work_dir: str | os.PathLike | None = None,
        use_cpu: bool = False,
        num_threads_preprocessing: int = 4,
        nr_threads_saving: int = 4,
        restore_workers: int | None = None,
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
            restore_workers: Number of parallel worker processes for the restore stage. Defaults to
                None → SLURM_CPUS_PER_TASK (or os.cpu_count()). The restore bottleneck is per-case
                resampling, which parallelizes cleanly across cases.
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
        self.restore_workers = restore_workers

    @staticmethod
    def _case_id(patient_id: str, study_date: str) -> str:
        return f"{patient_id}__{study_date}"

    def _iter_ct_studies(self):
        """Yield (study_dirpath, patient_id, study_date) for every CT.nii.gz in the tree."""
        if not os.path.isdir(self.input_dirpath):
            logger.error(f"Error: {self.input_dirpath} is not a valid directory.")
            return
        top_dirs = list_patient_dirs(self.input_dirpath)
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
                # preprocess skips writing when CT is already 1.5mm RAS w/ zero origin; flag rather than silently drop
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
        expanded_tasks = _expand_task_deps(self.task_ids)
        cases = self._case_to_study()

        # Parallelize across cases; peak memory scales with concurrent cases
        n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 4)
        n_workers = max(1, self.restore_workers or n_cpus)
        per_worker_threads = max(1, n_cpus // n_workers)

        jobs = [
            {
                "case": case,
                "dirpath": dirpath,
                "study_date": study_date,
                "seg_dir": self.seg_dir,
                "metadata_dir": self.metadata_dir,
                "preprocessed_dir": self.preprocessed_dir,
                "expanded_tasks": expanded_tasks,
                "num_threads": per_worker_threads,
            }
            for case, (dirpath, _patient_id, study_date) in cases.items()
        ]
        logger.info(
            f"[CADS restore] {len(jobs)} case(s) across {n_workers} worker(s) "
            f"({per_worker_threads} thread(s) each, {n_cpus} CPUs)."
        )

        counts: dict[str, int] = {}

        def _handle(res: dict) -> None:
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            case = res["case"]
            status = res["status"]
            if status == "wrote":
                logger.info(f"Wrote {res['output_fpath']}.")
                # Update patient_info.json serially in the parent (studies of one patient share it)
                self._update_patient_info(res["dirpath"], res["study_date"], res["output_fpath"])
            elif status == "exists":
                logger.info(f"CTcads.nii.gz already exists for {case}, skipping restore.")
            elif status == "no_seg":
                logger.warning(f"No segmentation dir for {case}; run inference first. Skipping.")
            elif status == "no_metadata":
                logger.warning(f"No metadata for {case}; cannot restore. Skipping.")
            elif status == "no_parts":
                logger.error(f"No part files could be combined for {case}; CTcads.nii.gz not written.")
            elif status == "error":
                logger.error(f"Restore failed for {case}: {res.get('error')}")

        if n_workers == 1:
            for job in jobs:
                _handle(_restore_one_case(job))
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for res in executor.map(_restore_one_case, jobs):
                    _handle(res)

        logger.info(f"[CADS restore] Done. Summary: {counts}")

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
    parser.add_argument(
        "--cads-restore-workers",
        type=int,
        default=None,
        help="Parallel worker processes for restore. Default: SLURM_CPUS_PER_TASK (or CPU count).",
    )
    args = parser.parse_args()
    CadsInference(
        args.input_dirpath_processed,
        tasks=args.cads_tasks,
        work_dir=args.cads_work_dir,
        restore_workers=args.cads_restore_workers,
    ).restore()


if __name__ == "__main__":
    cads_inference_entrypoint()
