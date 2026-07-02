import json
import logging
import os
import pathlib as plb
import shutil
import subprocess

from .utils import list_patient_dirs

logger = logging.getLogger(__name__)

# Names BOA writes into its --output-dir for the BCA segmentation maps. They are
# moved into the study root with these MUSIQ-style names. ``total.nii.gz`` is the
# reused TotalSegmentator organ map and is intentionally excluded (it is the seed,
# not a BCA product).
BCA_SEG_RENAME = {
    "tissues.nii.gz": "CTbca_tissues.nii.gz",
    "body_regions.nii.gz": "CTbca_body_regions.nii.gz",
}


class BoaInference:
    def __init__(
        self,
        input_dirpath_processed: str | os.PathLike,
        weights_dirpath: str | os.PathLike | None = None,
        image: str = "shipai/boa-cli",
        fast_bca: bool = False,
        no_pdf: bool = True,
        device: str = "gpu",
        reuse_total: bool = True,
    ) -> None:
        """Run the UMEssen BOA Body Composition Analysis (BCA) on every ``CT.nii.gz``
        in the processed tree by shelling out to the ``shipai/boa-cli`` Docker image.

        BOA's ``--input-image`` accepts a single NIfTI, so MUSIQ's existing
        ``CT.nii.gz`` is fed directly. BCA depends on TotalSegmentator's ``total``
        model; when ``reuse_total`` is set and MUSIQ already produced ``CTseg.nii.gz``
        it is seeded as ``total.nii.gz`` in BOA's output dir so BOA skips recomputing
        it (BOA reuses existing segmentations unless ``--force-recompute`` is passed).

        The BCA tissue/body-region models have no MUSIQ equivalent and always run.

        Args:
            input_dirpath_processed: Processed output tree (processed/<patient>/<study_date>/).
            weights_dirpath: Local BOA/TotalSegmentator weights dir, mounted at /app/weights.
                If None, BOA downloads weights on first inference.
            image: BOA Docker image tag.
            fast_bca: Pass --fast-bca (single-fold instead of 5-fold ensemble).
            no_pdf: Pass --bca-no-pdf (skip the PDF report, keep JSON measurements).
            device: BOA --device value ("gpu", "cuda" or "cpu").
            reuse_total: Seed CTseg.nii.gz as total.nii.gz so BOA skips the total model.
        """
        self.input_dirpath = input_dirpath_processed
        self.weights_dirpath = weights_dirpath
        self.image = image
        self.fast_bca = fast_bca
        self.no_pdf = no_pdf
        self.device = device
        self.reuse_total = reuse_total

    def run(self) -> None:
        if not os.path.isdir(self.input_dirpath):
            logger.error(f"Error: {self.input_dirpath} is not a valid directory.")
            return

        logger.info(f"Starting BOA BCA inference in {self.input_dirpath}")
        top_dirs = list_patient_dirs(self.input_dirpath)

        for top_dir in top_dirs:
            top_dir_path = os.path.join(self.input_dirpath, top_dir)

            for dirpath, dirnames, filenames in os.walk(top_dir_path):
                rel_parts = plb.Path(os.path.relpath(dirpath, self.input_dirpath)).parts
                if len(rel_parts) != 2:
                    continue
                dirnames.clear()
                patient_id, study_date = rel_parts

                if "CT.nii.gz" not in filenames:
                    continue

                self._process_study(dirpath, patient_id, study_date)

    def _process_study(self, dirpath: str, patient_id: str, study_date: str) -> None:
        # Idempotency: skip if the BCA tissue map already lives next to CT.nii.gz.
        final_seg = os.path.join(dirpath, BCA_SEG_RENAME["tissues.nii.gz"])
        if os.path.isfile(final_seg):
            logger.info(f"BCA outputs already exist for {patient_id} {study_date}, skipping.")
            return

        work_dir = os.path.join(dirpath, "boa")
        os.makedirs(work_dir, exist_ok=True)

        # Seed the reused TotalSegmentator organ map so BOA skips the `total` model.
        ctseg_path = os.path.join(dirpath, "CTseg.nii.gz")
        if self.reuse_total:
            if os.path.isfile(ctseg_path):
                self._seed_total(work_dir, ctseg_path)
            else:
                logger.warning(
                    f"reuse_total set but CTseg.nii.gz missing for {patient_id} {study_date}; "
                    "BOA will compute the total segmentation itself."
                )

        cmd = self._build_docker_cmd(dirpath)
        logger.info(f"Running BOA on {patient_id} {study_date}: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stderr:
                logger.info(result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error during BOA inference for {patient_id} {study_date}:\n{e.stderr}")
            return

        seg_paths = self._collect_segmentations(work_dir, dirpath)
        metrics = self._collect_metrics(work_dir)
        self._update_patient_info(dirpath, study_date, patient_id, seg_paths, metrics)

    def _seed_total(self, work_dir: str, ctseg_path: str) -> None:
        """Place CTseg.nii.gz as total.nii.gz in BOA's output dir via a relative symlink
        (copy fallback). Both files share the study dir, so a relative link resolves
        correctly inside the Docker bind mount."""
        total_path = os.path.join(work_dir, "total.nii.gz")
        if os.path.lexists(total_path):
            os.remove(total_path)
        try:
            os.symlink(os.path.join("..", "CTseg.nii.gz"), total_path)
        except OSError:
            shutil.copy2(ctseg_path, total_path)
        logger.info(f"Seeded {total_path} from CTseg.nii.gz so BOA reuses the total segmentation.")

    def _build_docker_cmd(self, study_dir: str) -> list[str]:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--shm-size=8g",
            "--ulimit",
            "memlock=-1",
            "--ulimit",
            "stack=67108864",
            "-e",
            f"DOCKER_USER={os.getuid()}:{os.getgid()}",
            "-v",
            f"{os.path.abspath(study_dir)}:/workspace",
        ]
        if self.weights_dirpath:
            cmd += ["-v", f"{os.path.abspath(self.weights_dirpath)}:/app/weights"]
        cmd += [
            self.image,
            "python",
            "-m",
            "body_organ_analysis",
            "--input-image",
            "/workspace/CT.nii.gz",
            "--output-dir",
            "/workspace/boa",
            "--models",
            "total+bca",
            "--device",
            self.device,
            "--verbose",
        ]
        if self.fast_bca:
            cmd.append("--fast-bca")
        if self.no_pdf:
            cmd.append("--bca-no-pdf")
        return cmd

    def _collect_segmentations(self, work_dir: str, study_dir: str) -> dict[str, str]:
        """Move BOA's BCA segmentation NIfTIs up into the study root with MUSIQ-style
        names. Returns {json_key: path} for the ones found."""
        seg_paths: dict[str, str] = {}
        for src_name, dst_name in BCA_SEG_RENAME.items():
            src = os.path.join(work_dir, src_name)
            if not os.path.isfile(src):
                logger.warning(f"Expected BOA output {src_name} not found in {work_dir}.")
                continue
            dst = os.path.join(study_dir, dst_name)
            shutil.move(src, dst)
            key = f"{dst_name[:-7]}Path"  # e.g. CTbca_tissuesPath
            seg_paths[key] = dst

        # Surface any other (unexpected) segmentation files so naming drift is visible.
        for f in os.listdir(work_dir):
            if f.endswith(".nii.gz") and f != "total.nii.gz" and f not in BCA_SEG_RENAME:
                logger.warning(f"Unmapped BOA segmentation file left in {work_dir}: {f}")
        return seg_paths

    def _collect_metrics(self, work_dir: str) -> dict:
        """Read BOA's JSON measurement files into a single dict keyed by file stem."""
        metrics: dict = {}
        for f in sorted(os.listdir(work_dir)):
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(work_dir, f)) as fh:
                    metrics[f[:-5]] = json.load(fh)
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(f"Could not read BOA measurement file {f}: {e}")
        return metrics

    def _update_patient_info(
        self,
        study_dir: str,
        study_date: str,
        patient_id: str,
        seg_paths: dict[str, str],
        metrics: dict,
    ) -> None:
        patient_dirpath = os.path.dirname(study_dir)
        patient_info_path = os.path.join(patient_dirpath, "patient_info.json")
        if not os.path.isfile(patient_info_path):
            logger.error(f"Missing patient_info.json in {patient_dirpath}; cannot record BOA results.")
            return

        with open(patient_info_path) as f:
            patient_info = json.load(f)

        try:
            # A single CT series per study is assumed throughout the pipeline.
            series_name = next(iter(patient_info["Studies"][study_date]["Modalities"]["CT"][0]))
            ct_entry = patient_info["Studies"][study_date]["Modalities"]["CT"][0][series_name]
        except (KeyError, IndexError, StopIteration):
            logger.error(f"Could not locate CT series slot for {patient_id} {study_date} in patient_info.json.")
            return

        ct_entry.update(seg_paths)
        if metrics:
            ct_entry["BCA"] = metrics

        with open(patient_info_path, "w") as f:
            json.dump(patient_info, f)
        logger.info(f"Recorded BOA BCA results for {patient_id} {study_date} in patient_info.json.")


def boa_inference_entrypoint() -> None:
    """Entry point to run the BOA BCA stage without the full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.boa_inference")

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run the UMEssen BOA Body Composition Analysis (BCA) on all CT.nii.gz "
        "files in a processed folder via the shipai/boa-cli Docker image. Reuses MUSIQ's CTseg.nii.gz "
        "as BOA's total segmentation when present."
    )
    parser.add_argument(
        "--input-dirpath-processed", type=str, required=True, help="Path to the processed output folder."
    )
    parser.add_argument(
        "--boa-weights-path", type=str, default=None, help="Local BOA weights dir mounted at /app/weights."
    )
    parser.add_argument("--boa-image", type=str, default="shipai/boa-cli", help="BOA Docker image tag.")
    parser.add_argument("--boa-fast", action="store_true", help="Use the fast single-fold BCA variant.")
    parser.add_argument("--boa-no-pdf", action="store_true", help="Skip the BCA PDF report (keep JSON measurements).")
    parser.add_argument("--boa-device", type=str, default="gpu", help="BOA device: gpu, cuda or cpu.")
    parser.add_argument(
        "--boa-no-reuse-total",
        action="store_true",
        help="Let BOA compute the total segmentation instead of reusing CTseg.nii.gz.",
    )
    args = parser.parse_args()

    BoaInference(
        input_dirpath_processed=args.input_dirpath_processed,
        weights_dirpath=args.boa_weights_path,
        image=args.boa_image,
        fast_bca=args.boa_fast,
        no_pdf=args.boa_no_pdf,
        device=args.boa_device,
        reuse_total=not args.boa_no_reuse_total,
    ).run()


if __name__ == "__main__":
    boa_inference_entrypoint()
