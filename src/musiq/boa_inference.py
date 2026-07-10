import json
import logging
import os
import pathlib as plb
import shutil
import subprocess

from .utils import list_patient_dirs

logger = logging.getLogger(__name__)

# BOA output name -> MUSIQ study-root name for BCA segmentation maps.
# ``total.nii.gz`` is excluded: it's the reused seed, not a BCA product.
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
        runtime: str = "docker",
        sif_path: str | os.PathLike | None = None,
    ) -> None:
        """Run the UMEssen BOA Body Composition Analysis (BCA) on every ``CT.nii.gz`` in the processed tree.

        Reuses MUSIQ's TotalSegmentator ``CTseg.nii.gz`` as BOA's ``total`` seg (when ``reuse_total``);
        runs the container via Docker (default) or Apptainer/Singularity.
        """
        self.input_dirpath = input_dirpath_processed
        self.weights_dirpath = weights_dirpath
        self.image = image
        self.fast_bca = fast_bca
        self.no_pdf = no_pdf
        self.device = device
        self.reuse_total = reuse_total
        self.runtime = runtime
        self.sif_path = sif_path

        if self.runtime not in ("docker", "apptainer"):
            raise ValueError(f"Invalid BOA runtime {self.runtime!r}; expected 'docker' or 'apptainer'.")
        if self.runtime == "apptainer":
            if not self.sif_path:
                raise ValueError("runtime='apptainer' requires sif_path pointing at the BOA .sif image.")
            if not self.weights_dirpath:
                logger.warning(
                    "runtime='apptainer' without weights_dirpath: the SIF filesystem is read-only, so BOA "
                    "cannot download weights at runtime. Pass --boa-weights-path to a pre-populated dir if "
                    "the image does not already bundle the weights."
                )

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
        # Skip if BCA outputs already exist
        final_seg = os.path.join(dirpath, BCA_SEG_RENAME["tissues.nii.gz"])
        if os.path.isfile(final_seg):
            logger.info(f"BCA outputs already exist for {patient_id} {study_date}, skipping.")
            return

        work_dir = os.path.join(dirpath, "boa")
        os.makedirs(work_dir, exist_ok=True)

        # Seed total seg so BOA skips the 'total' model
        ctseg_path = os.path.join(dirpath, "CTseg.nii.gz")
        if self.reuse_total:
            if os.path.isfile(ctseg_path):
                self._seed_total(work_dir, ctseg_path)
            else:
                logger.warning(
                    f"reuse_total set but CTseg.nii.gz missing for {patient_id} {study_date}; "
                    "BOA will compute the total segmentation itself."
                )

        cmd = self._build_cmd(dirpath)
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
        """Place CTseg.nii.gz as total.nii.gz in BOA's output dir (relative symlink, copy fallback)."""
        total_path = os.path.join(work_dir, "total.nii.gz")
        if os.path.lexists(total_path):
            os.remove(total_path)
        try:
            os.symlink(os.path.join("..", "CTseg.nii.gz"), total_path)
        except OSError:
            shutil.copy2(ctseg_path, total_path)
        logger.info(f"Seeded {total_path} from CTseg.nii.gz so BOA reuses the total segmentation.")

    def _build_cmd(self, study_dir: str) -> list[str]:
        if self.runtime == "apptainer":
            return self._build_apptainer_cmd(study_dir)
        return self._build_docker_cmd(study_dir)

    def _boa_args(self) -> list[str]:
        # Container-side paths (/workspace = bind-mounted study dir)
        args = [
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
            args.append("--fast-bca")
        if self.no_pdf:
            args.append("--bca-no-pdf")
        return args

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
        cmd.append(self.image)
        cmd += self._boa_args()
        return cmd

    def _build_apptainer_cmd(self, study_dir: str) -> list[str]:
        """Apptainer equivalent of the Docker command.

        ``exec`` bypasses ENTRYPOINT (runs as the invoking user); ``--nv`` exposes the GPU.
        """
        cmd = ["apptainer", "exec"]
        if self.device != "cpu":
            cmd.append("--nv")
        cmd += ["--bind", f"{os.path.abspath(study_dir)}:/workspace"]
        if self.weights_dirpath:
            cmd += ["--bind", f"{os.path.abspath(self.weights_dirpath)}:/app/weights"]
        cmd.append(os.fspath(self.sif_path))
        cmd += self._boa_args()
        return cmd

    def _collect_segmentations(self, work_dir: str, study_dir: str) -> dict[str, str]:
        """Move BCA segmentation NIfTIs to the study root with MUSIQ names; return {json_key: path}."""
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

        # Log unexpected seg files
        for f in os.listdir(work_dir):
            if f.endswith(".nii.gz") and f != "total.nii.gz" and f not in BCA_SEG_RENAME:
                logger.warning(f"Unmapped BOA segmentation file left in {work_dir}: {f}")
        return seg_paths

    def _collect_metrics(self, work_dir: str) -> dict:
        """Aggregate BOA JSON measurements keyed by file stem."""
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
        description="Run BOA Body Composition Analysis (BCA) on all CT.nii.gz in a processed folder."
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
    parser.add_argument(
        "--boa-runtime",
        type=str,
        default="docker",
        choices=["docker", "apptainer"],
        help="Container runtime for BOA (default: docker). Use 'apptainer' on HPC clusters.",
    )
    parser.add_argument(
        "--boa-sif",
        type=str,
        default=None,
        help="Path to the BOA Apptainer/Singularity image (.sif). Required when --boa-runtime apptainer.",
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
        runtime=args.boa_runtime,
        sif_path=args.boa_sif,
    ).run()


if __name__ == "__main__":
    boa_inference_entrypoint()
