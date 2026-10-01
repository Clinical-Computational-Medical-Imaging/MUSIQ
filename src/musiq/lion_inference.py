"""LION lesion segmentation stage — runs under .venv_lion (no musiq imports)."""

import json
import logging
import os
import pathlib as plb
import shutil
import sys
import tempfile

logger = logging.getLogger(__name__)

# Must match RESERVED_PROCESSED_DIRS in utils.py — inlined here because this script runs in .venv_lion.
_RESERVED_DIRS = {"cads_staging", "plots", "logger"}


class LionInference:
    def __init__(
        self,
        input_dirpath_processed: str | os.PathLike,
        lion_model: str = "fdg",
        pet_metric: str | list[str] | None = None,
        lion_accelerator: str | None = None,
    ) -> None:
        """Walk the processed tree and run LION tumor segmentation on SUV/SUL NIfTIs.

        Produces PETseg_LION.nii.gz (SUV) or PETsegSUL_LION.nii.gz (SUL) per study dir.
        Updates patient_info.json with PETsegLIONPath / PETsegSULLIONPath.

        Args:
            input_dirpath_processed: Root of the processed output tree.
            lion_model: LION model — 'fdg' or 'psma'.
            pet_metric: Metric(s) to segment — 'SUV', 'SUL', or both. Defaults to ['SUV', 'SUL'].
            lion_accelerator: Device override — 'cuda', 'cpu', or 'mps'. Auto-detected when None.
        """
        if pet_metric is None:
            pet_metric = ["SUV", "SUL"]
        pet_metrics = [pet_metric] if isinstance(pet_metric, str) else list(pet_metric)
        for m in pet_metrics:
            if m not in ("SUV", "SUL"):
                raise ValueError(f"pet_metric must be 'SUV' or 'SUL', got '{m}'")
        if lion_model not in ("fdg", "psma"):
            raise ValueError(f"lion_model must be 'fdg' or 'psma', got '{lion_model}'")
        self.input_dirpath = str(input_dirpath_processed)
        self.lion_model = lion_model
        self.pet_metrics = pet_metrics
        self.lion_accelerator = lion_accelerator

    def _resolve_accelerator(self) -> str:
        if self.lion_accelerator:
            return self.lion_accelerator
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def run(self) -> None:
        import lionz

        accelerator = self._resolve_accelerator()
        logger.info(f"LION accelerator: {accelerator}, model: {self.lion_model}")

        for metric in self.pet_metrics:
            petseg_fname = "PETseg_LION.nii.gz" if metric == "SUV" else "PETsegSUL_LION.nii.gz"
            petseg_key = "PETseg_LIONPath" if metric == "SUV" else "PETsegSUL_LIONPath"
            found_any = False

            for dirpath, dirnames, filenames in os.walk(self.input_dirpath):
                dirnames[:] = [d for d in dirnames if d not in _RESERVED_DIRS]
                if f"{metric}.nii.gz" not in filenames:
                    continue
                found_any = True
                patient_series = plb.Path(dirpath).parts[-2:]
                logger.info(f"Processing {patient_series[0]}/{patient_series[1]}")

                if os.path.isfile(os.path.join(dirpath, petseg_fname)):
                    logger.info(f"Skipping {patient_series}: {petseg_fname} already exists.")
                    continue

                patient_dirpath = os.path.dirname(dirpath)
                study_date = os.path.basename(dirpath)
                json_fpath = os.path.join(patient_dirpath, "patient_info.json")
                flag_json_exists = os.path.isfile(json_fpath)
                if not flag_json_exists:
                    logger.warning(f"Missing patient_info.json in {patient_dirpath}, skipping JSON update.")

                if flag_json_exists:
                    with open(json_fpath) as f:
                        patient_info = json.load(f)

                input_fpath = os.path.join(dirpath, f"{metric}.nii.gz")
                with tempfile.TemporaryDirectory() as tmp:
                    # LION Python API takes a file path (PT_-prefixed) and returns the output path.
                    pt_copy = os.path.join(tmp, "PT_input.nii.gz")
                    shutil.copy(input_fpath, pt_copy)
                    try:
                        seg_fpath = lionz.lion(
                            pt_copy, self.lion_model, output_dir=tmp, accelerator=accelerator, verbose_console=True
                        )
                        if not seg_fpath or not os.path.exists(seg_fpath):
                            logger.error(f"LION produced no output for {patient_series}.")
                            continue
                        shutil.copy(seg_fpath, os.path.join(dirpath, petseg_fname))
                        logger.info(f"Saved {petseg_fname} for {patient_series}.")

                        if flag_json_exists:
                            series_name = next(iter(patient_info["Studies"][study_date]["Modalities"]["PT"][0]))
                            patient_info["Studies"][study_date]["Modalities"]["PT"][0][series_name].update(
                                {petseg_key: os.path.join(dirpath, petseg_fname)}
                            )
                    except Exception as e:
                        logger.error(f"LION inference failed for {patient_series}: {e}")
                        continue

                if flag_json_exists:
                    with open(json_fpath, "w") as f:
                        json.dump(patient_info, f)

            if not found_any:
                msg = f"No {metric}.nii.gz files found under {self.input_dirpath}."
                if metric == "SUL":
                    msg += " SUL.nii.gz is produced by the sul task — ensure it has run first."
                logger.warning(msg)


def lion_inference_entrypoint() -> None:
    """Entry point for standalone LION inference (runs under .venv_lion)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        stream=sys.stdout,
    )
    global logger
    logger = logging.getLogger("musiq.lion_inference")

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run LION on all SUV.nii.gz or SUL.nii.gz files in a processed tree."
    )
    parser.add_argument("--input-dirpath-processed", required=True, help="Processed output tree root.")
    parser.add_argument(
        "--lion-model", type=str, default="fdg", choices=["fdg", "psma"], help="LION model (default: fdg)."
    )
    parser.add_argument(
        "--pet-metric",
        type=str,
        nargs="+",
        choices=["SUV", "SUL"],
        default=["SUV", "SUL"],
        help="PET metric(s) to segment (default: SUV SUL).",
    )
    parser.add_argument(
        "--lion-accelerator",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device override (default: auto-detect).",
    )
    args = parser.parse_args()

    LionInference(
        input_dirpath_processed=args.input_dirpath_processed,
        lion_model=args.lion_model,
        pet_metric=args.pet_metric,
        lion_accelerator=args.lion_accelerator,
    ).run()


if __name__ == "__main__":
    lion_inference_entrypoint()
