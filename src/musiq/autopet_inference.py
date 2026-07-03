import json
import logging
import os
import pathlib as plb
import shutil
import subprocess
import tempfile

import torch

from .utils import list_patient_dirs, resample_image

logger = logging.getLogger(__name__)


class AutopetInference:
    def __init__(
        self,
        input_dirpath_processed: str | os.PathLike,
        autopet_checkpoint_dirpath: str | os.PathLike,
        pet_metric: str | list[str] | None = None,
    ) -> None:
        """Resamples CT.nii images to PT size and runs the AutoPET inference on all SUV.nii.gz or SUL.nii.gz
        files in the input directory. Creates CTres.nii and PETseg.nii.
        Expects exactly one PT and matching CT series per study date.

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the PET metric files. Can be nested.
            autopet_checkpoint_dirpath (str | os.PathLike): Directory containing the nnUNet checkpoint for
            AutoPET inference. See README for details on how to obtain the checkpoint and how to name the folder.
            pet_metric (str | list[str] | None): PET metric(s) to use as input.
                Accepts "SUV", "SUL", or both. Defaults to ["SUV", "SUL"].
        """
        if pet_metric is None:
            pet_metric = ["SUV", "SUL"]
        pet_metrics = [pet_metric] if isinstance(pet_metric, str) else list(pet_metric)
        for m in pet_metrics:
            if m not in ("SUV", "SUL"):
                raise ValueError(f"pet_metric must be 'SUV' or 'SUL', got '{m}'")
        self.input_dirpath = input_dirpath_processed
        self.autopet_checkpoint_dirpath = autopet_checkpoint_dirpath
        self.pet_metrics = pet_metrics

    def run(self) -> None:
        top_dirs = list_patient_dirs(self.input_dirpath)

        for metric in self.pet_metrics:
            petseg_fname = "PETseg.nii.gz" if metric == "SUV" else "PETsegSUL.nii.gz"
            petseg_key = "PETsegPath" if metric == "SUV" else "PETsegSULPath"
            found_any = False

            for top_dir in top_dirs:
                top_dir_path = os.path.join(self.input_dirpath, top_dir)

                for dirpath, _, filenames in os.walk(top_dir_path):
                    for filename in filenames:
                        if filename == f"{metric}.nii.gz":
                            found_any = True
                            patient_series = plb.Path(dirpath).parts[-2:]
                            logger.info(f"Processing {dirpath}")
                            patient_dirpath = os.path.dirname(dirpath)
                            study_date = dirpath.split(os.sep)[-1]

                            if not os.path.isfile(os.path.join(patient_dirpath, "patient_info.json")):
                                logger.error(f"Missing patient_info.json in {patient_dirpath}.")
                                flag_json_exists = False
                            else:
                                flag_json_exists = True
                                with open(os.path.join(patient_dirpath, "patient_info.json")) as json_file:
                                    patient_info = json.load(json_file)

                            if os.path.isfile(os.path.join(dirpath, petseg_fname)):
                                logger.info(f"Skipping {patient_series} as {petseg_fname} already exists.")
                                continue

                            if not os.path.isfile(os.path.join(dirpath, "CT.nii.gz")):
                                logger.info(f"Skipping {patient_series} as CT.nii.gz is missing.")
                                continue

                            ctres_fpath = os.path.join(dirpath, "CTres.nii.gz")
                            if not os.path.isfile(ctres_fpath):
                                logger.info("Resampling CT.nii.gz to PET size.")
                                resample_image(
                                    source_img=os.path.join(dirpath, "CT.nii.gz"),
                                    target_img=os.path.join(dirpath, "PET.nii.gz"),
                                    nii_output_dirpath=dirpath,
                                    output_fname="CTres.nii.gz",
                                    interpolation="continuous",
                                    fill_value=-1024,
                                )
                                if flag_json_exists:
                                    series_name = next(iter(patient_info["Studies"][study_date]["Modalities"]["CT"][0]))
                                    patient_info["Studies"][study_date]["Modalities"]["CT"][0][series_name].update(
                                        {"CTresPath": f"{dirpath}/CTres.nii.gz"}
                                    )

                            with tempfile.TemporaryDirectory() as tmp:
                                shutil.copy(
                                    os.path.join(dirpath, "CTres.nii.gz"), os.path.join(tmp, "case_0000.nii.gz")
                                )
                                shutil.copy(
                                    os.path.join(dirpath, f"{metric}.nii.gz"), os.path.join(tmp, "case_0001.nii.gz")
                                )
                                try:
                                    with tempfile.TemporaryDirectory() as output_folder:
                                        output_folder = plb.Path(str(output_folder))
                                        os.environ["nnUNet_raw"] = ""  # noqa: SIM112
                                        os.environ["nnUNet_preprocessed"] = ""  # noqa: SIM112
                                        os.environ["nnUNet_results"] = ""  # noqa: SIM112

                                        logger.info(f"GPU available: {torch.cuda.is_available()}")

                                        command = [
                                            "nnUNetv2_predict_from_modelfolder",
                                            "-i",
                                            tmp,
                                            "-o",
                                            str(output_folder),
                                            "-m",
                                            self.autopet_checkpoint_dirpath,
                                            "-device",
                                            "cuda" if torch.cuda.is_available() else "cpu",
                                        ]

                                        logger.info("Running nnUNet prediction...")
                                        subprocess.run(command, check=True, capture_output=True, text=True)

                                        nii = next(output_folder.glob("*nii.gz"))
                                        shutil.copy(nii, os.path.join(dirpath, petseg_fname))
                                    if flag_json_exists:
                                        series_name = next(
                                            iter(patient_info["Studies"][study_date]["Modalities"]["PT"][0])
                                        )
                                        patient_info["Studies"][study_date]["Modalities"]["PT"][0][series_name].update(
                                            {petseg_key: f"{dirpath}/{petseg_fname}"}
                                        )

                                except subprocess.CalledProcessError as e:
                                    logger.error(
                                        f"Error during nnUNet prediction: {e}\nstdout: {e.stdout}\nstderr: {e.stderr}"
                                    )
                                except Exception as e:
                                    logger.error(f"Unexpected error: {e}")

                            if flag_json_exists:
                                with open(os.path.join(patient_dirpath, "patient_info.json"), "w") as f:
                                    json.dump(patient_info, f)

            if not found_any:
                msg = f"No {metric}.nii.gz files found under {self.input_dirpath}."
                if metric == "SUL":
                    msg += " SUL.nii.gz is created by the muscle-fat task — make sure it has been run first."
                logger.warning(msg)


def autopet_inference_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.autopet_inference")

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run AutoPET on all SUV.nii.gz or SUL.nii.gz files in a folder."
    )
    parser.add_argument(
        "--input-dirpath-processed",
        type=str,
        help="Path to the input folder containing the PET metric files",
        required=True,
    )
    parser.add_argument("--nnunet-checkpoint", type=str, help="Path to the nnunet checkpoint folder")
    parser.add_argument(
        "--pet-metric",
        type=str,
        nargs="+",
        choices=["SUV", "SUL"],
        default=["SUV", "SUL"],
        help="PET metric(s) to use as input. Pass one or both: --pet-metric SUV SUL (default: SUV SUL).",
    )
    args = parser.parse_args()

    AutopetInference(
        input_dirpath_processed=args.input_dirpath_processed,
        autopet_checkpoint_dirpath=args.nnunet_checkpoint,
        pet_metric=args.pet_metric,
    ).run()


if __name__ == "__main__":
    autopet_inference_entrypoint()
