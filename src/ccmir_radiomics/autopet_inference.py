import json
import logging
import os
import pathlib as plb
import shutil
import subprocess
import tempfile

import torch

from .utils import natural_key, resample_image

logger = logging.getLogger(__name__)


class AutopetInference:
    def __init__(
        self, input_dirpath_processed: str | os.PathLike, autopet_checkpoint_dirpath: str | os.PathLike
    ) -> None:
        """Resamples CT.nii images to PT size and runs the AutoPET inference on all SUV.nii.gz files in the
        input directory. Creates CTres.nii and PETseg.nii.
        Expects exactly one PT and matching CT series per study date.

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the SUV.nii.gz files. Can be nested.
            autopet_checkpoint_dirpath (str | os.PathLike): Directory containing the nnUNet checkpoint for
            AutoPET inference. See README for details on how to obtain the checkpoint and how to name the folder.
        """
        self.input_dirpath = input_dirpath_processed
        self.autopet_checkpoint_dirpath = autopet_checkpoint_dirpath

    def run(self) -> None:
        top_dirs = [d for d in os.listdir(self.input_dirpath) if os.path.isdir(os.path.join(self.input_dirpath, d))]
        top_dirs.sort(key=natural_key)

        for top_dir in top_dirs:
            top_dir_path = os.path.join(self.input_dirpath, top_dir)

            for dirpath, _, filenames in os.walk(top_dir_path):
                for filename in filenames:
                    if filename == "SUV.nii.gz":
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

                        if os.path.isfile(os.path.join(dirpath, "PETseg.nii.gz")):
                            logger.info(f"Skipping {patient_series} as PETseg.nii.gz already exists.")
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
                            shutil.copy(os.path.join(dirpath, "CTres.nii.gz"), os.path.join(tmp, "ALPS_0000.nii.gz"))
                            shutil.copy(os.path.join(dirpath, "SUV.nii.gz"), os.path.join(tmp, "ALPS_0001.nii.gz"))
                            try:
                                with tempfile.TemporaryDirectory() as output_folder:
                                    output_folder = plb.Path(str(output_folder))
                                    os.environ["nnUNet_raw"] = ""
                                    os.environ["nnUNet_preprocessed"] = ""
                                    os.environ["nnUNet_results"] = ""

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
                                    subprocess.run(command, check=True)

                                    nii = next(output_folder.glob("*nii.gz"))
                                    shutil.copy(nii, os.path.join(dirpath, "PETseg.nii.gz"))
                                if flag_json_exists:
                                    series_name = next(iter(patient_info["Studies"][study_date]["Modalities"]["PT"][0]))
                                    patient_info["Studies"][study_date]["Modalities"]["PT"][0][series_name].update(
                                        {"PETsegPath": f"{dirpath}/PETseg.nii.gz"}
                                    )

                            except subprocess.CalledProcessError as e:
                                logger.error(f"Error during nnUNet prediction: {e}")
                            except Exception as e:
                                logger.error(f"Unexpected error: {e}")

                        if flag_json_exists:
                            with open(os.path.join(patient_dirpath, "patient_info.json"), "w") as f:
                                json.dump(patient_info, f)


def autopet_inference_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("ccmir_radiomics.autopet_inference")

    import argparse

    parser = argparse.ArgumentParser(description="Recursively run AutoPET on all SUV.nii.gz files in a folder. ")
    parser.add_argument(
        "--input-dirpath-processed",
        type=str,
        help="Path to the input folder containing SUV.nii.gz files",
        required=True,
    )
    parser.add_argument("--nnunet-checkpoint", type=str, help="Path to the nnunet checkpoint folder")
    args = parser.parse_args()

    AutopetInference(
        input_dirpath_processed=args.input_dirpath_processed, autopet_checkpoint_dirpath=args.nnunet_checkpoint
    ).run()


if __name__ == "__main__":
    autopet_inference_entrypoint()
