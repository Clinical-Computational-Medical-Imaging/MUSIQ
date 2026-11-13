import json
import os

from moosez import moose

from musiq.utils import create_logger, list_patient_dirs

logger = create_logger("musiq.moose_inference")


class MooseInference:
    def __init__(self, input_dirpath_processed: str | os.PathLike, moose_task: str | list[str]) -> None:
        """Run Moose segmentation on CT.nii.gz files.

        See the moosez library for available tasks and their outputs.
        """
        self.input_dirpath = input_dirpath_processed
        if isinstance(moose_task, str):
            self.model_name = [moose_task]
        else:
            self.model_name = moose_task
        self.accelerator = "cuda"

    def run(self) -> None:
        """Run Moose segmentation on all CT.nii.gz files in the folder."""
        if not os.path.isdir(self.input_dirpath):
            raise ValueError(f"{self.input_dirpath} is not a valid directory.")

        logger.info(f"Starting Moose Segmentator inference in: {self.input_dirpath} using task: {self.model_name}")

        top_dirs = list_patient_dirs(self.input_dirpath)

        for top_dir in top_dirs[:3]:
            top_dir_path = os.path.join(self.input_dirpath, top_dir)

            for dirpath, _, filenames in os.walk(top_dir_path):
                for filename in filenames:
                    if filename == "CT.nii.gz":
                        input_fpath = os.path.join(dirpath, filename)

                        for task in self.model_name:
                            new_path = os.path.join(dirpath, "CTseg_moose_" + task + ".nii.gz")
                            if os.path.isfile(new_path):
                                logger.info(f"Output file {new_path} already exists.")
                                continue
                            logger.info(f"Processing file: {input_fpath}")

                            moose(input_fpath, task, dirpath, self.accelerator)

                            output_fpath = os.path.join(dirpath, task.replace("ct", "CT") + "_segmentation_CT.nii.gz")
                            if os.path.isfile(output_fpath):
                                os.rename(output_fpath, new_path)
                            else:
                                logger.warning("Something went wrong renaming the output file!")

                            patient_dirpath = os.path.dirname(dirpath)
                            patient_info = None
                            patient_info_path = os.path.join(patient_dirpath, "patient_info.json")
                            if os.path.isfile(patient_info_path):
                                json_exists = True
                                with open(patient_info_path) as json_file:
                                    patient_info = json.load(json_file)
                            else:
                                json_exists = False
                                logger.error(f"Missing patient_info.json in {patient_dirpath}.")

                            if json_exists and patient_info is not None:
                                # one CT per study
                                study_date = dirpath.split(os.sep)[-1]
                                series_name = next(iter(patient_info["Studies"][study_date]["Modalities"]["CT"][0]))
                                patient_info["Studies"][study_date]["Modalities"]["CT"][0][series_name][
                                    f"CTsegPath_moose_{task}"
                                ] = new_path
                                with open(patient_info_path, "w") as f:
                                    json.dump(patient_info, f, indent=2)
                            else:
                                logger.error(f"Empty patient_info.json for dirpath: {patient_dirpath}")


def moose_inference_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from musiq.utils import create_logger

    global logger
    logger = create_logger("musiq.moose_inference")

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run Moose on all CT.nii.gz files in a folder. "
        "Extract label mapping from the segmentation output and save metadata as CTseg.json."
    )
    parser.add_argument(
        "--input-dirpath-processed", type=str, help="Path to the input folder containing CT.nii.gz files", required=True
    )
    parser.add_argument("--moose_task", nargs="+", help="A list of moose segmentator functions", required=True)
    args = parser.parse_args()

    MooseInference(
        input_dirpath_processed=args.input_dirpath_processed,
        moose_task=args.moose_task,
    ).run()


if __name__ == "__main__":
    moose_inference_entrypoint()
