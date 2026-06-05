import json
import logging
import os
import sys
import shutil

from .utils import natural_key


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "CADS"))

from cads.utils.libs import setup_nnunet_env
setup_nnunet_env()
from cads.utils.inference import predict
from cads.utils.libs import get_model_weights_dir, check_or_download_model_weights


logger = logging.getLogger(__name__)


class CadsInference:
    def __init__(self, input_dirpath_processed: str | os.PathLike, tasks: list[str] | None = None) -> None:
        """Class to handle CADS model inference on CT.nii.gz files in a specified folder.
        It processes each file, runs segmentation, extracts label mapping. Creates CTcads.nii.
        All given tasks are stored in one file ant the labels are set as in the labelmap_all_structure 
        shared by the cads model (https://github.com/murong-xu/CADS/blob/main/cads/dataset_utils/bodyparts_labelmaps.py).

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the CT.nii.gz files. Can be nested.
        """        
        self.input_dirpath = input_dirpath_processed
        self.tasks = tasks

        cads_tasks_error = bool(
            any(
                t
                not in [
                    "551",
                    "552",
                    "553",
                    "554",
                    "555",
                    "556",
                    "557",
                    "558",
                    "559",
                    "all",
                    "",
                ]
                for t in tasks or []
            )
        )
        
        if cads_tasks_error:
            logger.error("Wrong input tasks for CADS model. Please use one/ some of the following separated by spaces: all 551 552 553 554 555 556 557 558 559")
            sys.exit(1)

        tasks_error = bool(
            any(
                t
                not in [
                    "551",
                    "552",
                    "553",
                    "554",
                    "555",
                    "556",
                    "557",
                    "558",
                    "559",
                ]
                for t in tasks or []
            )
        )

        if tasks is None or "all" in tasks or tasks_error:
            self.task_ids = [551,552,553,554,555,556,557,558,559]
        else:
            self.task_ids = list(map(int, tasks))
    
    def run(self) -> None:
        """
        Recursively search the folder for CT.nii.gz files.
        For each found file, create a 'CTcads' file, run segmentation using the given task options,
        extract the SUL image using the segmentation output, and save a metadata to the JSON file.
        """
        if not os.path.isdir(self.input_dirpath):
            logger.error(f"Error: {self.input_dirpath} is not a valid directory.")
            return

        logger.info(f"Starting CADS inference in {self.input_dirpath}")
        top_dirs = [d for d in os.listdir(self.input_dirpath) if os.path.isdir(os.path.join(self.input_dirpath, d))]
        top_dirs.sort(key=natural_key)

        for top_dir in top_dirs:
            top_dir_path = os.path.join(self.input_dirpath, top_dir)

            for dirpath, _, filenames in os.walk(top_dir_path):
                for filename in filenames:
                    # Determine if this is a CT file and set parameters accordingly
                    is_ct = filename == "CT.nii.gz"

                    if not is_ct:
                        continue
                    patient_id = os.path.basename(os.path.dirname(dirpath))
                    input_fpath = os.path.join(dirpath, filename)
                    
                    output_fpath = os.path.join(dirpath, f"CTcads.nii.gz")
                    modality = "CT"

                    if os.path.isfile(output_fpath):
                        logger.info(f"Output file {output_fpath} already exists.")
                        continue

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

                    logger.info(f"Processing file {filename} for patient {patient_id}.")
                    
                    model_folder = get_model_weights_dir()

                    for task_id in self.task_ids:
                        check_or_download_model_weights(task_id)

                    predict(
                        [input_fpath],
                        dirpath,
                        model_folder,
                        self.task_ids,
                        folds='all',
                        save_all_combined_seg=True,
                        save_separate_targets=False,
                        use_cpu=False,
                    )

                    os.rename(
                        os.path.join(dirpath, "CT", f"CT_combined.nii.gz"),
                        output_fpath
                    )
                    shutil.rmtree(
                        os.path.join(dirpath, "CT")
                    )

                    # Prepare metadata with settings, task/model info, and the label mapping obtained.
                    seg_metadata = {
                        "settings": {"in_preprocessed_images": input_fpath, "out": output_fpath,  "task": self.task_ids},
                        "model": "cads v1.0.0",
                    }
                    if json_exists and patient_info is not None:
                        study_date = dirpath.split(os.sep)[-1]
                        if modality == "CT":
                            series_index = 0
                        else:
                            mr_series = patient_info["Studies"][study_date]["Modalities"][modality]
                            # Find the index where the filename matches the MRPath value
                            series_index = None
                            for idx, serie in enumerate(mr_series):
                                for _serie_name, serie_data in serie.items():
                                    if "MRPath" in serie_data and filename in os.path.basename(serie_data["MRPath"]):
                                        series_index = idx
                                        break
                                if series_index is not None:
                                    break

                        if series_index is None:
                            logger.error(f"Could not find series index for {filename} in patient_info.json.")
                            continue
                        series_name = next(
                            iter(patient_info["Studies"][study_date]["Modalities"][modality][series_index])
                        )
                        patient_info["Studies"][study_date]["Modalities"][modality][series_index][series_name].update(
                            {f"{modality}cadsPath": output_fpath}
                        )
                        patient_info["Studies"][study_date]["Modalities"][modality][series_index][series_name].update(
                            {f"{modality}cads_metadata": seg_metadata}
                        )
                        with open(patient_info_path, "w") as f:
                            json.dump(patient_info, f)
                    else:
                        with open(f"{filename[:-7]}_seg.json", "w") as f:
                            json.dump(seg_metadata, f)


def cads_inference_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.cads_inference")

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run CADS model on all CT.nii.gz files in a folder. "
        "Extract SUL images using the segmentation output and save metadata to json."
    )
    parser.add_argument(
        "--input-dirpath-processed", type=str, help="Path to the input folder containing CT.nii.gz files", required=True
    )
    parser.add_argument(
        "--cads-tasks",
        nargs="+",
        help="List of tasks to run in the CADS model. Possible tasks (different tasks can be separated by spaces): "
        "all, 551, 552, 553, 554, 555, 556, 557, 558, 559.",
        default=None,
    )
    args = parser.parse_args()

    CadsInference(
        input_dirpath_processed=args.input_dirpath_processed,
        tasks=args.cads_tasks,
    ).run()


if __name__ == "__main__":
    cads_inference_entrypoint()
