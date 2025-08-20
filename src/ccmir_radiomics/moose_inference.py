import json
import os

from moosez import moose

from ccmir_radiomics.utils import create_logger, natural_key

logger = create_logger("ccmir_radiomics.moose_inference")


class MooseInference:
    def __init__(self, input_dirpath_processed: str | os.PathLike, moose_task: str | list[str]) -> None:
        """
        Class to handle Moose inference on CT.nii.gz files in a specified folder.
        It processes each file, runs the given segmentation and safes in moose_task_segmentation_CT.nii.gz

        Moose tasks(model_name):
                clin_ct_body: it segments Legs, Body, Head, Arms
                clin_ct_body_composition: it segments skeletal_muscle, subcutaneous_fat, visceral_fat;
                clin_ct_cardiac: it segments heart_myocardium, heart_atrium_left, heart_atrium_right,
                    heart_ventricle_left, heart_ventricle_right, aorta, iliac_artery_left, iliac_artery_right,
                    iliac_vena_left, iliac_vena_right, inferior_vena_cava, portal_splenic_vein, pulmonary_artery;
                clin_ct_digestive: it segments colon, duodenum, esophagus, small_bowel;
                clin_ct_lungs: it segments lung_upper_lobe_left, lung_lower_lobe_left, lung_upper_lobe_right,
                    lung_middle_lobe_right, lung_lower_lobe_right;
                clin_ct_muscles: it segments autochthon_left, autochthon_right, gluteus_maximus_left,
                    gluteus_maximus_right, gluteus_medius_left, gluteus_medius_right, gluteus_minimus_left,
                    gluteus_minimus_right, iliopsoas_left, iliopsoas_right;
                clin_ct_organs: it segments adrenal_gland_left, adrenal_gland_right, bladder, brain, gallbladder,
                    kidney_left, kidney_right, liver, lung_lower_lobe_left, lung_lower_lobe_right,
                    lung_middle_lobe_right, lung_upper_lobe_left, lung_upper_lobe_right, pancreas, spleen, stomach,
                    thyroid_left, thyroid_right, trachea;
                clin_ct_peripheral_bones: it segments carpal_left, carpal_right, clavicle_left, clavicle_right,
                    femur_left, femur_right, fibula_left, fibula_right, fingers_left, fingers_right, humerus_left,
                    humerus_right, metacarpal_left, metacarpal_right, metatarsal_left, metatarsal_right, patella_left,
                    patella_right, radius_left, radius_right, scapula_left, scapula_right, skull, tarsal_left,
                    tarsal_right, tibia_left, tibia_right, toes_left, toes_right, ulna_left, ulna_right;
                clin_ct_ribs: it segments rib_left_1, rib_left_2, rib_left_3, rib_left_4, rib_left_5, rib_left_6,
                    rib_left_7, rib_left_8, rib_left_9, rib_left_10, rib_left_11, rib_left_12, rib_left_13, rib_right_1,
                    rib_right_2, rib_right_3, rib_right_4, rib_right_5, rib_right_6, rib_right_7, rib_right_8,
                    rib_right_9, rib_right_10, rib_right_11, rib_right_12, rib_right_13, sternum;
                clin_ct_vertebrae: it segments vertebra_C1, vertebra_C2, vertebra_C3, vertebra_C4, vertebra_C5,
                    vertebra_C6, vertebra_C7, vertebra_T1, vertebra_T2, vertebra_T3, vertebra_T4, vertebra_T5,
                    vertebra_T6, vertebra_T7, vertebra_T8, vertebra_T9, vertebra_T10, vertebra_T11, vertebra_T12,
                    vertebra_L1, vertebra_L2, vertebra_L3, vertebra_L4, vertebra_L5, vertebra_L6, hip_left, hip_right,
                    sacrum;

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the CT.nii.gz files. Can be nested.
            moose_task (str | list[str]): Contains one or more out of Moose tasks from above.
        """
        self.input_dirpath = input_dirpath_processed
        if isinstance(moose_task, str):
            self.model_name = [moose_task]
        else:
            self.model_name = moose_task
        self.accelerator = "cuda"

    def run(self) -> None:
        """
        Recursively search the folder for CT.nii.gz files.
        For each found file, run segmentation using moose.
        """
        if not os.path.isdir(self.input_dirpath):
            raise ValueError(f"{self.input_dirpath} is not a valid directory.")

        logger.info(f"Starting Moose Segmentator inference in: {self.input_dirpath} using task: {self.model_name}")

        top_dirs = [d for d in os.listdir(self.input_dirpath) if os.path.isdir(os.path.join(self.input_dirpath, d))]
        top_dirs.sort(key=natural_key)

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
                                # expecting exactly one CT per study
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
    from ccmir_radiomics.utils import create_logger

    global logger
    logger = create_logger("ccmir_radiomics.moose_inference")

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
