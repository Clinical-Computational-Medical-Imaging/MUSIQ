import json
import logging
import numpy as np
import nibabel as nib
import os

from totalsegmentator.config import get_version
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from totalsegmentator.python_api import totalsegmentator

from .utils import natural_key, conv_time

logger = logging.getLogger(__name__)


class TotalSegmentatorMuscleFat:
    def __init__(self, input_dirpath_processed: os.PathLike | str) -> None:
        """Class to handle TotalSegmentator muscle fat analysis on CT.nii.gz files in a specified folder.
        It processes each file, runs segmentation, extracts label mapping. Creates CTseg.nii.

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the CT.nii.gz files. Can be nested.
        """
        self.input_dirpath = input_dirpath_processed

    def run(self) -> None:
        """
        Recursively search the folder for CT.nii.gz files.
        For each found file, run the tissue_4_types segmentation using ml option,
        extract the label mapping from the segmentation output, and save a metadata JSON file.
        It also calculates the lean bodymas and starts the SUL image creation.
        """
        if not os.path.isdir(self.input_dirpath):
            logger.error(f"Error: {self.input_dirpath} is not a valid directory.")
            return

        logger.info(f"Starting TotalSegmentator inference in {self.input_dirpath}")
        top_dirs = [d for d in os.listdir(self.input_dirpath) if os.path.isdir(os.path.join(self.input_dirpath, d))]
        top_dirs.sort(key=natural_key)

        for top_dir in top_dirs:
            top_dir_path = os.path.join(self.input_dirpath, top_dir)

            for dirpath, _, filenames in os.walk(top_dir_path):
                for filename in filenames:
                    # Determine if this is a CT or MR file and set parameters accordingly
                    is_ct = filename == "CT.nii.gz"
                    is_mr = (
                        filename.endswith("nii.gz")
                        and not filename.startswith(("CT", "SUV", "PET"))
                        and not filename.endswith("seg.nii.gz")
                    )

                    if not (is_ct or is_mr):
                        continue
                    patient_id = os.path.basename(os.path.dirname(dirpath))
                    input_fpath = os.path.join(dirpath, filename)
                    if is_ct:
                        output_fpath = os.path.join(dirpath, "CT_muscle_fat.nii.gz")
                        task = "tissue_4_types"
                        modality = "CT"

                    else:
                        output_fpath = os.path.join(dirpath, f"{filename[:-7]}_muscle_fat.nii.gz")
                        task = "tissue_types_mr"
                        modality = "MR"

                    metadata_key = f"{modality}muscle_fat_metadata" 
                    seg_path_key = f"{modality}muscle_fatPath"

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

                    # Run TotalSegmentator using the Python API with ml option and appropriate task.
                    try:
                        totalsegmentator(
                            input_fpath,
                            output_fpath,
                            ml=True,
                            task=task,
                            device="gpu:0",
                            statistics=False,
                            radiomics=False,
                        )
                        logger.info("Segmentation successfully completed.")
                    except Exception as e:
                        logger.error(f"Error during segmentation for {input_fpath}:\n  {e}")
                        continue
                    

                    # Load the segmentation file to extract the label mapping from its extended header.
                    try:
                        segmentation_img, label_map_dict = load_multilabel_nifti(output_fpath)
                        logger.info("Label mapping successfully loaded from segmentation file.")
                    except Exception as e:
                        logger.error(f"Error loading segmentation file {output_fpath}: {e}")
                        label_map_dict = {}

                    #logger.info(f"Labels in segmentation {label_map_dict}")

                    layers = ["full_picture", "l3", "glut_to_c6"]
                    for layer in layers:
                        calculation = self.calc_size(input_fpath, segmentation_img, label_map_dict, layer)

                        # Prepare metadata with settings, task/model info, and the label mapping obtained.
                        seg_metadata = {
                            "settings": {"input_fpath": input_fpath, "task": task, "ml": True},
                            "model": "total",
                            "ts_version": get_version(),
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
                            analysis_dict = patient_info["Studies"][study_date]["Modalities"][modality][series_index][series_name].setdefault(
                                "body_composition_analysis", {}
                            )
                            logger.info(f"Series: {filename}")
                            logger.info(f"Output path: {output_fpath}")
                            logger.info(f"Series index: {series_index}")
                            analysis_dict.update({seg_path_key: output_fpath})
                            analysis_dict.update({metadata_key: seg_metadata})
                            analysis_dict.setdefault(layer, {})
                            for label, value in calculation.items():
                                analysis_dict[layer][label] = value

                            with open(patient_info_path, "w") as f:
                                json.dump(patient_info, f)
                        else:
                            with open(f"{filename[:-7]}_muscle_fat.json", "w") as f:
                                json.dump(seg_metadata, f)

                    pet_path = os.path.isfile(os.path.join(dirpath, "PET.nii.gz"))
                    if not pet_path:
                        logger.info(f"No PET file found for {patient_id}")

                    with open(patient_info_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    series_name = next(
                                iter(data["Studies"][study_date]["Modalities"][modality][series_index])
                            )
                    weight = data["Studies"][study_date]["Modalities"][modality][series_index][series_name]["DICOM"]["PatientWeight"]
                    fat_in_percent = data["Studies"][study_date]["Modalities"][modality][series_index][series_name]["body_composition_analysis"]["glut_to_c6"]["total_fat_in_%"]
                    lean_bodymas = weight * (1 - fat_in_percent / 100)
                    sul_path = self.convert_pet2sul(dirpath, lean_bodymas, study_date)

                    data["Studies"][study_date]["Modalities"][modality][series_index][series_name]["SULPath"] = sul_path
                    data["Studies"][study_date]["Modalities"][modality][series_index][series_name]["PatientLBM"] = lean_bodymas
                    
                    with open(patient_info_path, "w") as f:
                        json.dump(data, f)
    
    def calc_size(self, path:os.PathLike, fat_img:nib.Nifti1Image, labels:dict[int, str], layer:str) -> dict[float, float, float]:
        """
            Calculate the volume (in mL) of each label in a segmentation and 
            compute total fat and muscle percentages relative to the whole scan.

            Args:
                path (os.PathLike): Path to the original CT/MR NIfTI image.
                img (nib.Nifti1Image): Segmentation NIfTI image.
                labels (dict[int, str]): Mapping of label numbers to label names.

            Returns:
                dict[str, float]: Dictionary with volume per label (mL) and
                        total fat/muscle percentages to the body volume (%)
                        muscle/fat ratio.        
        """       
        fallback_dict = {"total_fat_in_%": None,
                    "total_muscle_in_%": None,
                    "muscle_fat_ratio": None}
        total_seg_path = str(path).replace(".nii.gz", "seg.nii.gz")
        if not os.path.isfile(total_seg_path):
            logger.error(f"{total_seg_path} fehlt. Bitte TotalSegmentator task='total' zuerst ausführen.")
            return fallback_dict

        base_img = nib.load(path)
        total_seg_img, label_map_dict = load_multilabel_nifti(total_seg_path)

        seg_data = total_seg_img.get_fdata().astype(int)  # Labels als int

        if base_img.shape != total_seg_img.shape:
            logger.error("Shape mismatch!")

        if not np.allclose(base_img.affine, total_seg_img.affine):
            logger.warning("Affine mismatch!")
        # Alle eindeutigen Labels und deren Anzahl
        #unique_labels, counts = np.unique(seg_data, return_counts=True)

        #print("Eindeutige Labels:", unique_labels)




        name_to_label = {v: k for k, v in label_map_dict.items()}
        #logger.info(f"Labels in segmentation {name_to_label}")

        # affine aus der Nifti-Datei
        affine = total_seg_img.affine
        axcodes = nib.aff2axcodes(affine)

        # Z-Achse als längste Achse
        #axis_lengths = [seg_data.shape[i] * abs(affine[i,i]) for i in range(3)]
        #z_axis = np.argmax(axis_lengths)

        axcodes = nib.aff2axcodes(total_seg_img.affine)
        # S = superior, I = inferior
        z_axis = next((i for i, c in enumerate(axcodes) if c in ('S', 'I')), None)
        if z_axis is None:
            raise ValueError("Keine Kopf-Fuß-Achse gefunden!")

        # Humerus-Koordinaten
        humerus = np.isin(seg_data, [name_to_label["humerus_left"], name_to_label["humerus_right"]])
        coords = np.argwhere(humerus)
        if coords.size == 0:
            logger.error("No humerus found")
        else:
            coords_h = np.c_[coords, np.ones(len(coords))]
            hum_coords = coords_h @ affine.T
            hum_coords = hum_coords[:, :3]
            hum_z_min = np.percentile(hum_coords[:, z_axis], 5)

            # T4-Koordinaten
            t4 = (seg_data == name_to_label["vertebrae_T4"])
            t4_coords = np.argwhere(t4)
            coords_h_t4 = np.c_[t4_coords, np.ones(len(t4_coords))]
            t4_world = coords_h_t4 @ affine.T
            t4_world = t4_world[:, :3]
            t4_z_max = np.percentile(t4_world[:, z_axis], 95)

            # Vergleich
            if hum_z_min < t4_z_max - 100:
                logger.warning("Arms are beside the body")
                return fallback_dict

        fat_data = np.asanyarray(fat_img.dataobj).copy()

        #img nur im bereich(maske)
        if layer == "full_picture":
            fat_data = fat_data
            
        elif layer == "l3":
            l3 = (seg_data == name_to_label["vertebrae_L3"])
            l_min, l_max = np.argwhere(l3)[:, 0].min(), np.argwhere(l3)[:, 0].max()
            slice_mask = np.zeros_like(fat_data, dtype=bool)
            slice_mask[l_min:l_max+1, :, :] = True
            fat_data[~slice_mask] = 0

        elif layer == "glut_to_c6":
            glut = np.isin(seg_data, [name_to_label["gluteus_maximus_left"], name_to_label["gluteus_maximus_right"]])
            g_min = np.argwhere(glut)[:, 0].min()
            c6 = (seg_data == name_to_label["vertebrae_C6"])
            c_max = np.argwhere(c6)[:, 0].max()
            slice_mask = np.zeros_like(fat_data, dtype=bool)
            slice_mask[g_min:c_max+1, :, :] = True
            fat_data[~slice_mask] = 0
        
        voxel_spacing = fat_img.header.get_zooms()
        voxel_volume = np.prod(voxel_spacing)

        base_data = np.asanyarray(base_img.dataobj)
        labeled_data = fat_data

        base_mask = base_data > -1000
        total_vol = np.sum(base_mask) * voxel_volume / 1000
        unique, counts = np.unique(labeled_data, return_counts=True)
        label_counts = dict(zip(unique, counts))

        vol = {}
        for num, label in labels.items():
            vol[label] = label_counts.get(num, 0) * voxel_volume / 1000

        result_dict = {}
        total_fat = 0
        total_muscle = 0
        for label, vols in vol.items():
            result_dict[f"{label}_in_ml"] = vols
            if label.endswith("fat"):
                total_fat += vols
            elif label.endswith("muscle"):
                total_muscle += vols

        result_dict["total_fat_in_%"] = total_fat / total_vol * 100
        result_dict["total_muscle_in_%"] = total_muscle / total_vol * 100
        result_dict["muscle_fat_ratio"] = total_muscle / total_fat if total_fat > 0 else None
        return result_dict
    

    def convert_pet2sul(self, output_dirpath: str | os.PathLike, lean_bodymas:float, study_date:str) -> os.PathLike:
        """
        Coordinates the conversion of PET to SUL image.

        Args:
            path (os.PathLike): Path to the NIfTI images.
            lean_bodymas (float): Body weight without the weight of the fat.
            study_date (str): Date of the study for json access.

        Returns:
            path (os.PathLike): Path to the SUL NIfTI image.
        """
        out_pet_fpath = os.path.join(output_dirpath, "PET.nii.gz")
        out_sul_fpath = os.path.join(output_dirpath, "SUL.nii.gz")  

        if os.path.isfile(out_sul_fpath):
            logger.info(f"SUL NIfTI already exist at {out_pet_fpath}")
            return 
        else:      
            sul_corr_factor = self.load_sul_faktor(output_dirpath, lean_bodymas, study_date)

            sul_pet_nii = self.convert_pet(
                nib.load(out_pet_fpath),
                sul_factor=sul_corr_factor,  # type: ignore
            )
            nib.save(img=sul_pet_nii, filename=out_sul_fpath)  # type: ignore
            
            return out_sul_fpath

    def time_to_seconds(self, t: str | float | int) -> float:
        """
        Converts time as str to float to seconds after 00:00, 

        Args:
            t (str | float | int): Time as str in the formate of HHMMSS.MS or HH:MM:SS.MS.
            
        Returns:
            float: Time in seconds.  
        """
        t = str(t).strip()

        if ":" in t:
            h, m, s = t.split(":")
            return int(h)*3600 + int(m)*60 + float(s)

        h = int(t[0:2])
        m = int(t[2:4])
        s = float(t[4:])
        return h*3600 + m*60 + s

    def load_sul_faktor(self, path:os.PathLike, lean_bodymas:float, study_date:str) -> float:
        """
        Reads the safed patient_info.json file to get the PET infos and calculatest the SUL factor.

        Args:
            path (os.PathLike): Path to the NIfTI images.
            lean_bodymas (float): Body weight without the weight of the fat.
            study_date (str): Date of the study for json access.

        Returns:
            float: SUL factor.  
        """
        with open(os.path.join(os.path.dirname(path), "patient_info.json"), "r") as f:
            data = json.load(f)

        series_name = next(
            iter(data["Studies"][study_date]["Modalities"]["PT"][0])
        )
        total_dose = data["Studies"][study_date]["Modalities"]["PT"][0][series_name]["DICOM"]["InjectedRadioactivity"]
        half_life = data["Studies"][study_date]["Modalities"]["PT"][0][series_name]["DICOM"]["RadionuclideHalfLife"]
        acq_time = data["Studies"][study_date]["Modalities"]["PT"][0][series_name]["DICOM"]["AcquisitionTime"]
        start_time = data["Studies"][study_date]["Modalities"]["PT"][0][series_name]["DICOM"]["RadiopharmaceuticalStartTime"]

        time_diff = self.time_to_seconds(acq_time) - self.time_to_seconds(start_time)
        act_dose = total_dose * 0.5 ** (time_diff / half_life)
        sul_factor = 1000 * lean_bodymas / act_dose

        return sul_factor

    def convert_pet(self, pet, sul_factor) -> nib.Nifti1Image:
        """Conversion of PET values to SUL (should work on Siemens PET/CT)"""
        affine = pet.affine
        pet_data = pet.get_fdata()
        pet_suv_data = (pet_data * sul_factor).astype(np.float32)
        pet_suv = nib.Nifti1Image(pet_suv_data, affine)  # type: ignore
        return pet_suv

def totalsegmentator_muscle_fat_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.totalsegmentator_muscle_fat")

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run TotalSegmentator for muscle and fat (ml option) on all CT.nii.gz files in a folder. "
        "Extract label mapping from the segmentation output and save metadata as CT_muscle_fat.json."
    )
    parser.add_argument(
        "--input-dirpath-processed", type=str, help="Path to the input folder containing CT.nii.gz files", required=True
    )
    args = parser.parse_args()

    TotalSegmentatorMuscleFat(
        input_dirpath_processed=args.input_dirpath_processed,
    ).run()


if __name__ == "__main__":
    totalsegmentator_muscle_fat_entrypoint()
