import json
import logging
import os
import pathlib as plb

import nibabel as nib
import numpy as np
import pydicom
from totalsegmentator.config import get_version
from totalsegmentator.nifti_ext_header import load_multilabel_nifti
from totalsegmentator.python_api import totalsegmentator

from .utils import calculate_suv_factor, convert_pet, is_mr_filename, load_mr_keywords, natural_key

logger = logging.getLogger(__name__)


class TotalSegmentatorMuscleFatSUL:
    def __init__(self, input_dirpath_processed: os.PathLike | str) -> None:
        """Class to handle TotalSegmentator muscle fat analysis on CT.nii.gz and MRI files in a specified folder.
        It processes each file, runs segmentation, extracts label mapping and computes a SUL image.
        Creates CT_muscle_fat.nii and SUL.nii.gz.

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the CT.nii.gz files. Can be nested.
        """
        self.input_dirpath = input_dirpath_processed

    def run(self) -> None:
        """
        Recursively search the folder for CT.nii.gz files.
        For each found file, run the tissue_4_types for CT or tissue_types_mr for MRI segmentation,
        extract the label mapping from the segmentation output, and save a metadata JSON file.
        It also calculates the lean body mass and starts the SUL image creation.
        """
        if not os.path.isdir(self.input_dirpath):
            logger.error(f"Error: {self.input_dirpath} is not a valid directory.")
            return

        logger.info(f"Starting TotalSegmentator inference in {self.input_dirpath}")
        mr_keywords = load_mr_keywords()
        top_dirs = [d for d in os.listdir(self.input_dirpath) if os.path.isdir(os.path.join(self.input_dirpath, d))]
        top_dirs.sort(key=natural_key)

        for top_dir in top_dirs:
            top_dir_path = os.path.join(self.input_dirpath, top_dir)

            for dirpath, dirnames, filenames in os.walk(top_dir_path):
                rel_parts = plb.Path(os.path.relpath(dirpath, self.input_dirpath)).parts
                if len(rel_parts) != 2:
                    continue
                dirnames.clear()
                patient_id, study_date = rel_parts
                for filename in filenames:
                    # Determine if this is a CT or MR file and set parameters accordingly
                    is_ct = filename == "CT.nii.gz"
                    is_mr = (
                        filename.endswith("nii.gz")
                        and not filename.startswith(("CT", "SUV", "PET"))
                        and not filename.endswith("seg.nii.gz")
                        and is_mr_filename(filename, mr_keywords)
                    )

                    if not (is_ct or is_mr):
                        continue
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

                    sul_fpath = os.path.join(dirpath, "SUL.nii.gz")
                    if os.path.isfile(output_fpath) and os.path.isfile(sul_fpath):
                        logger.info(f"CT_muscle_fat.nii.gz and SUL.nii.gz already exist for {patient_id}, skipping.")
                        continue

                    segmentation_exists = os.path.isfile(output_fpath)

                    patient_dirpath = os.path.dirname(dirpath)
                    patient_info_path = os.path.join(patient_dirpath, "patient_info.json")
                    if os.path.isfile(patient_info_path):
                        json_exists = True
                        with open(patient_info_path) as json_file:
                            patient_info = json.load(json_file)
                    else:
                        json_exists = False
                        patient_info = None
                        logger.error(f"Missing patient_info.json in {patient_dirpath}.")

                    if modality == "CT":
                        series_index = 0
                    else:
                        if (
                            json_exists
                            and patient_info is not None
                            and modality in patient_info.get("Studies", {}).get(study_date, {}).get("Modalities", {})
                        ):
                            mr_series = patient_info["Studies"][study_date]["Modalities"][modality]
                            series_index = None
                            for idx, serie in enumerate(mr_series):
                                for _serie_name, serie_data in serie.items():
                                    if "MRPath" in serie_data and filename in os.path.basename(serie_data["MRPath"]):
                                        series_index = idx
                                        break
                                if series_index is not None:
                                    break
                        else:
                            series_index = None

                    if series_index is None:
                        logger.error(f"Could not find series index for {filename} in patient_info.json.")
                        continue

                    study_in_json = (
                        patient_info is not None
                        and study_date in patient_info.get("Studies", {})
                        and modality in patient_info["Studies"][study_date].get("Modalities", {})
                    )
                    if segmentation_exists and json_exists and study_in_json:
                        series_name_check = next(
                            iter(patient_info["Studies"][study_date]["Modalities"][modality][series_index])
                        )
                        series_data = patient_info["Studies"][study_date]["Modalities"][modality][series_index][
                            series_name_check
                        ]
                        if "body_composition_analysis" not in series_data or "glut_to_c6" not in series_data.get(
                            "body_composition_analysis", {}
                        ):
                            logger.info(
                                f"body_composition_analysis missing for {patient_id},"
                                " recomputing from existing segmentation."
                            )
                            segmentation_exists = False

                    if not segmentation_exists:
                        logger.info(f"Processing file {filename} for patient {patient_id}.")

                        # Run TotalSegmentator using the Python API with ml option and appropriate task.
                        if not os.path.isfile(output_fpath):
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

                        layers = ["full_picture", "l3", "glut_to_c6"]
                        for layer in layers:
                            calculation = self.calc_size(input_fpath, segmentation_img, label_map_dict, layer)
                            if all(v is None for v in calculation.values()):
                                break  # seg missing or arms beside body — same result for all layers

                            seg_metadata = {
                                "settings": {"input_fpath": input_fpath, "task": task, "ml": True},
                                "model": "total",
                                "ts_version": get_version(),
                            }
                            if json_exists and patient_info is not None:
                                series_name = next(
                                    iter(patient_info["Studies"][study_date]["Modalities"][modality][series_index])
                                )
                                analysis_dict = patient_info["Studies"][study_date]["Modalities"][modality][
                                    series_index
                                ][series_name].setdefault("body_composition_analysis", {})
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
                    else:
                        logger.info(f"CT_muscle_fat.nii.gz already exists for {patient_id}, skipping segmentation.")

                    if not json_exists:
                        logger.error(f"Cannot compute LBM/SUL for {patient_id}: patient_info.json is missing.")
                        continue

                    with open(patient_info_path, encoding="utf-8") as f:
                        data = json.load(f)

                    if study_date not in data.get("Studies", {}):
                        logger.warning(
                            f"study_date {study_date} not found in patient_info.json for {patient_id}, skipping."
                        )
                        continue

                    series_name = next(iter(data["Studies"][study_date]["Modalities"][modality][series_index]))
                    series_data = data["Studies"][study_date]["Modalities"][modality][series_index][series_name]
                    weight = series_data["DICOM"]["PatientWeight"]
                    fat_in_percent = (
                        series_data.get("body_composition_analysis", {}).get("glut_to_c6", {}).get("total_fat_in_%")
                    )
                    if fat_in_percent is None:
                        total_seg_path = input_fpath.replace(".nii.gz", "seg.nii.gz")
                        if not os.path.isfile(total_seg_path):
                            reason = (
                                f"CTseg.nii.gz not found at {total_seg_path} — run TotalSegmentator task='total' first"
                            )
                        else:
                            reason = (
                                "arms detected beside the body (humerus below T4 threshold); "
                                "LBM cannot be estimated reliably"
                            )
                        logger.warning(f"Skipping LBM/SUL computation for {patient_id}: {reason}.")
                        continue
                    lean_body_mass = weight * (1 - fat_in_percent / 100)
                    data["Studies"][study_date]["Modalities"][modality][series_index][series_name]["PatientLBM"] = (
                        lean_body_mass
                    )

                    pet_path = os.path.join(dirpath, "PET.nii.gz")
                    if os.path.isfile(pet_path):
                        sul_path = self.convert_pet2sul(dirpath, lean_body_mass, study_date)
                        pt_series_name = next(iter(data["Studies"][study_date]["Modalities"]["PT"][0]))
                        data["Studies"][study_date]["Modalities"]["PT"][0][pt_series_name]["SULPath"] = sul_path
                    else:
                        logger.info(f"No PET file found for {patient_id}, skipping SUL computation.")

                    with open(patient_info_path, "w") as f:
                        json.dump(data, f)

    def calc_size(
        self, path: os.PathLike, fat_img: nib.Nifti1Image, labels: dict[int, str], layer: str
    ) -> dict[float, float, float]:
        """
        Calculate the volume (in ml) of each label in a segmentation and
        compute total fat and muscle percentages relative to the whole scan.

        Args:
            path (os.PathLike): Path to the original CT/MR NIfTI image.
            img (nib.Nifti1Image): Segmentation NIfTI image.
            labels (dict[int, str]): Mapping of label numbers to label names.

        Returns:
            dict[str, float]: Dictionary with volume per label (ml) and
                    total fat/muscle percentages to the body volume (%)
                    muscle/fat ratio.
        """
        fallback_dict = {"total_fat_in_%": None, "total_muscle_in_%": None, "muscle_fat_ratio": None}
        total_seg_path = str(path).replace(".nii.gz", "seg.nii.gz")
        if not os.path.isfile(total_seg_path):
            logger.error(f"{total_seg_path} is mising. Please run TotalSegmentator task='total' first.")
            return fallback_dict

        base_img = nib.load(path)
        total_seg_img, label_map_dict = load_multilabel_nifti(total_seg_path)

        seg_data = total_seg_img.get_fdata().astype(int)

        if base_img.shape != total_seg_img.shape:
            logger.error("Shape mismatch!")

        if not np.allclose(base_img.affine, total_seg_img.affine):
            logger.warning("Affine mismatch!")

        name_to_label = {v: k for k, v in label_map_dict.items()}

        affine = total_seg_img.affine
        axcodes = nib.aff2axcodes(affine)
        z_axis = next((i for i, c in enumerate(axcodes) if c in ("S", "I")), None)
        if z_axis is None:
            raise ValueError("No head-to-toe axis found")

        humerus = np.isin(seg_data, [name_to_label["humerus_left"], name_to_label["humerus_right"]])
        coords = np.argwhere(humerus)
        if coords.size == 0:
            logger.error("No humerus found")
        else:
            coords_h = np.c_[coords, np.ones(len(coords))]
            hum_coords = coords_h @ affine.T
            hum_coords = hum_coords[:, :3]
            hum_z_min = np.percentile(hum_coords[:, z_axis], 5)

            t4 = seg_data == name_to_label["vertebrae_T4"]
            t4_coords = np.argwhere(t4)
            coords_h_t4 = np.c_[t4_coords, np.ones(len(t4_coords))]
            t4_world = coords_h_t4 @ affine.T
            t4_world = t4_world[:, :3]
            t4_z_max = np.percentile(t4_world[:, z_axis], 95)

            if hum_z_min < t4_z_max - 100:
                logger.warning("Arms are beside the body, remvoing arms")
                fat_img = self.remove_arms(fat_img, seg_data, name_to_label, affine, z_axis, path)

                result_fpath = os.path.join(os.path.dirname(path), "CTbody_masked.nii.gz")
                nib.save(fat_img, result_fpath)

        fat_data = np.asanyarray(fat_img.dataobj).copy()

        if layer == "l3":
            l3 = seg_data == name_to_label["vertebrae_L3"]
            l_min, l_max = np.argwhere(l3)[:, 0].min(), np.argwhere(l3)[:, 0].max()
            slice_mask = np.zeros_like(fat_data, dtype=bool)
            slice_mask[l_min : l_max + 1, :, :] = True
            fat_data[~slice_mask] = 0

        elif layer == "glut_to_c6":
            glut = np.isin(seg_data, [name_to_label["gluteus_maximus_left"], name_to_label["gluteus_maximus_right"]])
            g_min = np.argwhere(glut)[:, 0].min()
            c6 = seg_data == name_to_label["vertebrae_C6"]
            c_max = np.argwhere(c6)[:, 0].max()
            slice_mask = np.zeros_like(fat_data, dtype=bool)
            slice_mask[g_min : c_max + 1, :, :] = True
            fat_data[~slice_mask] = 0

        voxel_spacing = fat_img.header.get_zooms()
        voxel_volume = np.prod(voxel_spacing)

        base_data = np.asanyarray(base_img.dataobj)
        labeled_data = fat_data

        base_mask = base_data > -1000
        total_vol = np.sum(base_mask) * voxel_volume / 1000
        unique, counts = np.unique(labeled_data, return_counts=True)
        label_counts = dict(zip(unique, counts, strict=False))

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

    def remove_arms(self, img: nib.Nifti1Image, organ_img:nib.Nifti1Image, labels, affine, z_axis, input_fpath: os.PathLike) -> nib.Nifti1Image:
        output_fpath = os.path.join(os.path.dirname(input_fpath), "CTbody.nii.gz")
        if not os.path.isfile(output_fpath):
            try:
                totalsegmentator(
                    input_fpath,
                    output_fpath,
                    ml=True,
                    task="body",
                    device="gpu:0",
                    statistics=False,
                    radiomics=False,
                )
                logger.info("Segmentation successfully completed.")
            except Exception as e:
                logger.error(f"Error during segmentation for {input_fpath}:\n  {e}")
                return img
        body_img, label_map = load_multilabel_nifti(output_fpath)

        name_to_label = {v: k for k, v in label_map.items()}
        torso = (body_img.get_fdata() == name_to_label["body_trunc"]).astype(np.uint8)
        z_slices = np.where(torso.any(axis=(0, 1)))[0]

        hip = np.isin(organ_img, [labels["hip_left"]])
        coords = np.argwhere(hip)
        if coords.size == 0:
            logger.error("No hip_left found")
            best_z_bot = z_slices[0] 
        else:
            best_z_bot = int(np.percentile(coords[:, z_axis], 10))  

        shoulder = np.isin(organ_img, [labels["scapula_left"]])
        coords = np.argwhere(shoulder)
        if coords.size == 0:
            logger.error("No scapula_left found")
            best_z_top = z_slices[-1] 
        else:
            best_z_top = int(np.percentile(coords[:, z_axis], 95))     

        torso_extended = torso.copy()
        torso_extended[:, :, best_z_top:] = 1
        torso_extended[:, :, :best_z_bot] = 1

        data = np.asanyarray(img.dataobj).copy()
        for z in range(data.shape[2]):
            if torso_extended[:, :, z].any():
                data[:, :, z] *= torso_extended[:, :, z].astype(data.dtype)

        return nib.Nifti1Image(data, img.affine, img.header)

    def convert_pet2sul(self, output_dirpath: str | os.PathLike, lean_body_mass: float, study_date: str) -> os.PathLike:
        """
        Coordinates the conversion of PET to SUL image.

        Args:
            path (os.PathLike): Path to the NIfTI images.
            lean_body_mass (float): Body weight without the weight of the fat.
            study_date (str): Date of the study for json access.

        Returns:
            path (os.PathLike): Path to the SUL NIfTI image.
        """
        out_pet_fpath = os.path.join(output_dirpath, "PET.nii.gz")
        out_sul_fpath = os.path.join(output_dirpath, "SUL.nii.gz")

        if os.path.isfile(out_sul_fpath):
            logger.info(f"SUL NIfTI already exist at {out_sul_fpath}")
            return out_sul_fpath
        else:
            patient_info_path = os.path.join(os.path.dirname(output_dirpath), "patient_info.json")
            with open(patient_info_path) as f:
                data = json.load(f)

            series_name = next(iter(data["Studies"][study_date]["Modalities"]["PT"][0]))
            pt_series = data["Studies"][study_date]["Modalities"]["PT"][0][series_name]
            dicom_data = pt_series["DICOM"]
            required = [
                "InjectedRadioactivity",
                "RadionuclideHalfLife",
                "AcquisitionTime",
                "RadiopharmaceuticalStartTime",
            ]
            missing = [k for k in required if k not in dicom_data]
            if missing:
                input_dir = pt_series.get("InputDirPath")
                if input_dir and os.path.isdir(input_dir):
                    logger.info(f"Missing DICOM fields {missing}, recovering from {input_dir}.")
                    try:
                        first_dcm = os.listdir(input_dir)[0]
                        ds = pydicom.dcmread(os.path.join(input_dir, first_dcm))
                        seq = ds.RadiopharmaceuticalInformationSequence[0]
                        recoverable = {
                            "RadiopharmaceuticalStartTime": str(seq.RadiopharmaceuticalStartTime),
                            "InjectedRadioactivity": float(seq.RadionuclideTotalDose),
                            "RadionuclideHalfLife": float(seq.RadionuclideHalfLife),
                        }
                        for k in missing:
                            if k in recoverable:
                                dicom_data[k] = recoverable[k]
                        with open(patient_info_path, "w") as f:
                            json.dump(data, f)
                    except Exception as e:
                        logger.error(f"Failed to recover DICOM fields from {input_dir}: {e}")
                        return None
                else:
                    logger.error(
                        f"Cannot compute SUL for {output_dirpath}: missing DICOM fields {missing} "
                        "and InputDirPath is not accessible. Re-run series_selection to repopulate."
                    )
                    return None
                still_missing = [k for k in required if k not in dicom_data]
                if still_missing:
                    logger.error(f"Could not recover fields {still_missing} from DICOM. Skipping SUL.")
                    return None
            total_dose = dicom_data["InjectedRadioactivity"]
            half_life = dicom_data["RadionuclideHalfLife"]
            acq_time = dicom_data["AcquisitionTime"]
            start_time = dicom_data["RadiopharmaceuticalStartTime"]

            sul_corr_factor = calculate_suv_factor(total_dose, start_time, half_life, acq_time, lean_body_mass)

            sul_pet_nii = convert_pet(
                nib.load(out_pet_fpath),
                suv_factor=sul_corr_factor,  # type: ignore
            )
            nib.save(img=sul_pet_nii, filename=out_sul_fpath)  # type: ignore

            return out_sul_fpath


def totalsegmentator_muscle_fat_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.totalsegmentator_muscle_fat")

    import argparse

    parser = argparse.ArgumentParser(
        description="Recursively run TotalSegmentator for muscle and fat (ml option) on all CT.nii.gz or "
        "MRI files in a folder. It processes each file, runs segmentation, extracts label mapping and "
        "computes a SUL image. Creates CT_muscle_fat.nii, SUL.nii.gz images and extens patient_info.json"
    )
    parser.add_argument(
        "--input-dirpath-processed",
        type=str,
        help="Path to the input folder containing the patient folders with studies and nifti files",
        required=True,
    )
    args = parser.parse_args()

    TotalSegmentatorMuscleFatSUL(
        input_dirpath_processed=args.input_dirpath_processed,
    ).run()


if __name__ == "__main__":
    totalsegmentator_muscle_fat_entrypoint()
