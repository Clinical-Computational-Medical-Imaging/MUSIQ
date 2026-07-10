import json
import logging
import os
import pathlib as plb

import nibabel as nib
import numpy as np
from totalsegmentator.nifti_ext_header import load_multilabel_nifti

from .utils import is_mr_filename, list_patient_dirs, load_mr_keywords

logger = logging.getLogger(__name__)


def total_seg_path(input_fpath: str | os.PathLike) -> str:
    """Path to the TotalSegmentator 'total' segmentation for a given input image.

    ``CT.nii.gz`` -> ``CTseg.nii.gz``; MR ``<stem>.nii.gz`` -> ``<stem>_seg.nii.gz``.
    """
    dirpath, filename = os.path.split(str(input_fpath))
    if filename == "CT.nii.gz":
        return os.path.join(dirpath, "CTseg.nii.gz")
    return os.path.join(dirpath, f"{filename[:-7]}_seg.nii.gz")


class TotalSegmentatorMuscleFat:
    def __init__(self, input_dirpath_processed: os.PathLike | str) -> None:
        """Compute muscle/fat body composition and LBM from CT (and segmentation-only for MR)."""
        self.input_dirpath = input_dirpath_processed

    def run(self) -> None:
        """Process all CT/MR images: segment, compute body composition, record stats."""
        if not os.path.isdir(self.input_dirpath):
            logger.error(f"Error: {self.input_dirpath} is not a valid directory.")
            return

        logger.info(f"Starting TotalSegmentator inference in {self.input_dirpath}")
        mr_keywords = load_mr_keywords()
        top_dirs = list_patient_dirs(self.input_dirpath)

        for top_dir in top_dirs:
            top_dir_path = os.path.join(self.input_dirpath, top_dir)

            for dirpath, dirnames, filenames in os.walk(top_dir_path):
                rel_parts = plb.Path(os.path.relpath(dirpath, self.input_dirpath)).parts
                if len(rel_parts) != 2:
                    continue
                dirnames.clear()
                patient_id, study_date = rel_parts
                for filename in filenames:
                    try:
                        # Skip existing muscle/fat segmentations
                        if filename.endswith("_muscle_fat.nii.gz"):
                            continue

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

                        patient_info_path = os.path.join(os.path.dirname(dirpath), "patient_info.json")
                        # Fully processed only once seg exists AND body-composition stats are recorded (so re-runs skip)
                        if (
                            is_ct
                            and os.path.isfile(output_fpath)
                            and self._bca_recorded(patient_info_path, study_date, modality)
                        ):
                            logger.info(f"muscle/fat analysis already complete for {patient_id}, skipping.")
                            # Backfill series-level muscle/fat path (consistent with CTsegPath/CTcadsPath)
                            self._record_muscle_fat_path(
                                patient_info_path,
                                study_date,
                                modality,
                                series_index=0,
                                filename=filename,
                                seg_path_key=seg_path_key,
                                output_fpath=output_fpath,
                            )
                            # Recompute LBM in case PatientWeight was corrected since first run
                            self._backfill_lbm(patient_info_path, study_date, modality, patient_id, series_index=0)
                            continue

                        segmentation_exists = os.path.isfile(output_fpath)

                        # MR: segmentation only (fat%/LBM/SUL are CT/PET concepts)
                        if not is_ct and segmentation_exists:
                            logger.info(f"MR segmentation already exists for {filename}, skipping.")
                            continue

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
                                and modality
                                in patient_info.get("Studies", {}).get(study_date, {}).get("Modalities", {})
                            ):
                                mr_series = patient_info["Studies"][study_date]["Modalities"][modality]
                                series_index = None
                                for idx, serie in enumerate(mr_series):
                                    for serie_data in serie.values():
                                        if "MRPath" in serie_data and filename in os.path.basename(
                                            serie_data["MRPath"]
                                        ):
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

                            # Lazy import: torch/nnU-Net CUDA init is slow; skip it when all studies are skipped
                            if not os.path.isfile(output_fpath):
                                from totalsegmentator.python_api import totalsegmentator

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

                            # MR: segmentation only (fat%/LBM/SUL are CT/PET concepts)
                            if not is_ct:
                                continue

                            # Lazy import (pulls in torch)
                            from totalsegmentator.config import get_version

                            seg_metadata = {
                                "settings": {"input_fpath": input_fpath, "task": task, "ml": True},
                                "model": "total",
                                "ts_version": get_version(),
                            }

                            layers = ["full_picture", "l3", "glut_to_c6"]
                            # calc_size loads image+seg once, measures every layer (arms masked out first);
                            # fallback recorded for missing seg/landmark
                            layer_results = self.calc_size(input_fpath, segmentation_img, label_map_dict, layers)

                            if json_exists and patient_info is not None:
                                series_name = next(
                                    iter(patient_info["Studies"][study_date]["Modalities"][modality][series_index])
                                )
                                analysis_dict = patient_info["Studies"][study_date]["Modalities"][modality][
                                    series_index
                                ][series_name].setdefault("body_composition_analysis", {})
                                analysis_dict[seg_path_key] = output_fpath
                                analysis_dict[metadata_key] = seg_metadata
                                for layer, calculation in layer_results.items():
                                    analysis_dict.setdefault(layer, {}).update(calculation)
                                # Persist once, after all layers — each dump rewrites the whole patient_info.
                                with open(patient_info_path, "w") as f:
                                    json.dump(patient_info, f)
                            else:
                                with open(os.path.join(dirpath, f"{filename[:-7]}_muscle_fat.json"), "w") as f:
                                    json.dump(seg_metadata, f)
                        else:
                            logger.info(f"CT_muscle_fat.nii.gz already exists for {patient_id}, skipping segmentation.")

                        if not json_exists:
                            logger.error(f"Cannot compute LBM for {patient_id}: patient_info.json is missing.")
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
                        # Record muscle/fat seg path at series level; persist now (LBM block below may continue out)
                        if os.path.isfile(output_fpath) and series_data.get(seg_path_key) != output_fpath:
                            series_data[seg_path_key] = output_fpath
                            with open(patient_info_path, "w") as f:
                                json.dump(data, f)
                        # PatientWeight may be missing or a string; coerce to float
                        patient_weight = series_data.get("DICOM", {}).get("PatientWeight")
                        if patient_weight is None:
                            logger.warning(
                                f"No PatientWeight in patient_info.json for {patient_id}, skipping LBM computation."
                            )
                            continue
                        try:
                            weight = float(patient_weight)
                        except (TypeError, ValueError):
                            logger.warning(
                                f"Invalid PatientWeight {patient_weight!r} for {patient_id}, skipping LBM computation."
                            )
                            continue
                        if weight <= 0:
                            logger.warning(
                                f"Non-positive PatientWeight ({weight}) for {patient_id}, skipping LBM computation."
                            )
                            continue
                        fat_in_percent = (
                            series_data.get("body_composition_analysis", {}).get("glut_to_c6", {}).get("total_fat_in_%")
                        )
                        if fat_in_percent is None:
                            seg_path = total_seg_path(input_fpath)
                            if not os.path.isfile(seg_path):
                                reason = f"seg not found at {seg_path}; run TotalSegmentator 'total' first"
                            else:
                                reason = "arms beside body; LBM unreliable"
                            logger.warning(f"Skipping LBM computation for {patient_id}: {reason}.")
                            continue
                        # LBM feeds the sul stage (via PatientLBM)
                        lean_body_mass = weight * (1 - fat_in_percent / 100)
                        data["Studies"][study_date]["Modalities"][modality][series_index][series_name]["PatientLBM"] = (
                            lean_body_mass
                        )

                        with open(patient_info_path, "w") as f:
                            json.dump(data, f)
                    except Exception:
                        logger.exception(f"Failed to process {filename} for {patient_id} ({study_date}); skipping.")

    def _bca_recorded(self, patient_info_path: str | os.PathLike, study_date: str, modality: str) -> bool:
        """True if body-composition stats (glut_to_c6) are recorded for this study."""
        if not os.path.isfile(patient_info_path):
            return False
        try:
            with open(patient_info_path) as f:
                data = json.load(f)
            # CT is always series 0 (see below); this helper is only consulted for CT-only studies.
            series_data = next(iter(data["Studies"][study_date]["Modalities"][modality][0].values()))
        except (KeyError, IndexError, TypeError, StopIteration, json.JSONDecodeError):
            return False
        return "glut_to_c6" in series_data.get("body_composition_analysis", {})

    def _record_muscle_fat_path(
        self,
        patient_info_path: str | os.PathLike,
        study_date: str,
        modality: str,
        series_index: int | None,
        filename: str,
        seg_path_key: str,
        output_fpath: str | os.PathLike,
    ) -> None:
        """Store the muscle/fat seg path at series level (idempotent)."""
        if not os.path.isfile(output_fpath) or not os.path.isfile(patient_info_path):
            return
        with open(patient_info_path) as f:
            data = json.load(f)

        try:
            series_list = data["Studies"][study_date]["Modalities"][modality]
        except (KeyError, TypeError):
            logger.warning(f"Cannot record {seg_path_key}: {modality} {study_date} not in {patient_info_path}.")
            return

        if series_index is None:
            # MR: resolve the series by matching the source filename against MRPath.
            series_index = next(
                (
                    idx
                    for idx, serie in enumerate(series_list)
                    for serie_data in serie.values()
                    if "MRPath" in serie_data and filename in os.path.basename(serie_data["MRPath"])
                ),
                None,
            )
        if series_index is None:
            logger.warning(f"Cannot record {seg_path_key}: no series matching {filename} in {patient_info_path}.")
            return

        series_name = next(iter(series_list[series_index]))
        series_data = series_list[series_index][series_name]
        if series_data.get(seg_path_key) == output_fpath:
            return
        series_data[seg_path_key] = output_fpath
        with open(patient_info_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Recorded {seg_path_key} for {study_date} in patient_info.json.")

    def _backfill_lbm(
        self,
        patient_info_path: str | os.PathLike,
        study_date: str,
        modality: str,
        patient_id: str,
        series_index: int = 0,
    ) -> None:
        """Recompute and store PatientLBM if weight/fat% changed since first run (idempotent)."""
        if not os.path.isfile(patient_info_path):
            return
        with open(patient_info_path) as f:
            data = json.load(f)
        try:
            series_list = data["Studies"][study_date]["Modalities"][modality]
            series_name = next(iter(series_list[series_index]))
            series_data = series_list[series_index][series_name]
        except (KeyError, IndexError, TypeError, StopIteration):
            return

        patient_weight = series_data.get("DICOM", {}).get("PatientWeight")
        if patient_weight is None:
            return
        try:
            weight = float(patient_weight)
        except (TypeError, ValueError):
            logger.warning(f"Invalid PatientWeight {patient_weight!r} for {patient_id}, cannot backfill LBM.")
            return
        if weight <= 0:
            return
        fat_in_percent = series_data.get("body_composition_analysis", {}).get("glut_to_c6", {}).get("total_fat_in_%")
        if fat_in_percent is None:
            return

        lean_body_mass = weight * (1 - fat_in_percent / 100)
        if series_data.get("PatientLBM") == lean_body_mass:
            return  # already up to date
        series_data["PatientLBM"] = lean_body_mass
        with open(patient_info_path, "w") as f:
            json.dump(data, f)
        logger.info(f"Backfilled PatientLBM ({lean_body_mass:.2f}) for {patient_id} ({study_date}).")

    def calc_size(
        self, path: os.PathLike, fat_img: nib.Nifti1Image, labels: dict[int, str], layers: list[str]
    ) -> dict[str, dict]:
        """Compute body-composition stats (label volumes + fat/muscle %) for each requested layer.

        The base image and 'total' seg are loaded once (with the arms-beside-body check) and reused
        for every layer. Returns ``{layer: result_dict}``; a missing seg or landmark yields the fallback.
        """
        fallback = {"total_fat_in_%": None, "total_muscle_in_%": None, "muscle_fat_ratio": None}
        all_fallback = {layer: dict(fallback) for layer in layers}

        seg_path = total_seg_path(path)
        if not os.path.isfile(seg_path):
            logger.error(f"{seg_path} is missing. Please run TotalSegmentator task='total' first.")
            return all_fallback

        base_img = nib.load(path)
        total_seg_img, label_map_dict = load_multilabel_nifti(seg_path)
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
            hum_z_min = np.percentile((coords_h @ affine.T)[:, z_axis], 5)

            t4_coords = np.argwhere(seg_data == name_to_label["vertebrae_T4"])
            coords_h_t4 = np.c_[t4_coords, np.ones(len(t4_coords))]
            t4_z_max = np.percentile((coords_h_t4 @ affine.T)[:, z_axis], 95)

            if hum_z_min < t4_z_max - 100:
                logger.warning("Arms are beside the body, removing arms before measuring.")
                fat_img = self.remove_arms(fat_img, seg_data, name_to_label, affine, z_axis, path)
                nib.save(fat_img, os.path.join(os.path.dirname(path), "CTbody_masked.nii.gz"))

        voxel_volume = np.prod(fat_img.header.get_zooms())
        fat_full = np.asanyarray(fat_img.dataobj)
        # Body mask on the base image (everything denser than air). fat/muscle % is a fraction of the
        # body volume *within the same axial range as the layer*, so the denominator is restricted to
        # the layer's slice bounds below — not the whole scan FOV. Using the whole-scan volume made the
        # narrower layers (l3, glut_to_c6) report systematically low fat% on whole-body scans.
        body_mask = np.asanyarray(base_img.dataobj) > -1000

        results: dict[str, dict] = {}
        for i, layer in enumerate(layers):
            fat_data = fat_full.copy()
            # Axial (array axis 0) bounds defining the layer; default is the full extent (full_picture).
            z_lo, z_hi = 0, fat_data.shape[0] - 1

            if layer == "l3":
                l3_coords = np.argwhere(seg_data == name_to_label["vertebrae_L3"])
                if l3_coords.size == 0:
                    logger.warning("vertebrae_L3 not found in segmentation; cannot define L3 layer")
                    results.update({rem: dict(fallback) for rem in layers[i:]})
                    break
                l_min, l_max = l3_coords[:, 0].min(), l3_coords[:, 0].max()
                z_lo, z_hi = l_min, l_max
                fat_data[:l_min] = 0
                fat_data[l_max + 1 :] = 0

            elif layer == "glut_to_c6":
                glut = np.isin(
                    seg_data, [name_to_label["gluteus_maximus_left"], name_to_label["gluteus_maximus_right"]]
                )
                glut_coords = np.argwhere(glut)
                c6_coords = np.argwhere(seg_data == name_to_label["vertebrae_C6"])
                if glut_coords.size == 0 or c6_coords.size == 0:
                    missing = []
                    if glut_coords.size == 0:
                        missing.append("gluteus_maximus")
                    if c6_coords.size == 0:
                        missing.append("vertebrae_C6")
                    logger.warning(f"{' and '.join(missing)} not found in segmentation; cannot define glut_to_c6 layer")
                    results.update({rem: dict(fallback) for rem in layers[i:]})
                    break
                g_min = glut_coords[:, 0].min()
                c_max = c6_coords[:, 0].max()
                z_lo, z_hi = g_min, c_max
                fat_data[:g_min] = 0
                fat_data[c_max + 1 :] = 0

            # Body volume within the same axial range as the (masked) fat numerator.
            total_vol = np.sum(body_mask[z_lo : z_hi + 1]) * voxel_volume / 1000

            unique, counts = np.unique(fat_data, return_counts=True)
            label_counts = dict(zip(unique, counts, strict=False))

            result_dict = {}
            total_fat = 0
            total_muscle = 0
            for num, label in labels.items():
                vols = label_counts.get(num, 0) * voxel_volume / 1000
                result_dict[f"{label}_in_ml"] = vols
                if label.endswith("fat"):
                    total_fat += vols
                elif label.endswith("muscle"):
                    total_muscle += vols

            if total_vol > 0:
                result_dict["total_fat_in_%"] = total_fat / total_vol * 100
                result_dict["total_muscle_in_%"] = total_muscle / total_vol * 100
            else:
                logger.warning(f"Empty body volume for layer '{layer}'; cannot compute fat/muscle %.")
                result_dict["total_fat_in_%"] = None
                result_dict["total_muscle_in_%"] = None
            result_dict["muscle_fat_ratio"] = total_muscle / total_fat if total_fat > 0 else None
            results[layer] = result_dict

        return results

    def remove_arms(
        self, tissue_img: nib.Nifti1Image, organ_img, labels, affine, z_axis, input_fpath: os.PathLike
    ) -> nib.Nifti1Image:
        """Zero out arm voxels in the tissue seg using the TotalSegmentator body mask (arms-down studies)."""
        output_fpath = os.path.join(os.path.dirname(input_fpath), "CTbody.nii.gz")
        if not os.path.isfile(output_fpath):
            # Lazy import: pulls in torch/nnU-Net; only needed for arms-down studies.
            from totalsegmentator.python_api import totalsegmentator

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
                logger.info("Body segmentation successfully completed.")
            except Exception as e:
                logger.error(f"Error during body segmentation for {input_fpath}:\n  {e}")
                return tissue_img
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

        data = np.asanyarray(tissue_img.dataobj).copy()
        for z in range(data.shape[2]):
            if torso_extended[:, :, z].any():
                data[:, :, z] *= torso_extended[:, :, z].astype(data.dtype)

        return nib.Nifti1Image(data, tissue_img.affine, tissue_img.header)


def totalsegmentator_muscle_fat_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.totalsegmentator_muscle_fat")

    import argparse

    parser = argparse.ArgumentParser(
        description="Run TotalSegmentator muscle/fat on all CT/MR images, recording body-composition stats and LBM."
    )
    parser.add_argument(
        "--input-dirpath-processed",
        type=str,
        help="Path to the input folder containing the patient folders with studies and nifti files",
        required=True,
    )
    args = parser.parse_args()

    TotalSegmentatorMuscleFat(
        input_dirpath_processed=args.input_dirpath_processed,
    ).run()


if __name__ == "__main__":
    totalsegmentator_muscle_fat_entrypoint()
