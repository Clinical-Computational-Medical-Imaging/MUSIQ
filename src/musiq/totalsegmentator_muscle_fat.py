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

    Mirrors the naming produced by the totalsegmentator stage (see
    totalsegmentator_inference.py): ``CT.nii.gz`` -> ``CTseg.nii.gz`` while an MR series
    ``<stem>.nii.gz`` -> ``<stem>_seg.nii.gz``. A plain ``.nii.gz`` -> ``seg.nii.gz``
    replacement only works for CT (it drops the underscore for MR), which is why the seg
    was never found for MR series.
    """
    dirpath, filename = os.path.split(str(input_fpath))
    if filename == "CT.nii.gz":
        return os.path.join(dirpath, "CTseg.nii.gz")
    return os.path.join(dirpath, f"{filename[:-7]}_seg.nii.gz")


class TotalSegmentatorMuscleFat:
    def __init__(self, input_dirpath_processed: os.PathLike | str) -> None:
        """Class to handle TotalSegmentator muscle fat analysis on CT.nii.gz and MRI files in a specified folder.
        It processes each file, runs segmentation, extracts label mapping, computes body-composition
        stats and the lean body mass (LBM). Creates CT_muscle_fat.nii.gz and stores stats/LBM in
        patient_info.json. SUL is produced by the separate ``sul`` stage (see sul_computation.py).

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the CT.nii.gz files. Can be nested.
        """
        self.input_dirpath = input_dirpath_processed

    def run(self) -> None:
        """
        Recursively search the folder for CT.nii.gz files.
        For each found file, run the tissue_4_types for CT or tissue_types_mr for MRI segmentation,
        extract the label mapping from the segmentation output, and save a metadata JSON file.
        It also calculates the lean body mass (LBM) from the fat percentage obtained from the
        segmentation and the patient weight. The SUL image is produced by the separate ``sul`` stage.
        """
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
                        # Skip existing muscle/fat segmentations - we don't want to segment them again.
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
                        # A CT study is fully processed once the segmentation exists AND its body-composition
                        # stats are recorded. Recording the stats (not just the seg file) is what lets re-runs
                        # skip instead of reloading the whole-body seg and recomputing. SUL is a separate stage.
                        if (
                            is_ct
                            and os.path.isfile(output_fpath)
                            and self._bca_recorded(patient_info_path, study_date, modality)
                        ):
                            logger.info(f"muscle/fat analysis already complete for {patient_id}, skipping.")
                            # Backfill the series-level muscle/fat path for patients processed before this
                            # path was recorded (consistent with CTsegPath / CTcadsPath).
                            self._record_muscle_fat_path(
                                patient_info_path,
                                study_date,
                                modality,
                                series_index=0,
                                filename=filename,
                                seg_path_key=seg_path_key,
                                output_fpath=output_fpath,
                            )
                            # The body-composition stats already exist, but LBM may still be missing or
                            # stale — e.g. PatientWeight was absent/0 when the stats were first computed and
                            # has since been corrected. Recompute it here (cheap arithmetic, no seg reload)
                            # so a re-run picks up the new weight instead of skipping LBM entirely.
                            self._backfill_lbm(patient_info_path, study_date, modality, patient_id, series_index=0)
                            continue

                        segmentation_exists = os.path.isfile(output_fpath)

                        # For MR we only produce the segmentation file (no muscle/fat %, LBM or SUL),
                        # so if it already exists there is nothing left to do.
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

                            # Run TotalSegmentator using the Python API with ml option and appropriate task.
                            # Imported lazily: pulls in torch/nnU-Net (slow CUDA init), so a run where every
                            # study is skipped never pays that startup cost.
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

                            # For MR we only want the segmentation file — muscle/fat % (and the LBM/SUL
                            # it feeds) are CT/PET concepts, so skip them here.
                            if not is_ct:
                                continue

                            # Imported lazily: totalsegmentator.config pulls in torch, so keep it off the
                            # module-import path where every study may be skipped.
                            from totalsegmentator.config import get_version

                            seg_metadata = {
                                "settings": {"input_fpath": input_fpath, "task": task, "ml": True},
                                "model": "total",
                                "ts_version": get_version(),
                            }

                            layers = ["full_picture", "l3", "glut_to_c6"]
                            # calc_size loads the base image + whole-body seg once and measures every
                            # layer in a single pass (arms-down studies have the arms masked out first).
                            # A fallback (missing seg / missing landmark) is recorded for the affected
                            # layers so the completeness check above stays satisfied and re-runs don't
                            # reload and recompute (~10s).
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
                        # Record the muscle/fat segmentation path at the series level (consistent with how
                        # CTsegPath / CTcadsPath are stored), independent of whether body-composition stats
                        # were computed. Persist immediately, since the LBM block below may `continue`
                        # out before the final json.dump.
                        if os.path.isfile(output_fpath) and series_data.get(seg_path_key) != output_fpath:
                            series_data[seg_path_key] = output_fpath
                            with open(patient_info_path, "w") as f:
                                json.dump(data, f)
                        # PatientWeight may be missing entirely, or stored as a string (e.g. "80", "83.0")
                        # in patient_info.json, which would make `weight * (...)` raise
                        # "can't multiply sequence by non-int".
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
                                reason = (
                                    f"segmentation not found at {seg_path} — run TotalSegmentator task='total' first"
                                )
                            else:
                                reason = (
                                    "arms detected beside the body (humerus below T4 threshold); "
                                    "LBM cannot be estimated reliably"
                                )
                            logger.warning(f"Skipping LBM computation for {patient_id}: {reason}.")
                            continue
                        # LBM is consumed by the separate `sul` stage (via PatientLBM) to build SUL.nii.gz.
                        lean_body_mass = weight * (1 - fat_in_percent / 100)
                        data["Studies"][study_date]["Modalities"][modality][series_index][series_name]["PatientLBM"] = (
                            lean_body_mass
                        )

                        with open(patient_info_path, "w") as f:
                            json.dump(data, f)
                    except Exception:
                        logger.exception(f"Failed to process {filename} for {patient_id} ({study_date}); skipping.")

    def _bca_recorded(self, patient_info_path: str | os.PathLike, study_date: str, modality: str) -> bool:
        """True if body-composition stats (``glut_to_c6``) are already stored for this study's series.

        Mirrors the completeness check further down (see the ``body_composition_analysis`` /
        ``glut_to_c6`` guard): used by the skip gate so CT-only studies are only skipped once their
        stats have actually been computed, not merely because the segmentation file exists.
        """
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
        """Store the muscle/fat segmentation path at the series level in patient_info.json.

        Mirrors how CTsegPath / CTcadsPath are recorded. Idempotent: only writes when the key is
        missing or stale, so it can safely backfill already-processed patients on a re-run.
        """
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
        """Compute and store PatientLBM for an already-processed study, if it can be (re)derived.

        Used by the skip gate: when a CT study's body-composition stats already exist we don't want to
        redo the segmentation, but LBM may be missing or stale (e.g. PatientWeight was absent/0 at first
        run and has since been corrected). LBM is pure arithmetic on the recorded weight and fat%, so we
        recompute it cheaply here. Idempotent: only writes when the value is missing or actually changes.
        """
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

        The base image and the whole-body 'total' segmentation are layer-independent, so they are
        loaded — and the arms-beside-body check run — once here, then reused for every layer. (The
        previous implementation took a single ``layer`` and reloaded both from disk on every call.)

        Args:
            path: Path to the original CT/MR NIfTI image.
            fat_img: The muscle/fat segmentation image (already loaded).
            labels: Mapping of label numbers to names for ``fat_img``.
            layers: Layers to measure, e.g. ``["full_picture", "l3", "glut_to_c6"]``.

        Returns:
            ``{layer: result_dict}``. A missing 'total' segmentation maps all layers to the fallback.
            When arms are detected beside the body they are masked out (via a TotalSegmentator ``body``
            segmentation) before measuring, rather than skipped. A per-layer landmark miss (e.g. no L3)
            fills that layer and the remaining ones with the fallback, mirroring the original
            break-on-first-fallback behavior.
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
        total_vol = np.sum(np.asanyarray(base_img.dataobj) > -1000) * voxel_volume / 1000

        results: dict[str, dict] = {}
        for i, layer in enumerate(layers):
            fat_data = fat_full.copy()

            if layer == "l3":
                l3_coords = np.argwhere(seg_data == name_to_label["vertebrae_L3"])
                if l3_coords.size == 0:
                    logger.warning("vertebrae_L3 not found in segmentation; cannot define L3 layer")
                    results.update({rem: dict(fallback) for rem in layers[i:]})
                    break
                l_min, l_max = l3_coords[:, 0].min(), l3_coords[:, 0].max()
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
                fat_data[:g_min] = 0
                fat_data[c_max + 1 :] = 0

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

            result_dict["total_fat_in_%"] = total_fat / total_vol * 100
            result_dict["total_muscle_in_%"] = total_muscle / total_vol * 100
            result_dict["muscle_fat_ratio"] = total_muscle / total_fat if total_fat > 0 else None
            results[layer] = result_dict

        return results

    def remove_arms(
        self, tissue_img: nib.Nifti1Image, organ_img, labels, affine, z_axis, input_fpath: os.PathLike
    ) -> nib.Nifti1Image:
        """Zero out arm voxels in the tissue-type segmentation using a TotalSegmentator body mask.

        Runs the TotalSegmentator ``body`` task to obtain a trunk mask, extends it superiorly from the
        scapula and inferiorly from the hip, and removes voxels lateral to the trunk — so arms-down
        studies can still be quantified for LBM instead of being skipped.

        Args:
            tissue_img: Tissue-type segmentation image to be masked.
            organ_img: Integer label array from the 'total' segmentation.
            labels: 'total' segmentation mapping of label names to integers.
            affine: Affine matrix of the segmentation image.
            z_axis: Index of the head-to-toe axis (0, 1, or 2).
            input_fpath: Path to the original CT NIfTI; used to derive the CTbody.nii.gz output path.

        Returns:
            The tissue-type segmentation with arm voxels zeroed out (unmodified on body-seg failure).
        """
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
        description="Recursively run TotalSegmentator for muscle and fat (ml option) on all CT.nii.gz or "
        "MRI files in a folder. It processes each file, runs segmentation, extracts label mapping and "
        "computes body-composition stats and the lean body mass (LBM). Creates CT_muscle_fat.nii.gz and "
        "extends patient_info.json. Run the `sul` stage afterwards to produce SUL.nii.gz."
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
