import csv
import json
import logging
import os
import pathlib as plb
import random
import shutil
import string
import tempfile
from collections import defaultdict

import nibabel as nib
import numpy as np
import pydicom

from .utils import (
    agnostic_path,
    calculate_suv_factor,
    convert_pet,
    extract_dicom_data,
    find_mr_niftis,
    make_json_safe,
    mr_nifti_exists,
    repair_ct_affine_from_dicom,
    run_dcm2niix,
    setup_series_keywords,
)

logger = logging.getLogger(__name__)


class SeriesSelection:
    def __init__(
        self,
        input_dirpath: str | os.PathLike,
        output_dirpath: str | os.PathLike,
        series_keywords: dict[str, dict[str, list[str]]],
    ) -> None:
        """Class to select DICOM series for conversion to NIfTI format as well as to extract SUV images from PET series.

        Args:
            input_dirpath (str | os.PathLike): Directory containing the raw DICOM files. Can be nested.
            output_dirpath (str | os.PathLike): Directory containing the processed nifti files. Can be nested.
            series_keywords (dict[str, dict[str, list[str]]]): Optional. Keywords for automatic pre-selection of series.
        """
        self.output_dirpath = agnostic_path(os.path.abspath(output_dirpath))
        self.input_dirpath = agnostic_path(os.path.abspath(input_dirpath))
        self.series_keywords = series_keywords
        self.patient_results = {}
        # Cache of {patient_info.json path -> {raw series-dir basename -> recorded study key}}, used to
        # reuse the study key a previous run already assigned when StudyDate is an anonymized placeholder.
        self._recorded_key_cache: dict[str, dict[str, str]] = {}
        self.patient_tags = {
            "PatientName": ("0010", "0010"),
            "PatientID": ("0010", "0020"),
            "PatientBirthDate": ("0010", "0030"),
            "PatientSex": ("0010", "0040"),
        }
        self.series_tags = {
            "PatientAge": ("0010", "1010"),
            "PatientSize": ("0010", "1020"),  # in meters
            "PatientWeight": ("0010", "1030"),  # in kilograms
        }
        self.dicom_tags = {
            "Modality": ("0008", "0060"),
            "Manufacturer": ("0008", "0070"),
            "ManufacturersModelName": ("0008", "1090"),
            "DeviceSerialNumber": ("0018", "1000"),
            "SeriesInstanceUID": ("0020", "000e"),
            "StudyInstanceUID": ("0020", "000d"),
            "PatientName": ("0010", "0010"),
            "PatientID": ("0010", "0020"),
            "AccessionNumber": ("0008", "0050"),
            "PatientSex": ("0010", "0040"),
            "PatientWeight": ("0010", "1030"),
            "PatientAge": ("0010", "1010"),
            "BodyPart": ("0018", "0015"),
            "PatientPosition": ("0018", "5100"),
            "SoftwareVersions": ("0018", "1020"),
            "StudyDescription": ("0008", "1030"),
            "SeriesDescription": ("0008", "103e"),
            "ProtocolName": ("0018", "1030"),
            "ImageType": ("0008", "0008"),
            "SeriesNumber": ("0020", "0011"),
            "AcquisitionTime": ("0008", "0032"),
            "AcquisitionDateTime": ("0008", "002a"),
            "AcquisitionNumber": ("0020", "0012"),
            "ConvolutionKernel": ("0018", "1210"),
            "ExposureTime": ("0018", "1150"),
            "XRayTubeCurrent": ("0018", "1151"),
            "XRayExposure": ("0018", "1152"),
            "ImageOrientationPatientDICOM": ("0020", "0037"),
            "Radiopharmaceutical": ("0018", "0031"),
        }

    def run(self) -> None:
        self.grouped_series = self.collect_series()
        self.interactive_selection()

    def collect_series(self) -> defaultdict:
        """Collect DICOM series from the input directory and group them by patient and study date."""
        if not plb.Path(self.input_dirpath).exists():
            raise FileNotFoundError(f"Input directory {self.input_dirpath} does not exist.")

        sub_dirs = [plb.Path(x[0]) for x in os.walk(self.input_dirpath)]

        grouped = defaultdict(list)
        info = None
        for dir in sorted(sub_dirs):
            dicom_files = [
                f
                for f in dir.iterdir()
                if f.is_file()
                and f.name.lower() != "dicomdir"
                and f.suffix.lower()
                not in [
                    ".zip",
                    ".inf",
                    ".jar",
                    ".icns",
                    ".info",
                    ".exe",
                    ".pdf",
                    ".txt",
                    ".ini",
                    ".xml",
                    ".bmp",
                    ".sh",
                    ".json",
                ]
                and f.name != ".DS_Store"
                and f.name != "DeepUnity Media Viewer Mac"
            ]
            if not dicom_files:
                continue

            first_file = dicom_files[0]

            try:
                ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)

                patient_id = getattr(ds, "PatientID", None)
                study_date = getattr(ds, "StudyDate", None)
                modality = getattr(ds, "Modality", None)
                series_desc = getattr(ds, "SeriesDescription", "").lower()
                study_desc = getattr(ds, "StudyDescription", "N/A")
                manufacturer = getattr(ds, "Manufacturer", "Unknown")
                protocol_name = getattr(ds, "ProtocolName", None)

                if not (patient_id and study_date and modality):
                    continue

                if modality not in ("CT", "PT", "MR"):
                    continue

                out_path_patient_info = os.path.join(self.output_dirpath, patient_id, "patient_info.json")
                # Unique per-series study key; equals StudyDate unless it is an anonymized placeholder.
                study_key = self._study_key(dir, study_date, out_path_patient_info)

                out_path_CT = os.path.join(self.output_dirpath, patient_id, study_key, "CT.nii.gz")
                out_path_PT = os.path.join(self.output_dirpath, patient_id, study_key, "PET.nii.gz")
                out_path_SUV = os.path.join(self.output_dirpath, patient_id, study_key, "SUV.nii.gz")
                mr_study_dir = plb.Path(self.output_dirpath) / patient_id / study_key
                # dcm2niix names MR NIfTIs from `%p` (ProtocolName, falling back to
                # SeriesDescription when absent), so match on that to detect an already-converted series.
                mr_series_nii_exists = modality == "MR" and mr_nifti_exists(mr_study_dir, protocol_name, series_desc)
                if (
                    (
                        modality in ["CT", "PT"]
                        and all(
                            [os.path.isfile(out_path_CT), os.path.isfile(out_path_PT), os.path.isfile(out_path_SUV)]
                        )
                    )
                    or mr_series_nii_exists
                ) and os.path.isfile(out_path_patient_info):
                    new_info = f"Processed files for patient {patient_id} in study {study_date} already exist."
                    if new_info != info:
                        logger.info(new_info)
                        info = new_info
                    continue

                grouped[(patient_id, study_key)].append(
                    {
                        "PatientID": patient_id,
                        "StudyDate": study_key,
                        "Modality": modality,
                        "SeriesDescription": series_desc,
                        "StudyDescription": study_desc,
                        "SeriesPath": dir,
                        "StudyPath": dir.parent,
                        "PatientPath": plb.Path(*dir.parts[: dir.parts.index(patient_id) + 1]),
                        "Manufacturer": manufacturer,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to read DICOM: {first_file} — {e}")
                continue
        return grouped

    # DICOM "empty" date placeholders left behind by anonymizers. When StudyDate is one of these,
    # every series of a patient would collapse onto the same study folder and overwrite each other
    # (CT conversion writes a fixed CT.nii.gz), so we derive a unique key from the series dir name.
    _PLACEHOLDER_DATES = {"", "00000000", "00010101", "19000101"}

    def _study_key(self, series_dir: plb.Path, study_date: str | None, patient_info_path: str | os.PathLike) -> str:
        """Return a study key unique per series.

        Normally this is the DICOM StudyDate. For anonymized cohorts where StudyDate is a constant
        placeholder, resolve the key in this order:

        1. Reuse the study key a **previous run** already assigned to this series (matched by the raw
           series directory recorded in ``patient_info.json``). This keeps re-runs idempotent even
           when the study folders were later renamed to a recovered/real StudyDate that cannot be
           re-derived from the anonymized data — a series already processed maps back to its existing
           folder and is skipped rather than reprocessed into a differently-named one.
        2. Otherwise fall back to the 14-digit YYYYMMDDhhmmss datetime embedded as the last
           dot-separated token of the series directory name (e.g. ``...255564.20230406132841``),
           which is distinct per series.
        3. Falls back to StudyDate unchanged if no such token exists.
        """
        if study_date not in self._PLACEHOLDER_DATES:
            return study_date  # type: ignore[return-value]
        recorded = self._recorded_study_key(patient_info_path, series_dir)
        if recorded is not None:
            return recorded
        token = series_dir.name.rsplit(".", 1)[-1]
        if len(token) == 14 and token.isdigit():
            return token
        logger.warning(
            f"StudyDate is a placeholder ({study_date!r}) and no datetime token found in "
            f"'{series_dir.name}'; series may collide on the study folder."
        )
        return study_date  # type: ignore[return-value]

    def _recorded_study_key(self, patient_info_path: str | os.PathLike, series_dir: plb.Path) -> str | None:
        """Return the study key an existing patient_info.json already recorded for this series dir.

        Matches on the raw series directory basename against each recorded series' ``InputDirPath``,
        so a re-run reuses the exact (possibly renamed/recovered) study folder rather than
        recomputing a fresh one. Returns None when there is no prior record (e.g. a new patient, or a
        never-converted "dangling" study), so the caller falls through to the datetime-token key.
        """
        table = self._recorded_key_cache.get(str(patient_info_path))
        if table is None:
            table = {}
            if os.path.isfile(patient_info_path):
                try:
                    with open(patient_info_path) as f:
                        data = json.load(f)
                    for study_key, study in data.get("Studies", {}).items():
                        for series_list in study.get("Modalities", {}).values():
                            for series_dict in series_list:
                                for series_info in series_dict.values():
                                    input_dirpath = series_info.get("InputDirPath")
                                    if input_dirpath:
                                        table[os.path.basename(str(input_dirpath).rstrip("/"))] = study_key
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Could not read {patient_info_path} for study-key reuse: {e}")
            self._recorded_key_cache[str(patient_info_path)] = table
        return table.get(series_dir.name)

    def get_number_of_slices(self, series_path: os.PathLike):
        number_of_slices = 0
        for dicom_file in os.listdir(series_path):
            try:
                _ = pydicom.dcmread(os.path.join(series_path, dicom_file), stop_before_pixels=True)
                number_of_slices += 1
            except Exception as e:
                logger.debug(f"didn't count file: {dicom_file} ({e})")
        return number_of_slices

    def interactive_selection(self) -> None:
        """Interactive selection of DICOM series for conversion to NIfTI.
        This method allows the user to select series based on keywords and flags studies if no suitable series is found.
        Furthermore, the method handles the conversion of selected series to NIfTI format and validates the output.
        The patient_results dictionary is updated with the selected series and their metadata.
        """
        user_flags = {}
        patient_conversion_flags = {}
        try:
            user_wants_to_select = (
                input("Do you want to select manually? (y) yes manually, (N) No, use pre-selected indices: ")
                .strip()
                .lower()
            )
        except EOFError:
            logger.warning("No interactive terminal detected. Using pre-selected indices (n).")
            user_wants_to_select = "n"
        if user_wants_to_select not in ("y", "n"):
            logger.warning(
                f"You want: {user_wants_to_select}. Starting without interactive "
                "selection using (n) pre-selected indices."
            )
            user_wants_to_select = "n"

        previous_patient_id = None
        for idx, ((patient_id, study_date), study_info) in enumerate(sorted(self.grouped_series.items())):
            # Studies are sorted by (patient_id, study_date), so a patient's studies are contiguous.
            # As soon as we move to a new patient, flush the previous one's patient_info.json so an
            # interrupted run still leaves completed patients with valid, resumable metadata.
            if previous_patient_id is not None and patient_id != previous_patient_id:
                self._finalize_patient(previous_patient_id, user_flags, patient_conversion_flags)
            previous_patient_id = patient_id

            if not study_info:
                logger.warning(f"Skipping empty study: Patient ID: {patient_id} — Study Date: {study_date}")
                continue
            if patient_id not in user_flags:
                user_flags[patient_id] = []
            if patient_id not in patient_conversion_flags:
                patient_conversion_flags[patient_id] = []
            logger.info(
                f"📚 Study {idx + 1} of {len(self.grouped_series)} — Patient ID: {patient_id} — "
                f"Study Date: {study_date} - Study Desc: {study_info[0]['StudyDescription']}"
            )
            logger.info(f"Manufacturer: {study_info[0]['Manufacturer']}")
            logger.info("Available Series:")

            preselected_indices, fallback_flag = self.find_default_indices(study_info)

            for i, s in enumerate(study_info):
                pre = i in preselected_indices
                mark = "[*]" if pre else "[ ]"
                if "NumSlices" in s:
                    logger.info(
                        f"{mark} [{i:2}] {s['Modality']:>3} | slices: {s['NumSlices']} | {s['SeriesDescription']}"
                    )
                else:
                    logger.info(f"{mark} [{i:2}] {s['Modality']:>3} | {s['SeriesDescription']}")

            if fallback_flag:
                logger.info(
                    f"⚠️ No {self.series_keywords[study_info[0]['Modality']]['PRIMARY']} found — "
                    f"defaulted to {self.series_keywords[study_info[0]['Modality']]['SECONDARY']} "
                    f"and flagged study."
                )

            default_input = ",".join(str(i) for i in preselected_indices)
            if user_wants_to_select in ["n"]:
                logger.info(
                    f"Skipping interactive selection for Patient ID: {patient_id} - "
                    f"Study Date: {study_date}. Using preselected indices: {default_input}"
                )
                user_input = default_input
            else:
                user_input = (
                    input(
                        f"Enter numbers to select (comma-separated), add 'x' to flag study [default: {default_input}]: "
                    )
                    .strip()
                    .lower()
                )

            # Parse selection
            user_flag = bool("x" in user_input)
            user_flags[patient_id].append(user_flag)
            input_parts = [part.strip() for part in user_input.split(",") if part.strip().isdigit()]
            if not user_input or user_flag:
                indices = preselected_indices
            else:
                indices = [int(i) for i in input_parts if i.isdigit() and 0 <= int(i) < len(study_info)]

            if not indices:
                logger.warning(f"No series selected for patient: {patient_id}, study: {study_date}")
                continue

            selected_series = {patient_id: [study_info[i] for i in indices]}
            for s in selected_series[patient_id]:
                siblings = self._dynamic_sibling_dirs(study_info, s)
                if siblings:
                    s["DynamicSiblingPaths"] = siblings
            if patient_id not in self.patient_results:
                self.patient_results[patient_id] = {
                    "InputDirPath": str(selected_series[patient_id][0]["PatientPath"]),
                    **extract_dicom_data(plb.Path(selected_series[patient_id][0]["SeriesPath"]), self.patient_tags),
                    "Studies": {},
                }

            series_conversion_flags = self.handle_selected_series(selected_series)
            if series_conversion_flags:
                for flag in series_conversion_flags:
                    patient_conversion_flags[patient_id].append(flag)

        # Flush the final patient (the boundary-triggered flush above only fires on patient change).
        if previous_patient_id is not None:
            self._finalize_patient(previous_patient_id, user_flags, patient_conversion_flags)

    def _finalize_patient(self, patient_id: str, user_flags: dict, patient_conversion_flags: dict) -> None:
        """Validate and write one patient's patient_info.json as soon as its studies are all processed.

        Writing per-patient (instead of once at the very end) means an interrupted run still leaves
        completed patients with valid metadata, and a re-run resumes by skipping them. Merges into an
        existing file via _merge_studies so re-runs accumulate rather than clobber.
        """
        if patient_id not in self.patient_results:
            return

        self.validate_output(
            data_dict=self.patient_results[patient_id],
            output_csv_path=os.path.join(self.output_dirpath, "validation_results.csv"),
            user_flag=bool(any(user_flags.get(patient_id, []))),
            conversion_flags=patient_conversion_flags.get(patient_id, []),
        )

        json_path = os.path.join(self.output_dirpath, patient_id, "patient_info.json")
        if os.path.isfile(json_path):
            with open(json_path) as existing_f:
                existing_info = json.load(existing_f)
            self.patient_results[patient_id]["Studies"] = self._merge_studies(
                existing_info.get("Studies", {}),
                self.patient_results[patient_id].get("Studies", {}),
            )

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            self.patient_results[patient_id] = make_json_safe(self.patient_results[patient_id])
            json.dump(self.patient_results[patient_id], f)

    def find_default_indices(self, series_list: list) -> tuple[list, bool]:
        """Find default indices based on series keywords.
        This method checks the series descriptions against predefined keywords for primary and secondary selection.
        It also checks if there are other series descriptions with the same naming and safes the number of slices.
        It returns a list of indices for the selected series and a flag indicating if secondary keywords were used.
        """
        has_none = any(v is None for inner in self.series_keywords.values() for v in inner.values())
        empty_dict = all(
            hasattr(v, "__len__") and len(v) == 0 for inner in self.series_keywords.values() for v in inner.values()
        )
        if has_none or empty_dict:
            logger.info("No Keywords given, using all series")
            preselected_indices = list(range(len(series_list)))
            return preselected_indices, False

        desc_groups = defaultdict(list)
        for idx, s in enumerate(series_list):
            desc_groups[s["SeriesDescription"]].append(idx)

        for _desc, entries in desc_groups.items():
            if len(entries) > 1:
                for idx in entries:
                    if "NumSlices" not in series_list[idx]:
                        series_list[idx]["NumSlices"] = self.get_number_of_slices(series_list[idx]["SeriesPath"])

        preselected_indices = []
        secondary_used = False
        modality_matches = {}

        for i, s in enumerate(series_list):
            modality = s["Modality"]
            desc = s["SeriesDescription"].lower()

            if modality not in self.series_keywords:
                continue

            keywords = self.series_keywords[modality]
            primary_keywords = keywords.get("PRIMARY", [])
            secondary_keywords = keywords.get("SECONDARY", [])
            exclusion_keywords = keywords.get("EXCLUSION", [])

            if any(excl in desc for excl in exclusion_keywords) or desc is None or desc == "":
                continue

            if modality not in modality_matches:
                modality_matches[modality] = {"primary": [], "secondary": []}

            if any(pk in desc for pk in primary_keywords):
                modality_matches[modality]["primary"].append(i)
            elif any(sk in desc for sk in secondary_keywords):
                modality_matches[modality]["secondary"].append(i)

        for match_type in ["primary", "secondary"]:
            for _modality, match in modality_matches.items():
                indices = match[match_type]
                if not indices:
                    continue

                desc_group_indices = defaultdict(list)
                for idx in indices:
                    desc_group_indices[series_list[idx]["SeriesDescription"]].append(idx)

                for entries in desc_group_indices.values():
                    if len(entries) == 1:
                        preselected_indices.append(entries[0])
                    else:
                        best_idx = max(entries, key=lambda x: series_list[x]["NumSlices"])
                        preselected_indices.append(best_idx)
                        if match_type == "secondary":
                            secondary_used = True

        should_flag = not preselected_indices or secondary_used
        return preselected_indices, should_flag

    def handle_selected_series(self, selected_series: dict) -> list:
        """Handle the selected series for conversion to NIfTI format and extract SUV images from PET series.
        This method processes the selected series, performs necessary conversions and collects potential errors
        during conversion for later flagging, and updates the patient results.
        """
        logger.info("✅ Selected Series:")
        patient_id = list(selected_series.keys())[0]
        flags = []

        for i, series in enumerate(selected_series[patient_id]):
            study_date = series["StudyDate"]
            modality = series["Modality"]
            series_desc = series["SeriesDescription"]
            series_path = series["SeriesPath"]
            study_path = series["StudyPath"]

            logger.info(
                f"Processing Patient ID: {patient_id}, Date: {study_date}, "
                f"Modality: {modality}, Description: {series_desc}"
            )
            if i == 0:
                self.patient_results[patient_id]["Studies"].update(
                    {
                        study_date: {
                            "InputDirPath": str(study_path),
                            "StudyDescription": series["StudyDescription"],
                            **extract_dicom_data(series_path, self.series_tags),
                            "Modalities": {},
                        }
                    }
                )
            if modality not in self.patient_results[patient_id]["Studies"][study_date]["Modalities"]:
                self.patient_results[patient_id]["Studies"][study_date]["Modalities"].update({modality: []})

            flag, paths_and_dicom_tags = self.start_dcm2nii(
                modality=modality,
                dicom_input_dirpath=series_path,
                out_dirpath=os.path.join(self.output_dirpath, patient_id, study_date),
                dynamic_sibling_dirs=series.get("DynamicSiblingPaths"),
            )
            if paths_and_dicom_tags:
                self.patient_results[patient_id]["Studies"][study_date]["Modalities"][modality].append(
                    paths_and_dicom_tags
                )

            if flag:
                flags.append([patient_id, study_date, series_desc, series_path])
        logger.info("-" * 90)
        return flags

    def start_dcm2nii(self, modality, dicom_input_dirpath, out_dirpath, dynamic_sibling_dirs=None) -> tuple[bool, dict]:
        """Start the DICOM to NIfTI conversion process for the specified modality.

        Args:
            modality (str): The imaging modality (CT, PT, MR).
            dicom_input_dirpath (str | os.PathLike): The input directory containing DICOM files.
            out_dirpath (str | os.PathLike): The output directory for NIfTI files.

        Returns:
            tuple[bool, dict]: A tuple containing a flag indicating success or failure,
            and a dictionary with paths and DICOM tags.
        """
        paths_and_dicom_tags = {}
        os.makedirs(out_dirpath, exist_ok=True)
        suv_fpath = None
        try:
            if modality == "CT":
                out_fpath = os.path.join(out_dirpath, "CT.nii.gz")
                dicom_tags = self.convert_dcm2nii_CT(CT_dcm_dirpath=dicom_input_dirpath, output_dirpath=out_dirpath)
            elif modality == "PT":
                out_fpath = os.path.join(out_dirpath, "PET.nii.gz")
                suv_fpath = os.path.join(out_dirpath, "SUV.nii.gz")
                dicom_tags = self.convert_dcm2nii_PET(PET_dcm_dirpath=dicom_input_dirpath, output_dirpath=out_dirpath)
            elif modality == "MR":
                out_fpath, dicom_tags = self.convert_dcm2nii_MR(
                    MR_dcm_dirpath=dicom_input_dirpath,
                    output_dirpath=out_dirpath,
                    dynamic_sibling_dirs=dynamic_sibling_dirs,
                )
            if "SeriesDescription" not in dicom_tags:
                chars = string.ascii_letters + string.digits
                random_part = "".join(random.choices(chars, k=5))
                dicom_tags["SeriesDescription"] = f"Missing_SeriesDesc_{random_part}"

            paths_and_dicom_tags = {
                dicom_tags["SeriesDescription"]: {
                    "InputDirPath": str(dicom_input_dirpath),
                    f"{modality.upper()}Path": out_fpath,
                    **({"SUVPath": suv_fpath} if suv_fpath is not None else {}),
                    "DICOM": dicom_tags,
                }
            }
            return False, paths_and_dicom_tags
        except Exception as e:
            logger.error(f"Error processing {modality} series: {e}")
            return True, {}

    def _select_ct_volume(self, tmp: plb.Path, ct_dcm_dirpath: str | os.PathLike) -> plb.Path:
        """Pick which CT volume to keep from dcm2niix's output.

        dcm2niix may emit several NIfTIs for one input directory: a gantry-tilt-corrected ``*_Eq_1``
        version, or — when the directory bundles multiple reconstructions (e.g. two convolution
        kernels, or an ORIGINAL/PRIMARY plus an ORIGINAL/SECONDARY series, as in the anonymized
        whole-body cohorts) — one NIfTI per reconstruction. Preference order:

        1. the gantry-tilt-corrected ``*_Eq_1`` output, if present;
        2. ORIGINAL/PRIMARY over SECONDARY (read from each volume's JSON sidecar ``ImageType``);
        3. the volume with the most slices (largest anatomical coverage);
        4. largest file, then name — purely for determinism.

        Never silently drops the rest: discarded volumes are logged. Raises if dcm2niix produced no
        NIfTI (so the caller flags the series instead of crashing on an undefined variable).
        """
        nii_files = sorted(tmp.glob("*.nii.gz"))
        if not nii_files:
            raise ValueError(f"CT conversion produced no NIfTI files for {ct_dcm_dirpath}")
        eq = [f for f in nii_files if f.name.endswith("_Eq_1.nii.gz")]
        if eq:
            return eq[0]
        if len(nii_files) == 1:
            return nii_files[0]

        def _rank(f: plb.Path):
            image_type = []
            sidecar = f.with_suffix("").with_suffix(".json")
            if sidecar.is_file():
                with open(sidecar) as jf:
                    image_type = [str(x).upper() for x in json.load(jf).get("ImageType", [])]
            primary = "PRIMARY" in image_type and "SECONDARY" not in image_type
            shape = nib.load(str(f)).shape
            n_slices = shape[2] if len(shape) >= 3 else 0
            return (primary, len(shape) < 4, n_slices, f.stat().st_size)

        nii = max(nii_files, key=_rank)
        discarded = sorted(f.name for f in nii_files if f != nii)
        logger.warning(
            f"dcm2niix produced {len(nii_files)} CT volumes for {ct_dcm_dirpath}; "
            f"kept {nii.name} (shape {nib.load(str(nii)).shape}), discarded: {discarded}"
        )
        return nii

    def convert_dcm2nii_CT(self, CT_dcm_dirpath: str | os.PathLike, output_dirpath: str | os.PathLike) -> dict:
        """Conversion of CT DICOM (in the CT_dcm_path) to nifti and save in output_dirpath

        Args:
            CT_dcm_dirpath (str | os.PathLike): Directory containing the CT DICOM files.
            output_dirpath (str | os.PathLike): Directory to save the converted NIfTI files.

        Returns:
            dict: A dictionary containing the DICOM tags extracted from the converted NIfTI files.
        """
        out_fpath = os.path.join(output_dirpath, "CT.nii.gz")
        if not os.path.isfile(out_fpath):
            with tempfile.TemporaryDirectory() as tmp:  # convert CT
                tmp = plb.Path(str(tmp))
                # convert dicom directory to nifti (store results in temp directory)
                run_dcm2niix(CT_dcm_dirpath, plb.Path(tmp))
                nii = self._select_ct_volume(tmp, CT_dcm_dirpath)

                # copy chosen nifti to output folder with consistent naming
                shutil.copy(nii, out_fpath)
                # dcm2niix mis-derives the slice axis for series missing SpacingBetweenSlices
                # (e.g. Siemens NAEOTOM Alpha VMI), producing an upside-down/stretched volume;
                # repair the affine from the DICOM positions when it disagrees.
                try:
                    repair_ct_affine_from_dicom(out_fpath, CT_dcm_dirpath)
                except Exception as e:
                    logger.error(f"CT affine sanity-check failed for {out_fpath}: {e}")
                # read the sidecar matching the chosen volume (same stem)
                jsn = nii.with_suffix("").with_suffix(".json")
                if not jsn.is_file():
                    jsn = next(tmp.glob("*json"))
                with open(jsn) as json_file:
                    dicom_tags = json.load(json_file)
        else:
            logger.info(f"CT NIfTI already exists at {out_fpath}")
            dicom_tags = extract_dicom_data(plb.Path(CT_dcm_dirpath), self.dicom_tags)
        return dicom_tags

    def convert_dcm2nii_PET(self, PET_dcm_dirpath: str | os.PathLike, output_dirpath: str | os.PathLike) -> dict:
        """Conversion of PET DICOM (in the PET_dcm_path) to nifti (and SUV nifti) and save in output_dirpath.

        Args:
            PET_dcm_dirpath (str | os.PathLike): Directory containing the PET DICOM files.
            output_dirpath (str | os.PathLike): Directory to save the converted NIfTI files.

        Returns:
            dict: A dictionary containing the DICOM tags extracted from the converted NIfTI files.
        """
        out_pet_fpath = os.path.join(output_dirpath, "PET.nii.gz")
        out_suv_fpath = os.path.join(output_dirpath, "SUV.nii.gz")
        first_pt_dcm = os.listdir(PET_dcm_dirpath)[0]
        ds = pydicom.dcmread(os.path.join(PET_dcm_dirpath, first_pt_dcm))
        if os.path.isfile(out_pet_fpath) and os.path.isfile(out_suv_fpath):
            logger.info(f"PET NIfTI and SUV NIfTI already exist at {out_pet_fpath} and {out_suv_fpath}")
            dicom_tags = extract_dicom_data(plb.Path(PET_dcm_dirpath), self.dicom_tags)
            seq = ds.RadiopharmaceuticalInformationSequence[0]
            dicom_tags["RadiopharmaceuticalStartTime"] = seq.RadiopharmaceuticalStartTime
            dicom_tags["InjectedRadioactivity"] = seq.RadionuclideTotalDose
            dicom_tags["RadionuclideHalfLife"] = seq.RadionuclideHalfLife
            return dicom_tags
        else:
            total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
            start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
            half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
            acq_time = ds.AcquisitionTime
            weight = ds.PatientWeight
            suv_corr_factor = calculate_suv_factor(total_dose, start_time, half_life, acq_time, weight)

            with tempfile.TemporaryDirectory() as tmp:  # convert PET
                tmp = plb.Path(str(tmp))
                # convert dicom directory to nifti
                # (store results in temp directory)
                run_dcm2niix(PET_dcm_dirpath, plb.Path(tmp))
                nii = next(tmp.glob("*nii.gz"))
                # copy nifti to output folder with consistent naming
                out_pet_fpath = os.path.join(output_dirpath, "PET.nii.gz")
                shutil.copy(nii, out_pet_fpath)
                nii = next(tmp.glob("*json"))
                with open(nii) as json_file:
                    dicom_tags = json.load(json_file)

                dicom_tags["RadiopharmaceuticalStartTime"] = start_time
                dicom_tags["SUVFactor"] = suv_corr_factor

                # convert pet images to quantitative suv images and save nifti file
                out_suv_fpath = os.path.join(output_dirpath, "SUV.nii.gz")
                suv_pet_nii = convert_pet(
                    nib.load(os.path.join(output_dirpath, "PET.nii.gz")),
                    suv_factor=suv_corr_factor,  # type: ignore
                )
                nib.save(img=suv_pet_nii, filename=out_suv_fpath)  # type: ignore
            return dicom_tags

    def _dynamic_sibling_dirs(self, study_info: list, entry: dict) -> list | None:
        """Detect a dynamic acquisition stored as separate DICOM series (one per timepoint).

        Some scanners write each timepoint of a dynamic/DCE series as its own series (same
        SeriesDescription, distinct SeriesNumber/AcquisitionTime) rather than one multi-frame
        series. dcm2niix won't merge across series, so each would convert to a 3D volume.

        Returns the sibling series dirs ordered by AcquisitionTime when ALL hold (strict, to
        avoid wrongly merging genuinely separate acquisitions): same SeriesDescription, >= 3
        series, >= 3 distinct AcquisitionTimes, and a dynamic marker (``dyn``/``dce`` in the
        description or ``DYNAMIC`` in ImageType). Otherwise None (treat as a normal series).
        Geometry consistency (shape/affine) is verified later, at stack time.
        """
        if entry["Modality"] != "MR":
            return None
        desc = entry["SeriesDescription"]
        group = [s for s in study_info if s["Modality"] == "MR" and s["SeriesDescription"] == desc]
        if len(group) < 3:
            return None

        dynamic_marker = "dyn" in desc.lower() or "dce" in desc.lower()
        timed = []
        for s in group:
            files = [f for f in os.listdir(s["SeriesPath"]) if f.lower() != "dicomdir" and not f.startswith(".")]
            if not files:
                return None
            try:
                ds = pydicom.dcmread(os.path.join(s["SeriesPath"], files[0]), stop_before_pixels=True)
            except Exception:
                return None
            if "DYNAMIC" in [str(x).upper() for x in getattr(ds, "ImageType", [])]:
                dynamic_marker = True
            timed.append((getattr(ds, "AcquisitionTime", None), s["SeriesPath"]))

        times = [t for t, _ in timed]
        if not dynamic_marker or any(t is None for t in times) or len(set(times)) < 3:
            return None
        timed.sort(key=lambda x: x[0])
        return [p for _, p in timed]

    def _convert_dynamic_mr(
        self, sibling_dirs: list, output_dirpath: str | os.PathLike, fallback_dir: str | os.PathLike
    ) -> tuple[str | os.PathLike, dict]:
        """Convert each timepoint-series and stack them into one 4D NIfTI (ordered as given).

        Aborts to a normal single-series conversion (of ``fallback_dir``) if any timepoint
        fails to convert, is not 3D, or has geometry inconsistent with the first.
        """
        vols, affine, header, base_name, first_shape = [], None, None, None, None
        with tempfile.TemporaryDirectory() as tmproot:
            for i, d in enumerate(sibling_dirs):
                sub = plb.Path(tmproot) / str(i)
                sub.mkdir()
                run_dcm2niix(d, sub, merge=True)
                files = list(sub.glob("*.nii.gz"))
                if not files:
                    logger.warning(f"dynamic MR: no NIfTI for timepoint {d}; falling back to single volume.")
                    return self.convert_dcm2nii_MR(fallback_dir, output_dirpath)
                f = max(files, key=lambda x: (nib.load(str(x)).ndim, x.stat().st_size))
                img = nib.load(str(f))
                if img.ndim != 3:
                    logger.warning(f"dynamic MR: timepoint {f.name} is {img.shape}, not 3D; falling back.")
                    return self.convert_dcm2nii_MR(fallback_dir, output_dirpath)
                if affine is None:
                    affine, header, base_name, first_shape = img.affine, img.header, f.name, img.shape
                elif img.shape != first_shape or not np.allclose(img.affine, affine, atol=1e-3):
                    logger.warning(f"dynamic MR: geometry mismatch at {f.name}; falling back to single volume.")
                    return self.convert_dcm2nii_MR(fallback_dir, output_dirpath)
                vols.append(np.asanyarray(img.dataobj))

        data4d = np.stack(vols, axis=3)
        nii_path = os.path.join(output_dirpath, base_name)
        nib.save(nib.Nifti1Image(data4d, affine, header), nii_path)
        dicom_tags = extract_dicom_data(plb.Path(sibling_dirs[0]), self.dicom_tags)
        logger.info(f"dynamic MR: stacked {len(vols)} timepoints -> {os.path.basename(nii_path)} {data4d.shape}")
        return nii_path, dicom_tags

    def convert_dcm2nii_MR(
        self,
        MR_dcm_dirpath: str | os.PathLike,
        output_dirpath: str | os.PathLike,
        dynamic_sibling_dirs: list | None = None,
    ) -> tuple[str | os.PathLike, dict]:
        """Conversion of MR DICOM (in the MR_dcm_path) to nifti and save in output_dirpath.
        Args:
            MR_dcm_dirpath (str | os.PathLike): Directory containing the MR DICOM files.
            output_dirpath (str | os.PathLike): Directory to save the converted NIfTI files.

        Returns:
            tuple: Path to the NIfTI file and a dictionary of DICOM tags."""
        first_dcm = os.listdir(MR_dcm_dirpath)[0]
        ds = pydicom.dcmread(str(str(MR_dcm_dirpath) + "/" + first_dcm), stop_before_pixels=True)
        # dcm2niix names MR NIfTIs from `%p` (ProtocolName, falling back to SeriesDescription);
        # match on that to detect existing output.
        existing_niftis = find_mr_niftis(
            plb.Path(output_dirpath), getattr(ds, "ProtocolName", None), getattr(ds, "SeriesDescription", None)
        )
        if existing_niftis:
            logger.info(f"MRI NIfTI already exist at {output_dirpath}.")
            dicom_tags = extract_dicom_data(plb.Path(MR_dcm_dirpath), self.dicom_tags)
            return str(existing_niftis[0]), dicom_tags

        # Dynamic stored as separate per-timepoint series: convert all and stack into 4D.
        if dynamic_sibling_dirs and len(dynamic_sibling_dirs) > 1:
            return self._convert_dynamic_mr(dynamic_sibling_dirs, output_dirpath, MR_dcm_dirpath)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = plb.Path(tmp)

            # merge=True so a dynamic/DCE series split by dcm2niix into one file per timepoint
            # is reassembled into a single 4D NIfTI instead of collapsing to a 3D fragment.
            run_dcm2niix(MR_dcm_dirpath, tmp, merge=True)

            nii_files = list(tmp.glob("*.nii.gz"))

            if len(nii_files) == 0:
                logger.warning("No NIfTI files found. MRI conversion may have failed.")
                return "", {}

            # dcm2niix may still emit several files (e.g. DIXON water/fat/in/opp contrasts).
            # Pick deterministically — the volume with the most dimensions, then the most
            # timepoints, then the largest — rather than an arbitrary glob order, and never
            # silently drop the rest.
            def _rank(f: plb.Path):
                shape = nib.load(str(f)).shape
                ndim = len(shape)
                n_time = shape[3] if ndim >= 4 else 1
                return (ndim, n_time, f.stat().st_size)

            nii = max(nii_files, key=_rank)
            if len(nii_files) > 1:
                discarded = sorted(f.name for f in nii_files if f != nii)
                logger.warning(
                    f"dcm2niix produced {len(nii_files)} NIfTIs for MR series in {MR_dcm_dirpath}; "
                    f"kept {nii.name} (shape {nib.load(str(nii)).shape}), discarded: {discarded}"
                )

            # copy chosen nifti out and read its matching sidecar (same stem)
            nii_path = os.path.join(output_dirpath, nii.name)
            shutil.copy(nii, nii_path)
            jsn = nii.with_suffix("").with_suffix(".json")
            if not jsn.is_file():
                jsn = next(tmp.glob("*.json"))
            with open(jsn) as json_file:
                dicom_tags = json.load(json_file)
        return nii_path, dicom_tags

    @staticmethod
    def _series_identity(series_dict: dict) -> str:
        """Stable identity of a series entry, used to deduplicate across runs.

        A description-less series is keyed by a fresh random ``Missing_SeriesDesc_*`` name on every
        run, so deduplicating on that key lets re-runs accumulate a duplicate entry per run for the
        same physical series (e.g. CT-only studies, which the "already processed" guard never skips
        because it also requires PET/SUV). The raw ``InputDirPath`` (the series' SeriesInstanceUID
        directory) is stable across runs, so key on its basename; fall back to the description key
        only when InputDirPath is absent.
        """
        for key, info in series_dict.items():
            if isinstance(info, dict) and info.get("InputDirPath"):
                return os.path.basename(str(info["InputDirPath"]).rstrip("/"))
            return key
        return ""

    def _merge_studies(self, existing: dict, new: dict) -> dict:
        """Deep-merge new studies into existing ones without overwriting already-recorded series.

        Merges at three levels: study date → modality → series list (deduplicated by stable series
        identity, i.e. the InputDirPath basename — see ``_series_identity``). An already-recorded
        series is kept as-is (preserving detail added by later stages); only genuinely new series
        are appended.
        """
        merged = dict(existing)
        for study_date, new_study in new.items():
            if study_date not in merged:
                merged[study_date] = new_study
                continue
            existing_study = dict(merged[study_date])
            existing_modalities = existing_study.get("Modalities", {})
            for modality, new_series_list in new_study.get("Modalities", {}).items():
                if modality not in existing_modalities:
                    existing_modalities[modality] = new_series_list
                else:
                    existing_ids = {self._series_identity(sd) for sd in existing_modalities[modality]}
                    for series_dict in new_series_list:
                        ident = self._series_identity(series_dict)
                        if ident not in existing_ids:
                            existing_modalities[modality].append(series_dict)
                            existing_ids.add(ident)
            existing_study["Modalities"] = existing_modalities
            merged[study_date] = existing_study
        return merged

    def validate_output(self, data_dict, output_csv_path, user_flag: bool, conversion_flags: list) -> None:
        """Validate the output of the series selection and conversion process.
        This method checks for the existence of NIfTI files, verifies primary and secondary keywords,
        and flags any issues found during the validation process.

        Args:
            data_dict (dict): The data dictionary containing patient and study information.
            output_csv_path (str | os.PathLike): Path to the CSV file where validation results will be saved.
            user_flag (bool): Flag indicating if the study was flagged by the user.
            conversion_flags (list): List of flags indicating errors during the conversion process."""
        patient_id = data_dict.get("PatientID", "UNKNOWN")
        studies = data_dict.get("Studies", {})
        errors = []

        for study_date, study in studies.items():
            modalities = study.get("Modalities", {})

            for modality, series_list in modalities.items():
                keywords = self.series_keywords.get(modality.upper(), {})
                primary_keywords = keywords.get("PRIMARY", [])
                secondary_keywords = keywords.get("SECONDARY", [])
                has_primary = False
                has_secondary = False

                for series_dict in series_list:
                    if not isinstance(series_dict, dict) or not series_dict:
                        continue

                    for series_desc, series_info in series_dict.items():
                        desc_lower = series_desc.lower()

                        # 1. NIfTI file existence
                        nii_path = series_info.get(f"{modality.upper()}Path")
                        if nii_path and not os.path.exists(nii_path):
                            errors.append(
                                {
                                    "patient_id": patient_id,
                                    "StudyDate": study_date,
                                    "Modality": modality,
                                    "SeriesDesc": series_desc,
                                    "Reason": "NIfTI file does not exist",
                                }
                            )

                        # 2. PRIMARY / SECONDARY keyword check
                        if any(pk in desc_lower for pk in primary_keywords):
                            has_primary = True
                        elif any(sk in desc_lower for sk in secondary_keywords):
                            has_secondary = True

                        # 3. User flag
                        if user_flag:
                            errors.append(
                                {
                                    "patient_id": patient_id,
                                    "StudyDate": study_date,
                                    "Modality": modality,
                                    "SeriesDesc": series_desc,
                                    "Reason": "Study flagged by user",
                                }
                            )

                        # 4. Missing primary series
                        if not has_primary and primary_keywords:
                            reason = "No primary keyword match for modality in any series"
                            if has_secondary:
                                reason += " (but secondary keyword matched)"
                            errors.append(
                                {
                                    "patient_id": patient_id,
                                    "StudyDate": study_date,
                                    "Modality": modality,
                                    "SeriesDesc": f"(no matching series in {modality})",
                                    "Reason": reason,
                                }
                            )

        # 5. Conversion flag errors
        for flag_entry in conversion_flags:
            if len(flag_entry) >= 3:
                pid, s_date, s_desc = flag_entry[:3]
                errors.append(
                    {
                        "patient_id": pid,
                        "StudyDate": s_date,
                        "Modality": "",
                        "SeriesDesc": s_desc,
                        "Reason": "Error while converting the DICOM series to NIfTI",
                    }
                )

        if errors:
            file_exists = os.path.isfile(output_csv_path)

            with open(output_csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["patient_id", "StudyDate", "Modality", "SeriesDesc", "Reason"])
                if not file_exists:
                    writer.writeheader()
                writer.writerows(errors)


def series_selection_entrypoint():
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.series_selection")

    import argparse

    parser = argparse.ArgumentParser(description="Selection of DICOM series for conversion to NIfTI format.")
    parser.add_argument("--input-dir", help="Path to PET/CT input directory.", required=True)
    parser.add_argument("--output-dir", help="Path to designated output directory.", required=True)
    # nargs="*" so a flag passed without values yields an empty list (distinct from absent=None).
    # Passing any keyword flag empty disables keyword filtering and selects every series — useful
    # for anonymized cohorts whose Series/Study Description tags are empty.
    parser.add_argument(
        "--ct-primary-keywords",
        nargs="*",
        help="Keywords to look for in CT series descriptions for default selection. Pass empty to select all series.",
    )
    parser.add_argument(
        "--ct-secondary-keywords",
        nargs="*",
        help="Keywords to look for in CT series descriptions for alternative selection.",
    )
    parser.add_argument("--ct-exclusion-keywords", nargs="*", help="Keywords to exclude CT series from selection.")
    parser.add_argument(
        "--pt-primary-keywords",
        nargs="*",
        help="Keywords to look for in PT series descriptions for default selection.",
    )
    parser.add_argument(
        "--pt-secondary-keywords",
        nargs="*",
        help="Keywords to look for in PT series descriptions for alternative selection.",
    )
    parser.add_argument("--pt-exclusion-keywords", nargs="*", help="Keywords to exclude PT series from selection.")
    parser.add_argument(
        "--mr-primary-keywords",
        nargs="*",
        help="Keywords to look for in MR series descriptions for default selection.",
    )
    parser.add_argument(
        "--mr-secondary-keywords",
        nargs="*",
        help="Keywords to look for in MR series descriptions for alternative selection.",
    )
    parser.add_argument("--mr-exclusion-keywords", nargs="*", help="Keywords to exclude MR series from selection.")
    args = parser.parse_args()

    series_keywords = setup_series_keywords(
        ct_primary_keywords=args.ct_primary_keywords,
        ct_secondary_keywords=args.ct_secondary_keywords,
        ct_exclusion_keywords=args.ct_exclusion_keywords,
        pt_primary_keywords=args.pt_primary_keywords,
        pt_secondary_keywords=args.pt_secondary_keywords,
        pt_exclusion_keywords=args.pt_exclusion_keywords,
        mr_primary_keywords=args.mr_primary_keywords,
        mr_secondary_keywords=args.mr_secondary_keywords,
        mr_exclusion_keywords=args.mr_exclusion_keywords,
    )

    SeriesSelection(input_dirpath=args.input_dir, output_dirpath=args.output_dir, series_keywords=series_keywords).run()


if __name__ == "__main__":
    series_selection_entrypoint()
