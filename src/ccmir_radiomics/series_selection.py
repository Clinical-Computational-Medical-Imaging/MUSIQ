import csv
import json
import logging
import os
import pathlib as plb
import shutil
import tempfile
from collections import defaultdict

import nibabel as nib
import numpy as np
import pydicom

from .utils import agnostic_path, conv_time, extract_dicom_data, make_json_safe, run_dcm2niix, setup_series_keywords

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
        self.patient_tags = {
            "PatientName": ("0010", "0010"),
            "PatientID": ("0010", "0020"),
            "PatientBirthDate": ("0010", "0030"),
            "PatientSex": ("0010", "0040"),
        }
        self.serie_tags = {
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
                not in [".zip", ".inf", ".jar", ".icns", ".info", ".exe", ".pdf", ".txt", ".ini", ".xml", ".bmp", ".sh"]
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

                if not (patient_id and study_date and modality):
                    continue

                if modality not in ("CT", "PT", "MR"):
                    continue

                out_path_patient_info = os.path.join(self.output_dirpath, patient_id, "patient_info.json")
                out_path_CT = os.path.join(self.output_dirpath, patient_id, study_date, "CT.nii.gz")
                out_path_PT = os.path.join(self.output_dirpath, patient_id, study_date, "PET.nii.gz")
                out_path_MR = os.path.join(self.output_dirpath, patient_id, study_date, ".nii.gz")
                if (
                    (modality in ["CT", "PT"] and all([os.path.isfile(out_path_CT), os.path.isfile(out_path_PT)]))
                    or (modality == "MR" and any(out_path_MR))
                ) and os.path.isfile(out_path_patient_info):
                    new_info = f"Processed files for patient {patient_id} in study {study_date} already exist."
                    if new_info != info:
                        logger.info(new_info)
                        info = new_info
                    continue

                grouped[(patient_id, study_date)].append(
                    {
                        "PatientID": patient_id,
                        "StudyDate": study_date,
                        "Modality": modality,
                        "SeriesDescription": series_desc,
                        "StudyDescription": study_desc,
                        "SeriePath": dir,
                        "StudyPath": dir.parent,
                        "PatientPath": plb.Path(*dir.parts[: dir.parts.index(patient_id) + 1]),
                    }
                )

            except Exception as e:
                logger.error(f"Failed to read DICOM: {first_file} — {e}")
                continue
        return grouped

    def interactive_selection(self) -> None:
        """Interactive selection of DICOM series for conversion to NIfTI.
        This method allows the user to select series based on keywords and flags studies if no suitable series is found.
        Furthermore, the method handles the conversion of selected series to NIfTI format and validates the output.
        The patient_results dictionary is updated with the selected series and their metadata.
        """
        user_flags = {}
        patient_conversion_flags = {}
        user_wants_to_select = input(
            "Do you want to select series interactively and flag studies? (y/n): "
        ).strip().lower()
        if user_wants_to_select not in ("y", "n"):
            logger.warning("Invalid input. Starting without interactive selection.")
            user_wants_to_select = "n"

        for idx, ((patient_id, study_date), study_info) in enumerate(sorted(self.grouped_series.items())):
            user_flags[patient_id] = []
            if patient_id not in patient_conversion_flags:
                patient_conversion_flags[patient_id] = []
            logger.info(
                f"📚 Study {idx + 1} of {len(self.grouped_series)} — Patient ID: {patient_id} — "
                f"Study Date: {study_date} - Study Desc: {study_info[0]['StudyDescription']}"
            )
            logger.info("Available Series:")

            preselected_indices, fallback_flag = self.find_default_indices(study_info)

            for i, s in enumerate(study_info):
                pre = i in preselected_indices
                mark = "[*]" if pre else "[ ]"
                logger.info(f"{mark} [{i:2}] {s['Modality']:>3} | {s['SeriesDescription']}")

            if fallback_flag:
                logger.info(
                    f"⚠️ No {self.series_keywords[study_info[0]['Modality']]['PRIMARY']} found — "
                    f"defaulted to {self.series_keywords[study_info[0]['Modality']]['SECONDARY']} "
                    f"and flagged study."
                )

            default_input = ",".join(str(i) for i in preselected_indices)
            if user_wants_to_select == "n":
                logger.info(
                    f"Skipping interactive selection for Patient ID: {patient_id} - "
                    f"Study Date: {study_date}. Using preselected indices: {default_input}"
                )
                user_input = default_input
            else:
                user_input = (
                    input(f"Enter numbers to select (comma-separated), add 'x' to flag study [default: {default_input}]: ")
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

            selected_series = {patient_id: [study_info[i] for i in indices]}
            if patient_id not in self.patient_results:
                self.patient_results[patient_id] = {
                    "InputDirPath": str(selected_series[patient_id][0]["PatientPath"]),
                    **extract_dicom_data(plb.Path(selected_series[patient_id][0]["SeriePath"]), self.patient_tags),
                    "Studies": {},
                }

            serie_conversion_flags = self.handle_selected_series(selected_series)
            if serie_conversion_flags:
                for flag in serie_conversion_flags:
                    patient_conversion_flags[patient_id].append(flag)

        for patient_id in self.patient_results:
            self.validate_output(
                data_dict=self.patient_results[patient_id],
                output_csv_path=os.path.join(self.output_dirpath, "validation_results.csv"),
                user_flag=bool(any(user_flags[patient_id])),
                conversion_flags=patient_conversion_flags.get(patient_id, []),
            )

            with open(os.path.join(self.output_dirpath, patient_id, "patient_info.json"), "w") as f:
                self.patient_results[patient_id] = make_json_safe(self.patient_results[patient_id])
                json.dump(self.patient_results[patient_id], f)

    def find_default_indices(self, series_list: list) -> tuple[list, bool]:
        """Find default indices based on series keywords.
        This method checks the series descriptions against predefined keywords for primary and secondary selection.
        It returns a list of indices for the selected series and a flag indicating if secondary keywords were used.
        """
        preselected_indices = []
        secondary_used = False

        # Track modality-wise matches
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

            # Initialize modality record if needed
            if modality not in modality_matches:
                modality_matches[modality] = {"primary": [], "secondary": []}

            # Classify into primary or secondary match
            if any(pk in desc for pk in primary_keywords):
                modality_matches[modality]["primary"].append(i)
            elif any(sk in desc for sk in secondary_keywords):
                modality_matches[modality]["secondary"].append(i)

        # Combine selected indices and evaluate flag
        for match in modality_matches.values():
            if match["primary"]:
                preselected_indices.extend(match["primary"])
            elif match["secondary"]:
                preselected_indices.extend(match["secondary"])
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

        for i, serie in enumerate(selected_series[patient_id]):
            study_date = serie["StudyDate"]
            modality = serie["Modality"]
            serie_desc = serie["SeriesDescription"]
            serie_path = serie["SeriePath"]
            study_path = serie["StudyPath"]

            logger.info(
                f"Processing Patient ID: {patient_id}, Date: {study_date}, "
                f"Modality: {modality}, Description: {serie_desc}"
            )
            if i == 0:
                self.patient_results[patient_id]["Studies"].update(
                    {
                        study_date: {
                            "InputDirPath": str(study_path),
                            "StudyDescription": serie["StudyDescription"],
                            **extract_dicom_data(serie_path, self.serie_tags),
                            "Modalities": {},
                        }
                    }
                )
            if modality not in self.patient_results[patient_id]["Studies"][study_date]["Modalities"]:
                self.patient_results[patient_id]["Studies"][study_date]["Modalities"].update({modality: []})

            flag, paths_and_dicom_tags = self.start_dcm2nii(
                modality=modality,
                dicom_input_dirpath=serie_path,
                out_dirpath=os.path.join(self.output_dirpath, patient_id, study_date),
            )
            if paths_and_dicom_tags:
                self.patient_results[patient_id]["Studies"][study_date]["Modalities"][modality].append(
                    paths_and_dicom_tags
                )

            if flag:
                flags.append([patient_id, study_date, serie_desc, serie_path])
        logger.info("-" * 90)
        return flags

    def start_dcm2nii(self, modality, dicom_input_dirpath, out_dirpath) -> tuple[bool, dict]:
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
                    MR_dcm_dirpath=dicom_input_dirpath, output_dirpath=out_dirpath
                )

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
                dicom_tags = {}
                # convert dicom directory to nifti
                # (store results in temp directory)
                run_dcm2niix(CT_dcm_dirpath, plb.Path(tmp))
                if len(os.listdir(tmp)) == 2:
                    nii = next(tmp.glob("*nii.gz"))
                elif len(os.listdir(tmp)) == 3:
                    nii = next(tmp.glob("*Eq_1.nii.gz"))
                else:
                    # raise ValueError("CT conversion failed")
                    logger.info("CT conversion failed")

                # copy niftis to output folder with consistent naming
                out_fpath = os.path.join(output_dirpath, "CT.nii.gz")
                shutil.copy(nii, out_fpath)
                nii = next(tmp.glob("*json"))
                with open(nii) as json_file:
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
        if os.path.isfile(out_pet_fpath) and os.path.isfile(out_suv_fpath):
            logger.info(f"PET NIfTI and SUV NIfTI already exist at {out_pet_fpath} and {out_suv_fpath}")
            dicom_tags = extract_dicom_data(plb.Path(PET_dcm_dirpath), self.dicom_tags)
            return dicom_tags
        else:
            first_pt_dcm = os.listdir(PET_dcm_dirpath)[0]
            suv_corr_factor = self.calculate_suv_factor(os.path.join(PET_dcm_dirpath, first_pt_dcm))

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

                # convert pet images to quantitative suv images and save nifti file
                out_suv_fpath = os.path.join(output_dirpath, "SUV.nii.gz")
                suv_pet_nii = self.convert_pet(
                    nib.load(os.path.join(output_dirpath, "PET.nii.gz")),
                    suv_factor=suv_corr_factor,  # type: ignore
                )
                nib.save(img=suv_pet_nii, filename=out_suv_fpath)  # type: ignore
            return dicom_tags

    def convert_dcm2nii_MR(
        self, MR_dcm_dirpath: str | os.PathLike, output_dirpath: str | os.PathLike
    ) -> tuple[str | os.PathLike, dict]:
        """Conversion of MR DICOM (in the MR_dcm_path) to nifti and save in output_dirpath.
        Args:
            MR_dcm_dirpath (str | os.PathLike): Directory containing the MR DICOM files.
            output_dirpath (str | os.PathLike): Directory to save the converted NIfTI files.

        Returns:
            tuple: Path to the NIfTI file and a dictionary of DICOM tags."""
        first_dcm = os.listdir(MR_dcm_dirpath)[0]
        ds = pydicom.dcmread(str(str(MR_dcm_dirpath) + "/" + first_dcm), stop_before_pixels=True)
        series_desc = str(ds.SeriesDescription).lower().replace("  ", "_").replace(" ", "_")
        nii_files = [f for f in os.listdir(output_dirpath) if series_desc in f.lower() and f.endswith(".nii.gz")]
        if nii_files:
            logger.info(f"MRI NIfTI already exist at {output_dirpath}.")
            dicom_tags = extract_dicom_data(plb.Path(MR_dcm_dirpath), self.dicom_tags)
            return os.path.join(output_dirpath, nii_files[0]), dicom_tags

        with tempfile.TemporaryDirectory() as tmp:
            tmp = plb.Path(tmp)

            run_dcm2niix(MR_dcm_dirpath, tmp)

            nii_files = list(tmp.glob("*.nii.gz"))

            if len(nii_files) == 0:
                logger.warning("No NIfTI files found. MRI conversion may have failed.")
                return "", {}

            try:
                nii = next(tmp.glob("*nii.gz"))
            except StopIteration:
                logger.info("MR conversion failed")

            # copy niftis to output folder with consistent naming
            nii_path = os.path.join(output_dirpath, nii.name)
            shutil.copy(nii, nii_path)
            jsn = next(tmp.glob("*.json"))
            with open(jsn) as json_file:
                dicom_tags = json.load(json_file)
        return nii_path, dicom_tags

    def calculate_suv_factor(self, dcm_path: str | os.PathLike) -> float:
        """Calculation of the SUV conversion factor"""
        ds = pydicom.dcmread(dcm_path)
        total_dose = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
        start_time = ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime
        half_life = ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
        acq_time = ds.AcquisitionTime
        weight = ds.PatientWeight
        time_diff = conv_time(acq_time) - conv_time(start_time)
        act_dose = total_dose * 0.5 ** (time_diff / half_life)
        suv_factor = 1000 * weight / act_dose
        return suv_factor

    def convert_pet(self, pet, suv_factor) -> nib.Nifti1Image:
        """Conversion of PET values to SUV (should work on Siemens PET/CT)"""
        affine = pet.affine
        pet_data = pet.get_fdata()
        pet_suv_data = (pet_data * suv_factor).astype(np.float32)
        pet_suv = nib.Nifti1Image(pet_suv_data, affine)  # type: ignore
        return pet_suv

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
    logger = create_logger()

    import argparse

    parser = argparse.ArgumentParser(description="Selection of DICOM series for conversion to NIfTI format.")
    parser.add_argument("--input-dir", help="Path to PET/CT input directory.", required=True)
    parser.add_argument("--output-dir", help="Path to designated output directory.", required=True)
    parser.add_argument(
        "--ct-primary-keywords", help="List of keywords to look for in CT study descriptions for default selection."
    )
    parser.add_argument(
        "--ct-secondary-keywords",
        help="List of keywords to look for in CT study descriptions for alternative selection.",
    )
    parser.add_argument("--ct-exclusion-keywords", help="List of keywords to exclude CT studies from selection.")
    parser.add_argument(
        "--pt-primary-keywords", help="List of keywords to look for in PT study descriptions for default selection."
    )
    parser.add_argument(
        "--pt-secondary-keywords",
        help="List of keywords to look for in PT study descriptions for alternative selection.",
    )
    parser.add_argument("--pt-exclusion-keywords", help="List of keywords to exclude PT studies from selection.")
    parser.add_argument(
        "--mr-primary-keywords", help="List of keywords to look for in MR study descriptions for default selection."
    )
    parser.add_argument(
        "--mr-secondary-keywords",
        help="List of keywords to look for in MR study descriptions for alternative selection.",
    )
    parser.add_argument("--mr-exclusion-keywords", help="List of keywords to exclude MR studies from selection.")
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
