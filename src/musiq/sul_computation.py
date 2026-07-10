import json
import logging
import os
import pathlib as plb

import nibabel as nib
import pydicom

from .utils import calculate_suv_factor, convert_pet, list_dicom_files, list_patient_dirs

logger = logging.getLogger(__name__)


class SulInference:
    """Compute the lean-body-mass-corrected PET (SUL) image for each PET/CT study.

    CPU-only (no torch): reads ``PatientLBM`` from the muscle_fat stage and writes ``SUL.nii.gz``.
    Idempotent: skips a study when SUL.nii.gz exists, there is no PET.nii.gz, or PatientLBM is absent.
    """

    def __init__(self, input_dirpath_processed: str | os.PathLike) -> None:
        """
        Args:
            input_dirpath_processed: Processed output tree (processed/<patient>/<study_date>/).
        """
        self.input_dirpath = input_dirpath_processed

    def run(self) -> None:
        if not os.path.isdir(self.input_dirpath):
            logger.error(f"Error: {self.input_dirpath} is not a valid directory.")
            return

        logger.info(f"Starting SUL computation in {self.input_dirpath}")
        for top_dir in list_patient_dirs(self.input_dirpath):
            top_dir_path = os.path.join(self.input_dirpath, top_dir)
            for dirpath, dirnames, _filenames in os.walk(top_dir_path):
                rel_parts = plb.Path(os.path.relpath(dirpath, self.input_dirpath)).parts
                if len(rel_parts) != 2:
                    continue
                dirnames.clear()
                patient_id, study_date = rel_parts
                try:
                    self._process_study(dirpath, patient_id, study_date)
                except Exception:
                    logger.exception(f"Failed SUL computation for {patient_id} ({study_date}); skipping.")

    def _process_study(self, dirpath: str, patient_id: str, study_date: str) -> None:
        pet_fpath = os.path.join(dirpath, "PET.nii.gz")
        if not os.path.isfile(pet_fpath):
            return  # CT-only study: nothing to derive SUL from.

        sul_fpath = os.path.join(dirpath, "SUL.nii.gz")
        patient_info_path = os.path.join(os.path.dirname(dirpath), "patient_info.json")
        if not os.path.isfile(patient_info_path):
            logger.error(f"Missing patient_info.json for {patient_id}; cannot compute SUL.")
            return

        with open(patient_info_path) as f:
            data = json.load(f)

        lbm = self._patient_lbm(data, study_date)
        if lbm is None:
            logger.warning(
                f"No PatientLBM for {patient_id} ({study_date}) — run muscle_fat first, or "
                "PatientWeight/body-composition was unavailable. Skipping SUL."
            )
            return

        if os.path.isfile(sul_fpath):
            logger.info(f"SUL.nii.gz already exists for {patient_id} ({study_date}).")
            sul_path = sul_fpath
        else:
            # Persist any DICOM timing fields recovered by convert_pet2sul
            sul_path = self.convert_pet2sul(dirpath, data, study_date, lbm)
            if sul_path is None:
                with open(patient_info_path, "w") as f:
                    json.dump(data, f)
                return

        pt_series = data["Studies"][study_date]["Modalities"].get("PT")
        if pt_series:
            pt_series_name = next(iter(pt_series[0]))
            pt_series[0][pt_series_name]["SULPath"] = sul_path
        else:
            logger.warning(
                f"SUL available for {patient_id} but no PT entry in patient_info.json; SULPath not recorded."
            )
        with open(patient_info_path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def _patient_lbm(data: dict, study_date: str) -> float | None:
        """Return the ``PatientLBM`` recorded on the CT series (series 0) by the muscle_fat stage."""
        try:
            ct_series = data["Studies"][study_date]["Modalities"]["CT"][0]
            series_name = next(iter(ct_series))
            return ct_series[series_name].get("PatientLBM")
        except (KeyError, IndexError, StopIteration, TypeError):
            return None

    def convert_pet2sul(
        self, output_dirpath: str | os.PathLike, data: dict, study_date: str, lean_body_mass: float
    ) -> os.PathLike | None:
        """Convert PET.nii.gz to a lean-body-mass-corrected SUL image.

        Args:
            output_dirpath: Study directory holding PET.nii.gz / SUL.nii.gz.
            data: The loaded patient_info.json (mutated in place if DICOM fields are recovered).
            study_date: Study date key for patient_info.json access.
            lean_body_mass: Body weight without the fat weight.

        Returns:
            Path to SUL.nii.gz, or None if it could not be produced.
        """
        out_pet_fpath = os.path.join(output_dirpath, "PET.nii.gz")
        out_sul_fpath = os.path.join(output_dirpath, "SUL.nii.gz")

        if os.path.isfile(out_sul_fpath):
            logger.info(f"SUL NIfTI already exists at {out_sul_fpath}")
            return out_sul_fpath

        pt_list = data["Studies"][study_date]["Modalities"].get("PT")
        if not pt_list:
            logger.error(f"No PT series in patient_info.json for {output_dirpath}; cannot compute SUL.")
            return None
        series_name = next(iter(pt_list[0]))
        pt_series = pt_list[0][series_name]
        dicom_data = pt_series["DICOM"]

        # Preferred: reuse SUV factor. sul_factor = suv_factor * LBM / weight (shared decay reference)
        sul_corr_factor = None
        suv_factor = dicom_data.get("SUVFactor")
        pt_weight = dicom_data.get("PatientWeight")
        if suv_factor and pt_weight:
            try:
                w = float(pt_weight)
                if w > 0:
                    sul_corr_factor = float(suv_factor) * lean_body_mass / w
            except (TypeError, ValueError):
                sul_corr_factor = None

        if sul_corr_factor is None:
            # Fallback if SUVFactor wasn't recorded: recompute from DICOM timing
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
                        ds = pydicom.dcmread(list_dicom_files(input_dir)[0])
                        seq = ds.RadiopharmaceuticalInformationSequence[0]
                        recoverable = {
                            "RadiopharmaceuticalStartTime": str(seq.RadiopharmaceuticalStartTime),
                            "InjectedRadioactivity": float(seq.RadionuclideTotalDose),
                            "RadionuclideHalfLife": float(seq.RadionuclideHalfLife),
                        }
                        for k in missing:
                            if k in recoverable:
                                dicom_data[k] = recoverable[k]
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

            sul_corr_factor = calculate_suv_factor(
                total_dose=dicom_data["InjectedRadioactivity"],
                start_time=dicom_data["RadiopharmaceuticalStartTime"],
                half_life=dicom_data["RadionuclideHalfLife"],
                acq_time=dicom_data["AcquisitionTime"],
                weight=lean_body_mass,
            )

        sul_pet_nii = convert_pet(nib.load(out_pet_fpath), suv_factor=sul_corr_factor)  # type: ignore
        nib.save(img=sul_pet_nii, filename=out_sul_fpath)  # type: ignore
        return out_sul_fpath


def sul_computation_entrypoint() -> None:
    """Entry point to run the SUL stage without the full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.sul")

    import argparse

    parser = argparse.ArgumentParser(
        description="Compute the lean-body-mass-corrected PET image (SUL.nii.gz) for a processed tree "
        "(CPU-only; run muscle_fat first)."
    )
    parser.add_argument(
        "--input-dirpath-processed",
        type=str,
        help="Path to the input folder containing the patient folders with studies and nifti files",
        required=True,
    )
    args = parser.parse_args()

    SulInference(input_dirpath_processed=args.input_dirpath_processed).run()


if __name__ == "__main__":
    sul_computation_entrypoint()
