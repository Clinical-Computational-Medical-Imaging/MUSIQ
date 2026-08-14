"""Real DICOM -> NIfTI conversion tests for convert_dcm2nii_PET, against a genuine whole-body
PET reconstruction downloaded from TCIA (see conftest.py for how to point pytest at the data).
"""

import nibabel as nib
import pytest

from .conftest import find_series_dir

pytestmark = [pytest.mark.usefixtures("dcm2niix_available")]


@pytest.fixture()
def whole_body_pet_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-VP-A879",
        study_glob="Prostate CA PET",
        series_glob="PET WB",
    )


def test_real_whole_body_pet_converts_to_a_plausible_suv_volume(collector, tmp_path, whole_body_pet_series_dir):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    dicom_tags = collector.convert_dcm2nii_PET(PET_dcm_dirpath=whole_body_pet_series_dir, output_dirpath=out_dir)

    assert dicom_tags["Modality"] == "PT"
    assert dicom_tags["PatientWeight"] == pytest.approx(83.99)
    assert dicom_tags["DecayCorrectionReference"] == "START"
    assert dicom_tags["SUVFactor"] > 0

    pet = nib.load(str(out_dir / "PET.nii.gz"))
    suv = nib.load(str(out_dir / "SUV.nii.gz"))
    assert pet.shape[2] == 356  # stitched whole-body bed positions, not a cropped conversion
    assert suv.shape == pet.shape

    suv_data = suv.get_fdata()
    assert suv_data.min() >= 0
    assert 0 < suv_data.mean() < 5
    assert suv_data.max() < 50
