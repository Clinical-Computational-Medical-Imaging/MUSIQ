"""Unit tests for convert_dcm2nii_PET's volume-selection and sidecar lookup, mocking
run_dcm2niix so multi-candidate dcm2niix output (a bundled NAC/CTAC pair, a derived MIP, ...)
can be constructed deterministically -- real PET DICOM data never reliably reproduces this on
demand.
"""

import json
import logging

import nibabel as nib
import numpy as np
import pytest

from .._dicom_builder import pet_radiopharm_tags


@pytest.fixture()
def pet_series_dir(dicom_series_factory):
    tags = pet_radiopharm_tags()
    return dicom_series_factory(
        "PT",
        subdir="pet_series",
        n_slices=3,
        rows=8,
        cols=8,
        pixel_value=lambda i: 100,
        pixel_representation=0,
        extra_tags={
            "PatientWeight": 80.0,
            "SeriesTime": "120000",
            "DecayCorrection": "START",
            **tags,
        },
    )


def _write_nii(path, shape=(4, 4, 4), value=0):
    data = np.full(shape, value, dtype=np.int16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_multiple_volumes_uses_imagetype_ranking_not_glob_order(mocker, collector, tmp_path, pet_series_dir, caplog):
    """A derived MIP (fewer slices, no sidecar) alongside the real whole-body series (more
    slices, PRIMARY sidecar) -- the whole-body PRIMARY volume must be chosen, not whichever
    happens to sort first."""

    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        _write_nii(f"{output_folder}/aaa_mip.nii.gz", shape=(4, 4, 2), value=1)  # sorts first
        _write_nii(f"{output_folder}/zzz_wb.nii.gz", shape=(4, 4, 20), value=2)
        with open(f"{output_folder}/zzz_wb.json", "w") as f:
            json.dump({"Modality": "PT", "ImageType": ["ORIGINAL", "PRIMARY"]}, f)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with caplog.at_level(logging.WARNING):
        dicom_tags = collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_series_dir, output_dirpath=out_dir)

    assert dicom_tags["ImageType"] == ["ORIGINAL", "PRIMARY"]
    assert "discarded" in caplog.text
    pet = nib.load(str(out_dir / "PET.nii.gz"))
    assert (pet.get_fdata() == 2).all()  # the whole-body volume, not the MIP


def test_sidecar_lookup_prefers_exact_stem_over_any_json(mocker, collector, tmp_path, pet_series_dir):
    """Regression test: convert_dcm2nii_PET used to grab `next(tmp.glob("*json"))` unconditionally,
    without ever trying the sidecar matching the chosen volume's own stem first."""

    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        _write_nii(f"{output_folder}/chosen.nii.gz")
        with open(f"{output_folder}/chosen.json", "w") as f:
            json.dump({"Modality": "PT", "SeriesDescription": "correct match"}, f)
        _write_nii(f"{output_folder}/other.nii.gz")
        with open(f"{output_folder}/other.json", "w") as f:
            json.dump({"Modality": "PT", "SeriesDescription": "wrong match"}, f)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Force _select_pet_volume to return the "chosen" candidate deterministically, isolating the
    # sidecar-lookup behavior under test from the ranking behavior covered elsewhere.
    mocker.patch.object(collector, "_select_pet_volume", side_effect=lambda tmp, _: next(tmp.glob("chosen.nii.gz")))

    dicom_tags = collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_series_dir, output_dirpath=out_dir)

    assert dicom_tags["SeriesDescription"] == "correct match"


def test_sidecar_lookup_falls_back_to_any_json_and_warns_when_stem_does_not_match(
    mocker, collector, tmp_path, pet_series_dir, caplog
):
    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        _write_nii(f"{output_folder}/series_1.nii.gz")  # no matching sidecar
        with open(f"{output_folder}/series.json", "w") as f:
            json.dump({"Modality": "PT"}, f)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with caplog.at_level(logging.WARNING):
        dicom_tags = collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_series_dir, output_dirpath=out_dir)

    assert dicom_tags["Modality"] == "PT"
    assert "No JSON sidecar matching" in caplog.text
