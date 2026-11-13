"""Unit tests for SeriesSelection._select_pet_volume: picking which PET volume to keep when
dcm2niix emits more than one NIfTI for a series (e.g. a bundled NAC/CTAC pair, or a derived MIP
alongside the whole-body series).
"""

import json

import nibabel as nib
import numpy as np
import pytest


def _write_nii(path, shape=(4, 4, 4)):
    data = np.zeros(shape, dtype=np.int16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def _write_sidecar(nii_path, image_type):
    sidecar = nii_path.with_suffix("").with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump({"ImageType": image_type}, f)


def test_no_nifti_files_raises_value_error(collector, tmp_path):
    with pytest.raises(ValueError, match="no NIfTI files"):
        collector._select_pet_volume(tmp_path, "/dicom/pet")


def test_single_nifti_is_returned_without_ranking(collector, tmp_path):
    nii = tmp_path / "series_1.nii.gz"
    _write_nii(nii)

    result = collector._select_pet_volume(tmp_path, "/dicom/pet")

    assert result == nii


def test_primary_image_type_is_preferred_over_secondary(collector, tmp_path, caplog):
    whole_body = tmp_path / "zzz_wb.nii.gz"
    mip = tmp_path / "aaa_mip.nii.gz"
    _write_nii(whole_body, shape=(4, 4, 20))
    _write_nii(mip, shape=(4, 4, 2))  # fewer slices, no sidecar -- must still lose
    _write_sidecar(whole_body, ["ORIGINAL", "PRIMARY", "AXIAL"])

    result = collector._select_pet_volume(tmp_path, "/dicom/pet")

    assert result == whole_body
    assert "discarded" in caplog.text


def test_most_slices_wins_when_image_type_ties(collector, tmp_path):
    fewer = tmp_path / "series_a.nii.gz"
    more = tmp_path / "series_b.nii.gz"
    _write_nii(fewer, shape=(4, 4, 4))
    _write_nii(more, shape=(4, 4, 20))

    result = collector._select_pet_volume(tmp_path, "/dicom/pet")

    assert result == more
