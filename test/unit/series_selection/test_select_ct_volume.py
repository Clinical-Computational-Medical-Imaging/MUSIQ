import json

import nibabel as nib
import numpy as np
import pytest


def _write_nii(path, shape=(4, 4, 4), affine=None):
    data = np.zeros(shape, dtype=np.int16)
    nib.save(nib.Nifti1Image(data, affine if affine is not None else np.eye(4)), str(path))


def _write_sidecar(nii_path, image_type):
    sidecar = nii_path.with_suffix("").with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump({"ImageType": image_type}, f)


def test_no_nifti_files_raises_value_error(collector, tmp_path):
    with pytest.raises(ValueError, match="no NIfTI files"):
        collector._select_ct_volume(tmp_path, "/dicom/ct")


def test_single_nifti_is_returned_without_ranking(collector, tmp_path):
    nii = tmp_path / "series_1.nii.gz"
    _write_nii(nii)
    # No sidecar at all — the single-file shortcut must not need one.

    result = collector._select_ct_volume(tmp_path, "/dicom/ct")

    assert result == nii


def test_eq_1_volume_is_preferred_regardless_of_other_candidates(collector, tmp_path):
    plain = tmp_path / "series_1.nii.gz"
    eq = tmp_path / "series_1_Eq_1.nii.gz"
    _write_nii(plain, shape=(8, 8, 8))
    _write_nii(eq, shape=(2, 2, 2))

    result = collector._select_ct_volume(tmp_path, "/dicom/ct")

    assert result == eq


def test_primary_image_type_is_preferred_over_secondary(collector, tmp_path, caplog):
    primary = tmp_path / "series_a.nii.gz"
    secondary = tmp_path / "series_b.nii.gz"
    _write_nii(primary, shape=(4, 4, 4))
    _write_nii(secondary, shape=(4, 4, 10))  # more slices, but SECONDARY must still lose
    _write_sidecar(primary, ["ORIGINAL", "PRIMARY", "AXIAL"])
    _write_sidecar(secondary, ["ORIGINAL", "SECONDARY", "AXIAL"])

    result = collector._select_ct_volume(tmp_path, "/dicom/ct")

    assert result == primary
    assert "discarded" in caplog.text


def test_most_slices_wins_when_image_type_ties(collector, tmp_path):
    fewer = tmp_path / "series_a.nii.gz"
    more = tmp_path / "series_b.nii.gz"
    _write_nii(fewer, shape=(4, 4, 4))
    _write_nii(more, shape=(4, 4, 12))
    _write_sidecar(fewer, ["ORIGINAL", "PRIMARY"])
    _write_sidecar(more, ["ORIGINAL", "PRIMARY"])

    result = collector._select_ct_volume(tmp_path, "/dicom/ct")

    assert result == more


def test_missing_sidecar_is_treated_as_non_primary(collector, tmp_path):
    no_sidecar = tmp_path / "series_a.nii.gz"
    primary = tmp_path / "series_b.nii.gz"
    _write_nii(no_sidecar, shape=(4, 4, 20))  # more slices, but no ImageType info at all
    _write_nii(primary, shape=(4, 4, 4))
    _write_sidecar(primary, ["ORIGINAL", "PRIMARY"])

    result = collector._select_ct_volume(tmp_path, "/dicom/ct")

    assert result == primary


def test_largest_file_size_is_final_tiebreak(collector, tmp_path):
    small = tmp_path / "series_a.nii.gz"
    large = tmp_path / "series_b.nii.gz"
    _write_nii(small, shape=(4, 4, 4))
    _write_nii(large, shape=(4, 4, 4))
    # Pad one file so it is strictly larger on disk while shape/ImageType tie.
    with open(large, "ab") as f:
        f.write(b"\0" * 4096)

    result = collector._select_ct_volume(tmp_path, "/dicom/ct")

    assert result == large
