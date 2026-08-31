"""Unit tests for utils.repair_slice_direction_from_dicom: correcting a CT NIfTI's
superior-inferior sign from the DICOM ImagePositionPatient order.

series_selection's own unit tests (test_convert_dcm2nii_ct.py) mock this function out entirely
("These mock run_dcm2niix/select_dominant_ct_acquisition/repair_slice_direction_from_dicom/
repair_slice_spacing_from_dicom rather than driving them with a real dcm2niix conversion"), and
the one real-data integration test that
exercises the sibling repair_slice_spacing_from_dicom does not provoke a sign flip. So the actual
sign-correction logic had zero coverage of its own. These build a synthetic DICOM series (known
ImagePositionPatient order) via ``dicom_series_factory`` and a synthetic NIfTI with a
controllable affine via ``ct_nifti_factory``, so the exact geometry is known rather than left to
whatever a real download happens to contain.
"""

import pathlib as plb

import nibabel as nib
import numpy as np
import pytest

from musiq.utils import repair_slice_direction_from_dicom


@pytest.fixture()
def ct_dicom_dir(dicom_series_factory):
    """4 axial CT slices at z = 0, 10, 20, 30 (ImagePositionPatient), identity in-plane
    orientation -- so the slice normal is the pure z axis and proj_min/proj_max = 0/30."""
    return dicom_series_factory("CT", subdir="ct_geom", n_slices=4, rows=4, cols=4, slice_spacing=10.0)


def test_wrong_sign_is_flipped_to_match_dicom_order(ct_dicom_dir, ct_nifti_factory):
    """Voxel index 0 sits at the DICOM's proj_min (z=0), so the slice axis must increase with
    index (+z) -- a NIfTI recorded with the opposite (-z) sign must be flipped."""
    nifti_path = ct_nifti_factory(slice_col=(0.0, 0.0, -10.0), origin=(0.0, 0.0, 0.0))

    corrected = repair_slice_direction_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is True
    new_col = nib.load(nifti_path).affine[:3, 2]
    assert new_col == pytest.approx([0.0, 0.0, 10.0])


def test_already_correct_sign_is_left_untouched(ct_dicom_dir, ct_nifti_factory):
    nifti_path = ct_nifti_factory(slice_col=(0.0, 0.0, 10.0), origin=(0.0, 0.0, 0.0))
    original_bytes = plb.Path(nifti_path).read_bytes()

    corrected = repair_slice_direction_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is False
    assert plb.Path(nifti_path).read_bytes() == original_bytes


def test_oblique_series_is_left_untouched(ct_dicom_dir, ct_nifti_factory):
    """A slice axis that isn't (anti-)parallel to the DICOM slice normal is gantry-tilted/oblique
    geometry -- repairing it from the pure slice normal would be wrong, so it must be a no-op."""
    nifti_path = ct_nifti_factory(slice_col=(7.0, 7.0, 7.0), origin=(0.0, 0.0, 0.0))

    corrected = repair_slice_direction_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is False


def test_single_slice_nifti_is_left_untouched(ct_dicom_dir, ct_nifti_factory):
    nifti_path = ct_nifti_factory(shape=(4, 4, 1), slice_col=(0.0, 0.0, 10.0))

    corrected = repair_slice_direction_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is False


def test_undeterminable_dicom_geometry_is_left_untouched(tmp_path, ct_nifti_factory):
    """No DICOM files at all (or too few to derive a normal/extent) -> can't determine the
    correct sign, so the NIfTI must be left as-is rather than guessing."""
    empty_dicom_dir = tmp_path / "no_dicoms"
    empty_dicom_dir.mkdir()
    nifti_path = ct_nifti_factory(slice_col=(0.0, 0.0, -10.0))

    corrected = repair_slice_direction_from_dicom(nifti_path, empty_dicom_dir)

    assert corrected is False


def test_non_file_and_unreadable_dicom_dir_entries_are_skipped(ct_dicom_dir, ct_nifti_factory):
    """A stray subdirectory and a non-DICOM file living alongside the series must not break the
    slice-normal/extent scan or change its result."""
    plb.Path(ct_dicom_dir, "subdir").mkdir()
    plb.Path(ct_dicom_dir, "garbage.dcm").write_bytes(b"not a real dicom file at all")
    nifti_path = ct_nifti_factory(slice_col=(0.0, 0.0, -10.0), origin=(0.0, 0.0, 0.0))

    corrected = repair_slice_direction_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is True
    new_col = nib.load(nifti_path).affine[:3, 2]
    assert new_col == pytest.approx([0.0, 0.0, 10.0])


def test_zero_length_slice_column_is_left_untouched(mocker, ct_dicom_dir):
    """A zero-length slice-axis affine column is degenerate enough that nibabel itself refuses to
    write such a NIfTI to disk (its own affine decomposition fails) -- so, like the analogous
    guard in repair_slice_spacing_from_dicom, this is exercised by mocking the loaded image
    directly instead of round-tripping through a real file. Unlike repair_slice_spacing_from_dicom
    (which checks the column's own magnitude first), this function relies entirely on
    _is_axial_along_normal's internal zero-norm guard to catch it."""
    fake_img = mocker.Mock(ndim=3, shape=(4, 4, 4))
    fake_img.affine = np.eye(4)
    fake_img.affine[:3, 2] = (0.0, 0.0, 0.0)
    mocker.patch("musiq.utils.nib.load", return_value=fake_img)

    corrected = repair_slice_direction_from_dicom("unused/path.nii.gz", ct_dicom_dir)

    assert corrected is False
