"""Unit tests for utils.repair_slice_spacing_from_dicom: correcting a CT NIfTI's slice-axis
spacing from the physical extent of the DICOM ImagePositionPatient values.

The real-data integration test in test/integration/test_ct_conversion.py
(test_real_irregular_slice_spacing_is_repaired_to_the_dicom_derived_value) only checks the
*outcome* of a full convert_dcm2nii_CT run against one specific irregular-spacing series -- it
can't isolate or exercise the "already correct" / "degenerate column" / "oblique" / "single
slice" branches on demand. These build a synthetic DICOM series with a known, exact extent via
``dicom_series_factory`` so every branch is reachable directly.
"""

import pathlib as plb

import nibabel as nib
import numpy as np
import pytest

from musiq.utils import repair_slice_spacing_from_dicom


@pytest.fixture()
def ct_dicom_dir(dicom_series_factory):
    """4 axial CT slices spanning z = 0..30 (ImagePositionPatient) -> true spacing = 30/3 = 10mm."""
    return dicom_series_factory("CT", subdir="ct_geom", n_slices=4, rows=4, cols=4, slice_spacing=10.0)


def test_wrong_spacing_is_corrected_to_the_dicom_derived_value(ct_dicom_dir, ct_nifti_factory):
    nifti_path = ct_nifti_factory(slice_col=(0.0, 0.0, 15.0))

    corrected = repair_slice_spacing_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is True
    new_col = nib.load(nifti_path).affine[:3, 2]
    assert new_col == pytest.approx([0.0, 0.0, 10.0])


def test_direction_of_the_slice_axis_is_preserved_while_fixing_magnitude(ct_dicom_dir, ct_nifti_factory):
    """Only the magnitude is wrong here; the (negative) direction must survive the repair."""
    nifti_path = ct_nifti_factory(slice_col=(0.0, 0.0, -15.0))

    repair_slice_spacing_from_dicom(nifti_path, ct_dicom_dir)

    new_col = nib.load(nifti_path).affine[:3, 2]
    assert new_col == pytest.approx([0.0, 0.0, -10.0])


def test_already_correct_spacing_within_tolerance_is_left_untouched(ct_dicom_dir, ct_nifti_factory):
    nifti_path = ct_nifti_factory(slice_col=(0.0, 0.0, 10.0))
    original_bytes = plb.Path(nifti_path).read_bytes()

    corrected = repair_slice_spacing_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is False
    assert plb.Path(nifti_path).read_bytes() == original_bytes


def test_spacing_within_rel_tol_is_not_flagged_as_wrong(ct_dicom_dir, ct_nifti_factory):
    """9.95mm is within the default 1% relative tolerance of the true 10mm -> no-op."""
    nifti_path = ct_nifti_factory(slice_col=(0.0, 0.0, 9.95))

    corrected = repair_slice_spacing_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is False


def test_oblique_series_is_left_untouched(ct_dicom_dir, ct_nifti_factory):
    nifti_path = ct_nifti_factory(slice_col=(7.0, 7.0, 7.0))

    corrected = repair_slice_spacing_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is False


def test_degenerate_zero_length_slice_column_is_left_untouched(mocker, ct_dicom_dir, ct_nifti_factory):
    """A zero-length slice column is degenerate enough that nibabel itself refuses to write such
    a NIfTI to disk (its own affine decomposition fails) -- so this exercises the defensive
    ``current_spacing == 0`` guard by mocking the loaded image directly instead of round-tripping
    through a real file."""
    fake_img = mocker.Mock(ndim=3, shape=(4, 4, 4))
    fake_img.affine = np.eye(4)
    fake_img.affine[:3, 2] = (0.0, 0.0, 0.0)
    mocker.patch("musiq.utils.nib.load", return_value=fake_img)

    corrected = repair_slice_spacing_from_dicom("unused/path.nii.gz", ct_dicom_dir)

    assert corrected is False


def test_single_slice_nifti_is_left_untouched(ct_dicom_dir, ct_nifti_factory):
    nifti_path = ct_nifti_factory(shape=(4, 4, 1), slice_col=(0.0, 0.0, 15.0))

    corrected = repair_slice_spacing_from_dicom(nifti_path, ct_dicom_dir)

    assert corrected is False
