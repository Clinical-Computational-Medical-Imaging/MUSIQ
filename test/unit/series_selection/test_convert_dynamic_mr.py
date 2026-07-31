"""Tests for _convert_dynamic_mr: stacking per-timepoint series into one 4D NIfTI.

The happy path runs the real dcm2niix binary against small synthetic per-timepoint DICOM
series (so the stacking/ordering logic is verified against genuine conversion output); the
fallback branches (no NIfTI produced / not 3D / geometry mismatch) are exercised by faking
run_dcm2niix's output directly, since they describe conversion *failures* that are awkward to
provoke through a real binary and are about series_selection's own recovery logic, not dcm2niix.
"""

import logging

import nibabel as nib
import numpy as np
import pytest


def _make_timepoint_dirs(dicom_series_factory, n_timepoints=3, pixel_values=None):
    pixel_values = pixel_values or [10 * (i + 1) for i in range(n_timepoints)]
    dirs = []
    for t in range(n_timepoints):
        d = dicom_series_factory(
            "MR",
            subdir=f"timepoint_{t}",
            n_slices=3,
            rows=6,
            cols=6,
            pixel_value=lambda i, v=pixel_values[t]: v,
            protocol_name="dyn_t1",
        )
        dirs.append(d)
    return dirs


@pytest.mark.usefixtures("dcm2niix_available")
def test_stacks_timepoints_in_given_order_with_real_conversion(collector, tmp_path, dicom_series_factory):
    sibling_dirs = _make_timepoint_dirs(dicom_series_factory, n_timepoints=3, pixel_values=[10, 20, 30])
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector._convert_dynamic_mr(sibling_dirs, out_dir, fallback_dir=sibling_dirs[0])

    assert str(nii_path).startswith(str(out_dir))
    img = nib.load(str(nii_path))
    assert img.ndim == 4
    assert img.shape[3] == 3
    data = img.get_fdata()
    # Order must follow sibling_dirs (already time-sorted by the caller), not any re-sort here.
    assert data[..., 0] == pytest.approx(10)
    assert data[..., 1] == pytest.approx(20)
    assert data[..., 2] == pytest.approx(30)
    assert dicom_tags["Modality"] == "MR"


def _write_fake_nii(out_folder, shape, affine=None):
    import os

    out_folder = str(out_folder)
    os.makedirs(out_folder, exist_ok=True)
    data = np.zeros(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, affine if affine is not None else np.eye(4)), os.path.join(out_folder, "vol.nii.gz"))


def test_falls_back_to_single_volume_when_a_timepoint_produces_no_nifti(mocker, collector, tmp_path, caplog):
    sibling_dirs = [tmp_path / "t0", tmp_path / "t1", tmp_path / "t2"]
    for d in sibling_dirs:
        d.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        if str(input_folder) == str(sibling_dirs[1]):
            return  # simulate a failed/empty conversion for the second timepoint
        _write_fake_nii(output_folder, (4, 4, 4))

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    fallback = mocker.patch.object(collector, "convert_dcm2nii_MR", return_value=("/out/fallback.nii.gz", {}))

    with caplog.at_level(logging.WARNING):
        result = collector._convert_dynamic_mr(sibling_dirs, out_dir, fallback_dir=sibling_dirs[0])

    assert "no NIfTI for timepoint" in caplog.text
    fallback.assert_called_once_with(sibling_dirs[0], out_dir)
    assert result == ("/out/fallback.nii.gz", {})


def test_falls_back_to_single_volume_when_a_timepoint_is_not_3d(mocker, collector, tmp_path):
    sibling_dirs = [tmp_path / "t0", tmp_path / "t1"]
    for d in sibling_dirs:
        d.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        shape = (4, 4, 4) if str(input_folder) == str(sibling_dirs[0]) else (4, 4, 4, 2)
        _write_fake_nii(output_folder, shape)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    fallback = mocker.patch.object(collector, "convert_dcm2nii_MR", return_value=("/out/fallback.nii.gz", {}))

    result = collector._convert_dynamic_mr(sibling_dirs, out_dir, fallback_dir=sibling_dirs[0])

    fallback.assert_called_once_with(sibling_dirs[0], out_dir)
    assert result == ("/out/fallback.nii.gz", {})


def test_falls_back_to_single_volume_on_geometry_mismatch(mocker, collector, tmp_path):
    sibling_dirs = [tmp_path / "t0", tmp_path / "t1"]
    for d in sibling_dirs:
        d.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        shape = (4, 4, 4) if str(input_folder) == str(sibling_dirs[0]) else (8, 8, 4)
        _write_fake_nii(output_folder, shape)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    fallback = mocker.patch.object(collector, "convert_dcm2nii_MR", return_value=("/out/fallback.nii.gz", {}))

    result = collector._convert_dynamic_mr(sibling_dirs, out_dir, fallback_dir=sibling_dirs[0])

    fallback.assert_called_once_with(sibling_dirs[0], out_dir)
    assert result == ("/out/fallback.nii.gz", {})
