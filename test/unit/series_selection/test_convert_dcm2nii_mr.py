"""Unit tests for convert_dcm2nii_MR's branches not already covered by the real-dcm2niix
integration tests (test/integration/test_mr_conversion.py): delegating to the dynamic-series
stacker, a failed conversion producing no NIfTI, the multi-NIfTI ranking/discard logging, and
the sidecar-fallback lookup. These mock run_dcm2niix to keep the edge cases deterministic and
independent of the real binary's exact output for a given input.
"""

import json
import logging

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture()
def mr_series_dir(dicom_series_factory):
    return dicom_series_factory("MR", n_slices=3, rows=4, cols=4, series_description="t1 tse")


def test_delegates_to_dynamic_mr_conversion_when_siblings_are_given(mocker, collector, tmp_path, mr_series_dir):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    sibling_dirs = ["/dicom/t0", "/dicom/t1", "/dicom/t2"]
    convert_dynamic = mocker.patch.object(
        collector, "_convert_dynamic_mr", return_value=("/out/stacked.nii.gz", {"SeriesDescription": "dyn"})
    )

    result = collector.convert_dcm2nii_MR(
        MR_dcm_dirpath=mr_series_dir, output_dirpath=out_dir, dynamic_sibling_dirs=sibling_dirs
    )

    convert_dynamic.assert_called_once_with(sibling_dirs, out_dir, mr_series_dir)
    assert result == ("/out/stacked.nii.gz", {"SeriesDescription": "dyn"})


def test_single_sibling_does_not_trigger_dynamic_conversion(mocker, collector, tmp_path, mr_series_dir):
    """dynamic_sibling_dirs of length <= 1 is not a genuine multi-timepoint case."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    convert_dynamic = mocker.patch.object(collector, "_convert_dynamic_mr")
    mocker.patch("musiq.series_selection.run_dcm2niix")  # avoid a real conversion attempt

    collector.convert_dcm2nii_MR(
        MR_dcm_dirpath=mr_series_dir, output_dirpath=out_dir, dynamic_sibling_dirs=["/dicom/only_one"]
    )

    convert_dynamic.assert_not_called()


def test_returns_empty_result_when_conversion_produces_no_nifti(mocker, collector, tmp_path, mr_series_dir, caplog):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    mocker.patch("musiq.series_selection.run_dcm2niix")  # no-op: writes nothing

    with caplog.at_level(logging.WARNING):
        nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=mr_series_dir, output_dirpath=out_dir)

    assert nii_path == ""
    assert dicom_tags == {}
    assert "MRI conversion may have failed" in caplog.text


def test_multiple_niftis_picks_most_dimensions_and_logs_discarded(mocker, collector, tmp_path, mr_series_dir, caplog):
    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), f"{output_folder}/water.nii.gz")
        with open(f"{output_folder}/water.json", "w") as f:
            json.dump({"Modality": "MR", "Contrast": "water"}, f)
        # A 4D volume (more timepoints) must win the ranking over the plain 3D one above.
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4, 2), dtype=np.int16), np.eye(4)), f"{output_folder}/fat.nii.gz")
        with open(f"{output_folder}/fat.json", "w") as f:
            json.dump({"Modality": "MR", "Contrast": "fat"}, f)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with caplog.at_level(logging.WARNING):
        nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=mr_series_dir, output_dirpath=out_dir)

    assert str(nii_path).endswith("fat.nii.gz")
    assert dicom_tags["Contrast"] == "fat"
    assert "discarded" in caplog.text
    assert "water.nii.gz" in caplog.text


def test_sidecar_lookup_falls_back_to_any_json_when_stem_does_not_match(mocker, collector, tmp_path, mr_series_dir):
    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.int16), np.eye(4)), f"{output_folder}/series_1.nii.gz")
        with open(f"{output_folder}/series.json", "w") as f:
            json.dump({"Modality": "MR"}, f)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=mr_series_dir, output_dirpath=out_dir)

    assert dicom_tags["Modality"] == "MR"
