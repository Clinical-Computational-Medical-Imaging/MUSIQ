"""Unit tests for convert_dcm2nii_CT's rarer branches: the mixed-spacing dominant-acquisition
filter, a failing affine repair, and the sidecar-fallback lookup.

These mock run_dcm2niix/select_dominant_ct_acquisition/repair_slice_direction_from_dicom/
repair_slice_spacing_from_dicom rather than
driving them with a real dcm2niix conversion: symlink-based dominant-acquisition filtering isn't
reliably provokable through real DICOM data (it needs two mixed-spacing acquisitions bundled
under one SeriesInstanceUID), and os.symlink itself requires elevated privileges on Windows
without Developer Mode — orthogonal to what this test is verifying (that convert_dcm2nii_CT
wires the filtered file list through correctly). The plain single-acquisition conversion path
is covered with real dcm2niix data in test/integration/test_ct_conversion.py.
"""

import json
import logging

import nibabel as nib
import numpy as np


def _write_nii_and_sidecar(out_folder, stem="out", image_type=None):
    out_folder = str(out_folder)
    data = np.zeros((4, 4, 4), dtype=np.int16)
    nib.save(nib.Nifti1Image(data, np.eye(4)), f"{out_folder}/{stem}.nii.gz")
    with open(f"{out_folder}/{stem}.json", "w") as f:
        json.dump({"Modality": "CT", "ImageType": image_type or ["ORIGINAL", "PRIMARY"]}, f)


def test_dominant_acquisition_files_are_symlinked_into_a_filtered_conversion_dir(mocker, collector, tmp_path):
    dominant_files = ["/dicom/ct/f1.dcm", "/dicom/ct/f2.dcm"]
    mocker.patch("musiq.series_selection.select_dominant_ct_acquisition", return_value=dominant_files)
    symlink = mocker.patch("musiq.series_selection.os.symlink")

    captured = {}

    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        captured["conv_dcm_dirpath"] = input_folder
        _write_nii_and_sidecar(output_folder)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    mocker.patch("musiq.series_selection.repair_slice_direction_from_dicom", return_value=False)
    mocker.patch("musiq.series_selection.repair_slice_spacing_from_dicom", return_value=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    dicom_tags = collector.convert_dcm2nii_CT(CT_dcm_dirpath="/dicom/ct", output_dirpath=out_dir)

    assert dicom_tags["Modality"] == "CT"
    # Converted a filtered subdir ("acq"), not the raw series dir, and symlinked exactly the
    # dominant-acquisition files into it.
    assert str(captured["conv_dcm_dirpath"]).endswith("acq")
    assert symlink.call_count == len(dominant_files)
    linked_targets = {str(call.args[0]) for call in symlink.call_args_list}
    assert linked_targets == set(dominant_files)


def test_affine_repair_failure_is_logged_but_does_not_abort_conversion(mocker, collector, tmp_path, caplog):
    mocker.patch("musiq.series_selection.select_dominant_ct_acquisition", return_value=None)

    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        _write_nii_and_sidecar(output_folder)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    mocker.patch("musiq.series_selection.repair_slice_direction_from_dicom", side_effect=RuntimeError("boom"))

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with caplog.at_level(logging.ERROR):
        dicom_tags = collector.convert_dcm2nii_CT(CT_dcm_dirpath="/dicom/ct", output_dirpath=out_dir)

    assert dicom_tags["Modality"] == "CT"
    assert (out_dir / "CT.nii.gz").is_file()
    assert "CT affine sanity-check failed" in caplog.text


def test_sidecar_lookup_falls_back_to_any_json_when_stem_does_not_match(mocker, collector, tmp_path):
    """dcm2niix's Eq_1 (gantry-tilt-corrected) output sometimes shares the original's sidecar
    instead of getting its own matching-stem one; the code falls back to any json in the tmp
    conversion dir when the exact-stem sidecar is missing."""
    mocker.patch("musiq.series_selection.select_dominant_ct_acquisition", return_value=None)

    def fake_run_dcm2niix(input_folder, output_folder, merge=False):
        # nii named "series_1", sidecar named "series" (mismatched stem).
        data = np.zeros((4, 4, 4), dtype=np.int16)
        nib.save(nib.Nifti1Image(data, np.eye(4)), f"{output_folder}/series_1.nii.gz")
        with open(f"{output_folder}/series.json", "w") as f:
            json.dump({"Modality": "CT"}, f)

    mocker.patch("musiq.series_selection.run_dcm2niix", side_effect=fake_run_dcm2niix)
    mocker.patch("musiq.series_selection.repair_slice_direction_from_dicom", return_value=False)
    mocker.patch("musiq.series_selection.repair_slice_spacing_from_dicom", return_value=False)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    dicom_tags = collector.convert_dcm2nii_CT(CT_dcm_dirpath="/dicom/ct", output_dirpath=out_dir)

    assert dicom_tags["Modality"] == "CT"
