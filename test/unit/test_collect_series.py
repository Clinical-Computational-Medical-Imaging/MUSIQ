import logging
import pathlib as plb
import sys
from unittest.mock import MagicMock

import pytest

sys.modules["SimpleITK"] = MagicMock()
sys.modules["gdcm"] = MagicMock()

from musiq.series_selection import SeriesSelection  # noqa: E402


@pytest.mark.parametrize(
    "patient_id, study_path_parts, series_dir_names",
    [
        ("P001", ("10000000", "10000001"), {"100000E0", "100001AA"}),
        ("P002", ("1000031B", "1000031C"), {"100004A9"}),
    ],
)
def test_collect_series_dir_path(
    mocker,
    dummy_dicom_dataset,
    tmp_input_output,
    get_series_keywords,
    patient_id,
    study_path_parts,
    series_dir_names,
):
    input_dir, output_dir = tmp_input_output

    def dummy_dcmread(file_path, *args, **kwargs):
        if "P001" in str(file_path):
            return dummy_dicom_dataset["P001"]
        elif "P002" in str(file_path):
            return dummy_dicom_dataset["P002"]
        else:
            raise FileNotFoundError("Unexpectet file")

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=dummy_dcmread)

    collector = SeriesSelection(
        input_dirpath=input_dir, output_dirpath=output_dir, series_keywords=get_series_keywords
    )
    grouped = collector.collect_series()

    ds = dummy_dicom_dataset[patient_id]
    series_list = grouped[(patient_id, ds.StudyDate)]

    expect_study_path = input_dir.joinpath(patient_id, *study_path_parts)
    expect_series_paths = {expect_study_path / name for name in series_dir_names}
    expect_patient_path = input_dir / patient_id

    assert {s["SeriesPath"] for s in series_list} == expect_series_paths
    for series in series_list:
        assert series["StudyPath"] == expect_study_path
        assert series["PatientPath"] == expect_patient_path
        assert series["Manufacturer"] == ds.Manufacturer
        assert series["Modality"] == ds.Modality
        assert series["PatientID"] == ds.PatientID
        assert series["SeriesDescription"] == ds.SeriesDescription
        assert series["StudyDate"] == ds.StudyDate
        assert series["StudyDescription"] == ds.StudyDescription

    all_series_paths = {s["SeriesPath"] for sl in grouped.values() for s in sl}
    assert (input_dir / "P001") not in all_series_paths
    assert len(grouped) == 2


def test_file_not_found_error(get_series_keywords):
    """
    Tests for wrong path and checks if the FileNotFoundError is thrown.
    """
    collector = SeriesSelection(
        input_dirpath="invalid/test/input/path",
        output_dirpath="invalid/test/output/path",
        series_keywords=get_series_keywords,
    )

    with pytest.raises(FileNotFoundError):
        collector.collect_series()


def test_log_error_dcmread_fail(caplog, mocker, tmp_input_output, get_series_keywords):
    input_dir, output_dir = tmp_input_output

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=Exception("Thrown exception"))

    collector = SeriesSelection(input_dirpath=input_dir, output_dirpath=output_dir, series_keywords=get_series_keywords)

    with caplog.at_level(logging.ERROR):
        grouped = collector.collect_series()

    assert grouped == {}
    assert "Failed to read DICOM" in caplog.text
    assert "Thrown exception" in caplog.text


def _make_single_series_input(tmp_path, patient_id="P999", study_dir_name="study1", series_dir_name="series1"):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    series_dir = input_dir / patient_id / study_dir_name / series_dir_name
    series_dir.mkdir(parents=True)
    (series_dir / "file1").touch()
    return input_dir, output_dir


@pytest.mark.parametrize("modality", ["SC", "OT", "PR"])
def test_collect_series_skips_unsupported_modality(mocker, make_dummy_ds, tmp_path, get_series_keywords, modality):
    """Only CT, PT and MR are supported; anything else must be silently skipped."""
    input_dir, output_dir = _make_single_series_input(tmp_path)

    ds = make_dummy_ds(patient_id="P999", modality=modality)
    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=ds)

    collector = SeriesSelection(input_dirpath=input_dir, output_dirpath=output_dir, series_keywords=get_series_keywords)
    grouped = collector.collect_series()

    assert grouped == {}


@pytest.mark.parametrize("missing_field", ["patient_id", "study_date", "modality"])
def test_collect_series_skips_missing_required_fields(
    mocker, make_dummy_ds, tmp_path, get_series_keywords, missing_field
):
    """A series missing PatientID, StudyDate or Modality can't be grouped and must be skipped."""
    input_dir, output_dir = _make_single_series_input(tmp_path)

    ds_kwargs = {"patient_id": "P999", "study_date": "20240101", "modality": "CT"}
    ds_kwargs[missing_field] = None
    ds = make_dummy_ds(**ds_kwargs)
    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=ds)

    collector = SeriesSelection(input_dirpath=input_dir, output_dirpath=output_dir, series_keywords=get_series_keywords)
    grouped = collector.collect_series()

    assert grouped == {}


def test_collect_series_skips_already_processed_ct_pt(caplog, mocker, make_dummy_ds, tmp_path, get_series_keywords):
    """CT/PT series are skipped once CT/PET/SUV NIfTIs and patient_info.json already exist."""
    patient_id, study_date = "P999", "20240101"
    input_dir, output_dir = _make_single_series_input(tmp_path, patient_id=patient_id)

    ds = make_dummy_ds(patient_id=patient_id, study_date=study_date, modality="CT")
    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=ds)

    study_out_dir = output_dir / patient_id / study_date
    study_out_dir.mkdir(parents=True)
    (study_out_dir / "CT.nii.gz").touch()
    (study_out_dir / "PET.nii.gz").touch()
    (study_out_dir / "SUV.nii.gz").touch()
    (output_dir / patient_id / "patient_info.json").touch()

    collector = SeriesSelection(input_dirpath=input_dir, output_dirpath=output_dir, series_keywords=get_series_keywords)

    with caplog.at_level(logging.INFO):
        grouped = collector.collect_series()

    assert grouped == {}
    assert "already exist" in caplog.text


def test_collect_series_does_not_skip_when_patient_info_missing(mocker, make_dummy_ds, tmp_path, get_series_keywords):
    """Even with all NIfTIs present, a missing patient_info.json means the run never finished — don't skip."""
    patient_id, study_date = "P999", "20240101"
    input_dir, output_dir = _make_single_series_input(tmp_path, patient_id=patient_id)

    ds = make_dummy_ds(patient_id=patient_id, study_date=study_date, modality="CT")
    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=ds)

    study_out_dir = output_dir / patient_id / study_date
    study_out_dir.mkdir(parents=True)
    (study_out_dir / "CT.nii.gz").touch()
    (study_out_dir / "PET.nii.gz").touch()
    (study_out_dir / "SUV.nii.gz").touch()
    # no patient_info.json this time

    collector = SeriesSelection(input_dirpath=input_dir, output_dirpath=output_dir, series_keywords=get_series_keywords)
    grouped = collector.collect_series()

    assert (patient_id, study_date) in grouped


def test_collect_series_skips_existing_mr_nifti(mocker, make_dummy_ds, tmp_path, get_series_keywords):
    """MR series are skipped via mr_nifti_exists (matched by ProtocolName/SeriesDescription), not by CT/PET/SUV paths."""
    patient_id, study_date = "P999", "20240101"
    input_dir, output_dir = _make_single_series_input(tmp_path, patient_id=patient_id)

    ds = make_dummy_ds(patient_id=patient_id, study_date=study_date, modality="MR", series_description="t1 tse")
    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=ds)

    study_out_dir = output_dir / patient_id / study_date
    study_out_dir.mkdir(parents=True)
    (study_out_dir / "t1_tse_5.nii.gz").touch()
    (output_dir / patient_id / "patient_info.json").touch()

    collector = SeriesSelection(input_dirpath=input_dir, output_dirpath=output_dir, series_keywords=get_series_keywords)
    grouped = collector.collect_series()

    assert grouped == {}


@pytest.mark.parametrize(
    "study_date, dir_name, expected",
    [
        ("20240101", "1.2.840.113619.255564.20230406132841", "20240101"),  # real date -> passthrough, token ignored
        ("00000000", "1.2.840.113619.255564.20230406132841", "20230406132841"),  # placeholder -> use dir-name token
        ("", "1.2.840.113619.255564.20230406132841", "20230406132841"),  # empty StudyDate counts as placeholder too
        ("19000101", "series_without_token", "19000101"),  # placeholder, no valid token -> unchanged fallback
    ],
)
def test_study_key_placeholder_fallback(get_series_keywords, study_date, dir_name, expected):
    collector = SeriesSelection(input_dirpath=".", output_dirpath=".", series_keywords=get_series_keywords)
    series_dir = plb.Path("/some/input") / dir_name

    assert collector._study_key(series_dir, study_date) == expected


def test_study_key_warns_when_no_token_found(caplog, get_series_keywords):
    collector = SeriesSelection(input_dirpath=".", output_dirpath=".", series_keywords=get_series_keywords)
    series_dir = plb.Path("/some/input/series_without_token")

    with caplog.at_level(logging.WARNING):
        result = collector._study_key(series_dir, "00000000")

    assert result == "00000000"
    assert "no datetime token found" in caplog.text
