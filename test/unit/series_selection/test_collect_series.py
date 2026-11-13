import json
import logging
import pathlib as plb

import pytest


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
    make_collector,
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
            raise FileNotFoundError("Unexpected file")

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=dummy_dcmread)

    collector = make_collector(input_dirpath=input_dir, output_dirpath=output_dir)
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


def test_file_not_found_error(make_collector):
    """Tests for wrong path and checks if the FileNotFoundError is thrown."""
    collector = make_collector(input_dirpath="invalid/test/input/path", output_dirpath="invalid/test/output/path")

    with pytest.raises(FileNotFoundError):
        collector.collect_series()


def test_log_error_dcmread_fail(caplog, mocker, tmp_input_output, make_collector):
    input_dir, output_dir = tmp_input_output

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=Exception("Thrown exception"))

    collector = make_collector(input_dirpath=input_dir, output_dirpath=output_dir)

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
def test_collect_series_skips_unsupported_modality(mocker, make_dummy_ds, tmp_path, make_collector, modality):
    """Only CT, PT and MR are supported; anything else must be silently skipped."""
    input_dir, output_dir = _make_single_series_input(tmp_path)

    ds = make_dummy_ds(patient_id="P999", modality=modality)
    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=ds)

    collector = make_collector(input_dirpath=input_dir, output_dirpath=output_dir)
    grouped = collector.collect_series()

    assert grouped == {}


@pytest.mark.parametrize("missing_field", ["patient_id", "study_date", "modality"])
def test_collect_series_skips_missing_required_fields(mocker, make_dummy_ds, tmp_path, make_collector, missing_field):
    """A series missing PatientID, StudyDate or Modality can't be grouped and must be skipped."""
    input_dir, output_dir = _make_single_series_input(tmp_path)

    ds_kwargs = {"patient_id": "P999", "study_date": "20240101", "modality": "CT"}
    ds_kwargs[missing_field] = None
    ds = make_dummy_ds(**ds_kwargs)
    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=ds)

    collector = make_collector(input_dirpath=input_dir, output_dirpath=output_dir)
    grouped = collector.collect_series()

    assert grouped == {}


def test_collect_series_skips_already_processed_ct_pt(caplog, mocker, make_dummy_ds, tmp_path, make_collector):
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

    collector = make_collector(input_dirpath=input_dir, output_dirpath=output_dir)

    with caplog.at_level(logging.INFO):
        grouped = collector.collect_series()

    assert grouped == {}
    assert "already exist" in caplog.text


def test_collect_series_does_not_skip_when_patient_info_missing(mocker, make_dummy_ds, tmp_path, make_collector):
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

    collector = make_collector(input_dirpath=input_dir, output_dirpath=output_dir)
    grouped = collector.collect_series()

    assert (patient_id, study_date) in grouped


def test_collect_series_skips_existing_mr_nifti(mocker, make_dummy_ds, tmp_path, make_collector):
    """MR series are skipped via mr_nifti_exists (ProtocolName/SeriesDescription match), not the CT/PET/SUV paths."""
    patient_id, study_date = "P999", "20240101"
    input_dir, output_dir = _make_single_series_input(tmp_path, patient_id=patient_id)

    ds = make_dummy_ds(patient_id=patient_id, study_date=study_date, modality="MR", series_description="t1 tse")
    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=ds)

    study_out_dir = output_dir / patient_id / study_date
    study_out_dir.mkdir(parents=True)
    (study_out_dir / "t1_tse_5.nii.gz").touch()
    (output_dir / patient_id / "patient_info.json").touch()

    collector = make_collector(input_dirpath=input_dir, output_dirpath=output_dir)
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
def test_study_key_placeholder_fallback(collector, study_date, dir_name, expected):
    series_dir = plb.Path("/some/input") / dir_name

    assert collector._study_key(series_dir, study_date, "/nonexistent/patient_info.json") == expected


def test_study_key_warns_when_no_token_found(caplog, collector):
    series_dir = plb.Path("/some/input/series_without_token")

    with caplog.at_level(logging.WARNING):
        result = collector._study_key(series_dir, "00000000", "/nonexistent/patient_info.json")

    assert result == "00000000"
    assert "no datetime token found" in caplog.text


def _write_patient_info(path, studies):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"Studies": studies}))


def test_study_key_reuses_recorded_key_over_dirname_token(tmp_path, collector):
    """A previous run's recorded study key wins over the datetime-token fallback, so a series
    already converted into a renamed/recovered study folder maps back to it instead of a new one."""
    patient_info_path = tmp_path / "patient_info.json"
    series_dir = plb.Path("/raw/P001/study1/1.2.840.113619.255564.20230406132841")
    _write_patient_info(
        patient_info_path,
        {"recovered_study_date": {"Modalities": {"CT": [{"knochen ct": {"InputDirPath": str(series_dir)}}]}}},
    )

    result = collector._study_key(series_dir, "00000000", patient_info_path)

    assert result == "recovered_study_date"


def test_recorded_study_key_returns_none_when_patient_info_missing(tmp_path, collector):
    missing_path = tmp_path / "patient_info.json"

    assert collector._recorded_study_key(missing_path, plb.Path("/raw/series")) is None


def test_recorded_study_key_returns_none_for_dangling_series(tmp_path, collector):
    """A series not present in any recorded study (e.g. new, or never converted) falls through
    to the datetime-token fallback rather than reusing an unrelated recorded key."""
    patient_info_path = tmp_path / "patient_info.json"
    _write_patient_info(
        patient_info_path,
        {"20230101": {"Modalities": {"CT": [{"other": {"InputDirPath": "/raw/other_series"}}]}}},
    )

    result = collector._recorded_study_key(patient_info_path, plb.Path("/raw/new_series"))

    assert result is None


def test_recorded_study_key_matches_on_basename_regardless_of_trailing_slash(tmp_path, collector):
    patient_info_path = tmp_path / "patient_info.json"
    _write_patient_info(
        patient_info_path,
        {"20230101": {"Modalities": {"CT": [{"a": {"InputDirPath": "/raw/P001/study1/series1/"}}]}}},
    )

    result = collector._recorded_study_key(patient_info_path, plb.Path("/different/root/series1"))

    assert result == "20230101"


def test_recorded_study_key_warns_and_returns_none_on_corrupt_json(caplog, tmp_path, collector):
    patient_info_path = tmp_path / "patient_info.json"
    patient_info_path.write_text("{not valid json")

    with caplog.at_level(logging.WARNING):
        result = collector._recorded_study_key(patient_info_path, plb.Path("/raw/series"))

    assert result is None
    assert "Could not read" in caplog.text


def test_recorded_study_key_is_cached_per_patient_info_path(tmp_path, collector):
    """Once read, the patient_info.json -> study-key table is cached on the collector instance,
    so a later mutation of the file on disk must not change results within the same run."""
    patient_info_path = tmp_path / "patient_info.json"
    series_dir = plb.Path("/raw/seriesX")
    _write_patient_info(
        patient_info_path,
        {"20230101": {"Modalities": {"CT": [{"a": {"InputDirPath": str(series_dir)}}]}}},
    )

    first = collector._recorded_study_key(patient_info_path, series_dir)
    assert first == "20230101"

    _write_patient_info(
        patient_info_path,
        {"20240101": {"Modalities": {"CT": [{"a": {"InputDirPath": str(series_dir)}}]}}},
    )

    second = collector._recorded_study_key(patient_info_path, series_dir)
    assert second == "20230101"
