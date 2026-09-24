import csv


def _data_dict(patient_id, study_date, modality, series_desc, series_info):
    return {
        "PatientID": patient_id,
        "Studies": {study_date: {"Modalities": {modality: [{series_desc: series_info}]}}},
    }


def _read_rows(csv_path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def test_no_errors_means_no_file_written(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    data_dict = _data_dict("P1", "20240101", "CT", "knochen ct", {"CTPath": None})

    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=[])

    assert not csv_path.exists()


def test_missing_nifti_file_is_reported(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    missing_path = str(tmp_path / "does_not_exist.nii.gz")
    data_dict = _data_dict("P1", "20240101", "CT", "knochen ct", {"CTPath": missing_path})

    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=[])

    rows = _read_rows(csv_path)
    reasons = [r["Reason"] for r in rows]
    assert "NIfTI file does not exist" in reasons


def test_existing_nifti_file_with_primary_match_is_not_reported(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    nii_path = tmp_path / "CT.nii.gz"
    nii_path.touch()
    data_dict = _data_dict("P1", "20240101", "CT", "knochen ct", {"CTPath": str(nii_path)})

    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=[])

    assert not csv_path.exists()


def test_no_keyword_match_is_reported_without_secondary_note(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    data_dict = _data_dict("P1", "20240101", "CT", "unrelated series", {"CTPath": None})

    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=[])

    rows = _read_rows(csv_path)
    reason_row = next(r for r in rows if "No primary keyword match" in r["Reason"])
    assert "(but secondary keyword matched)" not in reason_row["Reason"]


def test_secondary_only_match_is_reported_with_secondary_note(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    data_dict = _data_dict("P1", "20240101", "CT", "ct weichteil", {"CTPath": None})

    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=[])

    rows = _read_rows(csv_path)
    reason_row = next(r for r in rows if "No primary keyword match" in r["Reason"])
    assert "(but secondary keyword matched)" in reason_row["Reason"]


def test_user_flag_adds_error_for_every_series(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    nii_path = tmp_path / "CT.nii.gz"
    nii_path.touch()
    data_dict = _data_dict("P1", "20240101", "CT", "knochen ct", {"CTPath": str(nii_path)})

    collector.validate_output(data_dict, str(csv_path), user_flag=True, conversion_flags=[])

    rows = _read_rows(csv_path)
    assert any(r["Reason"] == "Study flagged by user" for r in rows)


def test_conversion_flags_are_reported(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    data_dict = {"PatientID": "P1", "Studies": {}}
    conversion_flags = [["P1", "20240101", "broken series", "/some/path"]]

    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=conversion_flags)

    rows = _read_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["Reason"] == "Error while converting the DICOM series to NIfTI"
    assert rows[0]["SeriesDesc"] == "broken series"


def test_conversion_flag_entries_with_too_few_fields_are_skipped(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    data_dict = {"PatientID": "P1", "Studies": {}}
    conversion_flags = [["P1", "20240101"]]

    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=conversion_flags)

    assert not csv_path.exists()


def test_header_written_only_once_across_multiple_calls(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    data_dict = _data_dict("P1", "20240101", "CT", "unrelated series", {"CTPath": None})

    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=[])
    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=[])

    with open(csv_path) as f:
        lines = f.readlines()

    header_lines = [line for line in lines if line.startswith("patient_id")]
    assert len(header_lines) == 1


def test_non_dict_series_entries_are_skipped_without_error(tmp_path, collector):
    csv_path = tmp_path / "validation_results.csv"
    data_dict = {
        "PatientID": "P1",
        "Studies": {"20240101": {"Modalities": {"CT": [{}, None, {"knochen ct": {"CTPath": None}}]}}},
    }

    collector.validate_output(data_dict, str(csv_path), user_flag=False, conversion_flags=[])

    assert not csv_path.exists()
