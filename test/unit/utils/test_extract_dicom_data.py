"""Unit tests for utils.extract_dicom_data: pulling a fixed tag-name -> value map out of a
series' first DICOM file, used by series_selection.py to populate patient_info.json.

series_selection's own unit tests (test_handle_selected_series.py, test_interactive_selection.py)
mock ``extract_dicom_data`` out entirely, so its real behavior -- PersonName stringification,
PatientAge digit-stripping, the FDG radiopharmaceutical normalization, and the malformed-file
fallback. These build a minimal synthetic DICOM series (via
the shared ``dicom_series_factory`` fixture) to exercise it directly.
"""

import logging
import pathlib as plb

import pytest

from musiq.utils import extract_dicom_data

_TAGS = {
    "PatientName": ("0010", "0010"),
    "PatientAge": ("0010", "1010"),
    "Radiopharmaceutical": ("0018", "0031"),
    "NotPresent": ("0009", "0001"),
}


def _make_ct(dicom_series_factory, **extra_tags):
    return dicom_series_factory("CT", n_slices=1, rows=4, cols=4, extra_tags=extra_tags)


def test_person_name_is_converted_to_plain_string(dicom_series_factory):
    dcm_dir = _make_ct(dicom_series_factory)

    info = extract_dicom_data(plb.Path(dcm_dir), _TAGS)

    assert info["PatientName"] == "Test^Pat"
    assert isinstance(info["PatientName"], str)


@pytest.mark.parametrize(
    "raw_age,expected",
    [
        ("055Y", "55"),  # leading zero stripped
        ("007Y", "7"),
        ("000Y", ""),  # documented edge case: an all-zero age collapses to an empty string
    ],
)
def test_patient_age_keeps_only_digits_and_strips_leading_zeros(dicom_series_factory, raw_age, expected):
    dcm_dir = _make_ct(dicom_series_factory, PatientAge=raw_age)

    info = extract_dicom_data(plb.Path(dcm_dir), _TAGS)

    assert info["PatientAge"] == expected


@pytest.mark.parametrize(
    "raw_value",
    ["FDG", "Fluorodeoxyglucose", "FDG -- Fluorodeoxyglucose", "FDG -- fluorodeoxyglucose"],
)
def test_known_fdg_spellings_are_normalized(dicom_series_factory, raw_value):
    dcm_dir = _make_ct(dicom_series_factory, Radiopharmaceutical=raw_value)

    info = extract_dicom_data(plb.Path(dcm_dir), _TAGS)

    assert info["Radiopharmaceutical"] == "FDG"


def test_unrecognized_radiopharmaceutical_is_left_unchanged(dicom_series_factory):
    dcm_dir = _make_ct(dicom_series_factory, Radiopharmaceutical="Gallium-68 DOTATATE")

    info = extract_dicom_data(plb.Path(dcm_dir), _TAGS)

    assert info["Radiopharmaceutical"] == "Gallium-68 DOTATATE"


def test_tag_absent_from_dicom_is_absent_from_result_not_none(dicom_series_factory):
    dcm_dir = _make_ct(dicom_series_factory)

    info = extract_dicom_data(plb.Path(dcm_dir), _TAGS)

    assert "NotPresent" not in info


def test_empty_directory_returns_empty_dict(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert extract_dicom_data(empty_dir, _TAGS) == {}


def test_unreadable_file_logs_error_and_returns_empty_dict(tmp_path, caplog):
    (tmp_path / "not_a_dicom_file").write_bytes(b"this is not a DICOM file")

    with caplog.at_level(logging.ERROR):
        info = extract_dicom_data(tmp_path, _TAGS)

    assert info == {}
    assert "Error processing file" in caplog.text
