"""Unit tests for utils.list_dicom_files: picking candidate DICOM slice files out of a series
directory. Used directly by series_selection.py (convert_dcm2nii_PET) and internally by
resolve_pet_decay_reference/extract_dicom_data, so a wrong filter or sort order here would feed
the wrong "first slice" to every one of those. Exercised with plain touched files rather than a
full synthetic DICOM series, since only the filename-based filtering/sorting is under test.
"""

from musiq.utils import list_dicom_files


def test_filters_out_dicomdir_case_insensitively(tmp_path):
    (tmp_path / "slice_1.dcm").touch()
    (tmp_path / "DICOMDIR").touch()
    (tmp_path / "dicomdir").touch()

    result = [f.name for f in list_dicom_files(tmp_path)]

    assert result == ["slice_1.dcm"]


def test_filters_out_known_non_dicom_suffixes(tmp_path):
    (tmp_path / "slice_1.dcm").touch()
    for name in ("sidecar.json", "readme.txt", "archive.zip", "notes.xml", "install.exe"):
        (tmp_path / name).touch()

    result = [f.name for f in list_dicom_files(tmp_path)]

    assert result == ["slice_1.dcm"]


def test_filters_out_ds_store(tmp_path):
    (tmp_path / "slice_1.dcm").touch()
    (tmp_path / ".DS_Store").touch()

    result = [f.name for f in list_dicom_files(tmp_path)]

    assert result == ["slice_1.dcm"]


def test_ignores_subdirectories(tmp_path):
    (tmp_path / "slice_1.dcm").touch()
    (tmp_path / "subdir").mkdir()

    result = [f.name for f in list_dicom_files(tmp_path)]

    assert result == ["slice_1.dcm"]


def test_sorts_naturally_not_lexicographically(tmp_path):
    """Lexicographic sort would put slice_10 before slice_2; natural_key must not."""
    for name in ("slice_2.dcm", "slice_10.dcm", "slice_1.dcm"):
        (tmp_path / name).touch()

    result = [f.name for f in list_dicom_files(tmp_path)]

    assert result == ["slice_1.dcm", "slice_2.dcm", "slice_10.dcm"]


def test_empty_directory_returns_empty_list(tmp_path):
    assert list_dicom_files(tmp_path) == []
