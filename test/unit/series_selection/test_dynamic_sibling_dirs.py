import pathlib as plb


class DummyDicom:
    def __init__(self, acquisition_time=None, image_type=None):
        self.AcquisitionTime = acquisition_time
        self.ImageType = image_type or []


def _make_group(tmp_path, count, desc="dyn scan", modality="MR"):
    entries = []
    for i in range(count):
        series_dir = tmp_path / f"series{i}"
        series_dir.mkdir()
        (series_dir / "file1.dcm").touch()
        entries.append({"Modality": modality, "SeriesDescription": desc, "SeriesPath": series_dir})
    return entries


def test_non_mr_entry_returns_none(collector):
    study_info = [{"Modality": "CT", "SeriesDescription": "dyn ct"}]
    entry = study_info[0]

    assert collector._dynamic_sibling_dirs(study_info, entry) is None


def test_fewer_than_three_siblings_returns_none(collector, tmp_path):
    study_info = _make_group(tmp_path, count=2)
    entry = study_info[0]

    assert collector._dynamic_sibling_dirs(study_info, entry) is None


def test_dynamic_marker_in_description_and_distinct_times_returns_sorted_siblings(mocker, collector, tmp_path):
    study_info = _make_group(tmp_path, count=3, desc="dyn scan")
    entry = study_info[0]

    times = ["120000", "100000", "110000"]

    def dcmread(path, *args, **kwargs):
        idx = int(plb.Path(path).parent.name.replace("series", ""))
        return DummyDicom(acquisition_time=times[idx])

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=dcmread)

    result = collector._dynamic_sibling_dirs(study_info, entry)

    assert result == [study_info[1]["SeriesPath"], study_info[2]["SeriesPath"], study_info[0]["SeriesPath"]]


def test_dynamic_marker_from_image_type_when_description_is_plain(mocker, collector, tmp_path):
    study_info = _make_group(tmp_path, count=3, desc="plain series")
    entry = study_info[0]

    def dcmread(path, *args, **kwargs):
        idx = int(plb.Path(path).parent.name.replace("series", ""))
        return DummyDicom(acquisition_time=f"10000{idx}", image_type=["ORIGINAL", "DYNAMIC"])

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=dcmread)

    result = collector._dynamic_sibling_dirs(study_info, entry)

    assert result is not None
    assert len(result) == 3


def test_no_dynamic_marker_returns_none(mocker, collector, tmp_path):
    study_info = _make_group(tmp_path, count=3, desc="plain series")
    entry = study_info[0]

    def dcmread(path, *args, **kwargs):
        idx = int(plb.Path(path).parent.name.replace("series", ""))
        return DummyDicom(acquisition_time=f"10000{idx}")

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=dcmread)

    assert collector._dynamic_sibling_dirs(study_info, entry) is None


def test_duplicate_acquisition_times_returns_none(mocker, collector, tmp_path):
    study_info = _make_group(tmp_path, count=3, desc="dyn scan")
    entry = study_info[0]

    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=DummyDicom(acquisition_time="100000"))

    assert collector._dynamic_sibling_dirs(study_info, entry) is None


def test_missing_acquisition_time_returns_none(mocker, collector, tmp_path):
    study_info = _make_group(tmp_path, count=3, desc="dyn scan")
    entry = study_info[0]

    def dcmread(path, *args, **kwargs):
        idx = int(plb.Path(path).parent.name.replace("series", ""))
        return DummyDicom(acquisition_time=None if idx == 0 else f"10000{idx}")

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=dcmread)

    assert collector._dynamic_sibling_dirs(study_info, entry) is None


def test_empty_series_directory_returns_none(mocker, collector, tmp_path):
    study_info = _make_group(tmp_path, count=3, desc="dyn scan")
    # Empty out one of the series directories (not the first, so the loop actually reaches it
    # instead of failing earlier on a real dcmread against the dummy touched file).
    (study_info[1]["SeriesPath"] / "file1.dcm").unlink()
    entry = study_info[0]

    mocker.patch("musiq.series_selection.pydicom.dcmread", return_value=DummyDicom(acquisition_time="100000"))

    assert collector._dynamic_sibling_dirs(study_info, entry) is None


def test_dcmread_failure_returns_none(mocker, collector, tmp_path):
    study_info = _make_group(tmp_path, count=3, desc="dyn scan")
    entry = study_info[0]

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=Exception("broken dicom"))

    assert collector._dynamic_sibling_dirs(study_info, entry) is None
