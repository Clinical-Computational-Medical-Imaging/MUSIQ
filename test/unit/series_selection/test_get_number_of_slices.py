def test_counts_only_readable_dicom_files(mocker, tmp_path, collector):
    (tmp_path / "good1.dcm").touch()
    (tmp_path / "good2.dcm").touch()
    (tmp_path / "bad.dcm").touch()

    def dcmread(path, *args, **kwargs):
        if "bad" in str(path):
            raise Exception("corrupt")
        return object()

    mocker.patch("musiq.series_selection.pydicom.dcmread", side_effect=dcmread)

    assert collector.get_number_of_slices(tmp_path) == 2


def test_empty_directory_returns_zero(collector, tmp_path):
    assert collector.get_number_of_slices(tmp_path) == 0
