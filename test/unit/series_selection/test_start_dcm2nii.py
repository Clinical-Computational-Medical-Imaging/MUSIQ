import logging
import os


def test_ct_conversion_builds_expected_paths_and_tags(mocker, tmp_path, collector):
    mocker.patch.object(collector, "convert_dcm2nii_CT", return_value={"SeriesDescription": "knochen ct"})
    out_dirpath = tmp_path / "out"

    flag, paths_and_tags = collector.start_dcm2nii(
        modality="CT", dicom_input_dirpath="/dicom/ct", out_dirpath=out_dirpath
    )

    assert flag is False
    entry = paths_and_tags["knochen ct"]
    assert entry["CTPath"] == os.path.join(out_dirpath, "CT.nii.gz")
    assert entry["InputDirPath"] == "/dicom/ct"
    assert "SUVPath" not in entry
    assert entry["DICOM"]["SeriesDescription"] == "knochen ct"
    assert out_dirpath.exists()


def test_pt_conversion_includes_suv_path(mocker, tmp_path, collector):
    mocker.patch.object(collector, "convert_dcm2nii_PET", return_value={"SeriesDescription": "wb ctac"})
    out_dirpath = tmp_path / "out"

    flag, paths_and_tags = collector.start_dcm2nii(
        modality="PT", dicom_input_dirpath="/dicom/pt", out_dirpath=out_dirpath
    )

    assert flag is False
    entry = paths_and_tags["wb ctac"]
    assert entry["PTPath"] == os.path.join(out_dirpath, "PET.nii.gz")
    assert entry["SUVPath"] == os.path.join(out_dirpath, "SUV.nii.gz")


def test_mr_conversion_has_no_suv_path(mocker, tmp_path, collector):
    mocker.patch.object(
        collector, "convert_dcm2nii_MR", return_value=("/out/t1_tse.nii.gz", {"SeriesDescription": "t1 tse"})
    )
    out_dirpath = tmp_path / "out"

    flag, paths_and_tags = collector.start_dcm2nii(
        modality="MR", dicom_input_dirpath="/dicom/mr", out_dirpath=out_dirpath, dynamic_sibling_dirs=["/a", "/b"]
    )

    assert flag is False
    entry = paths_and_tags["t1 tse"]
    assert entry["MRPath"] == "/out/t1_tse.nii.gz"
    assert "SUVPath" not in entry


def test_missing_series_description_falls_back_to_random_key(mocker, tmp_path, collector):
    mocker.patch.object(collector, "convert_dcm2nii_CT", return_value={})
    mocker.patch("musiq.series_selection.random.choices", return_value=list("ABCDE"))
    out_dirpath = tmp_path / "out"

    flag, paths_and_tags = collector.start_dcm2nii(
        modality="CT", dicom_input_dirpath="/dicom/ct", out_dirpath=out_dirpath
    )

    assert flag is False
    assert list(paths_and_tags.keys()) == ["Missing_SeriesDesc_ABCDE"]


def test_exception_during_conversion_returns_flag_and_empty_dict(caplog, mocker, tmp_path, collector):
    mocker.patch.object(collector, "convert_dcm2nii_CT", side_effect=Exception("boom"))
    out_dirpath = tmp_path / "out"

    with caplog.at_level(logging.ERROR):
        flag, paths_and_tags = collector.start_dcm2nii(
            modality="CT", dicom_input_dirpath="/dicom/ct", out_dirpath=out_dirpath
        )

    assert flag is True
    assert paths_and_tags == {}
    assert "Error processing CT series" in caplog.text
