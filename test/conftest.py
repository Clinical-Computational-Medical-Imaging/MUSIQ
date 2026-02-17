import pytest
from pathlib import Path

@pytest.fixture()
def dummy_dicom_dataset():
        class DummyDS1:
                PatientID = "P001"
                StudyDate = "20240101"
                Modality = "CT"
                SeriesDescription = "series_desc"
                StudyDescription = "study_desc"
                Manufacturer = "manufacturer"
        """
        class DummyDS_1_mr:
                PatientID = "P001"
                StudyDate = "20240101"
                Modality = "CT"
                SeriesDescription = "series_desc"
                StudyDescription = "study_desc"
                Manufacturer = "manufacturer"
        """
        class DummyDS2:
                PatientID = "P002"
                StudyDate = "20240102"
                Modality = "MR"
                SeriesDescription = "series_desc2"
                StudyDescription = "study_desc2"
                Manufacturer = "manufacturer2"
        return {"P001": DummyDS1(), "P002": DummyDS2()} 

@pytest.fixture()
def tmp_input_output(tmp_path):
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"

        input_dir.mkdir()
        output_dir.mkdir()

        series_dir = input_dir / "P001" / "10000000" / "10000001" / "100000E0"
        series_dir.mkdir(parents=True)
        (series_dir / "100000E1").touch()
        (series_dir / "100000E2").touch()
        (input_dir / "P001" / "DICOMDIR").touch()
        (input_dir / "P001" / "test.DS_Store").touch()
        (input_dir / "P001" / "test.zip").touch()

        series_dir = input_dir / "P001" / "10000000" / "10000001" / "100001AA"
        series_dir.mkdir(parents=True)
        (series_dir / "100001AB").touch()
        (series_dir / "100001AC").touch()

        series_dir = input_dir / "P002" / "1000031B" / "1000031C" / "100004A9"
        series_dir.mkdir(parents=True)
        (series_dir / "100004AA").touch()
        (input_dir / "P002" / "test.info").touch()

        return input_dir, output_dir

@pytest.fixture()
def get_series_keywords():
        keys = {
        "SERIES_KEYWORDS":{ 
            "CT":{
                "PRIMARY": ["knochen", "i30f"],
                "SECONDARY": ["weichteil", "i70f"],
                "EXCLUSION": []},
            "PT":{
                "PRIMARY": ["pet gk ctac", "qc fx", "wb_ctac", "wb ctac", "tep tardif ac"],
                "SECONDARY": [],
                "EXCLUSION": ["nac motion free", "pet exam report", "mip", "nac"]},
            "MR":{
                "PRIMARY": ["t1 tse", "t2 tse", "t2w_mvxd_sag", "t2w_tse_sag", "dyn", "dce mi fov", "dwi", "1400", "adc", "tracew"],
                "SECONDARY": [],
                "EXCLUSION": ["carebolus", "localizer", "survey", "ds", "dixon", "haste", "t2-space-coronar"]
                }}}

        return keys
