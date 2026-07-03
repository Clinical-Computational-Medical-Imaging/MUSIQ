# from pathlib import Path
import pytest


@pytest.fixture()
def dummy_dicom_dataset():
    class DummyDS:
        def __init__(self, patientID, studyDate, modality, seriesDescription, studyDescription, manufacturer):
            self.PatientID = patientID
            self.StudyDate = studyDate
            self.Modality = modality
            self.SeriesDescription = seriesDescription
            self.StudyDescription = studyDescription
            self.Manufacturer = manufacturer

    return {
        "P001": DummyDS("P001", "20240101", "CT", "series_desc", "study_desc", "manufacturer"),
        "P002": DummyDS("P002", "20240102", "MR", "series_desc2", "study_desc2", "manufacturer2"),
    }


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
    (input_dir / "P001" / "test.zip").touch()
    (input_dir / "P001" / "test.inf").touch()
    (input_dir / "P001" / "test.jar").touch()
    (input_dir / "P001" / "test.icns").touch()
    (input_dir / "P001" / "test.info").touch()
    (input_dir / "P001" / "test.exe").touch()
    (input_dir / "P001" / "test.pdf").touch()
    (input_dir / "P001" / "test.txt").touch()
    (input_dir / "P001" / "test.ini").touch()
    (input_dir / "P001" / "test.xml").touch()
    (input_dir / "P001" / "test.bmp").touch()
    (input_dir / "P001" / "test.sh").touch()
    (input_dir / "P001" / "DeepUnity Media Viewer Mac").touch()
    (input_dir / "P001" / ".DS_Store").touch()

    series_dir = input_dir / "P001" / "10000000" / "10000001" / "100001AA"
    series_dir.mkdir(parents=True)
    (series_dir / "100001AB").touch()
    (series_dir / "100001AC").touch()

    series_dir = input_dir / "P002" / "1000031B" / "1000031C" / "100004A9"
    series_dir.mkdir(parents=True)
    (series_dir / "100004AA").touch()

    return input_dir, output_dir


@pytest.fixture()
def get_series_keywords():
    keys = {
        "SERIES_KEYWORDS": {
            "CT": {"PRIMARY": ["knochen", "i30f"], "SECONDARY": ["weichteil", "i70f"], "EXCLUSION": []},
            "PT": {
                "PRIMARY": ["pet gk ctac", "qc fx", "wb_ctac", "wb ctac", "tep tardif ac"],
                "SECONDARY": [],
                "EXCLUSION": ["nac motion free", "pet exam report", "mip", "nac"],
            },
            "MR": {
                "PRIMARY": [
                    "t1 tse",
                    "t2 tse",
                    "t2w_mvxd_sag",
                    "t2w_tse_sag",
                    "dyn",
                    "dce mi fov",
                    "dwi",
                    "1400",
                    "adc",
                    "tracew",
                ],
                "SECONDARY": [],
                "EXCLUSION": ["carebolus", "localizer", "survey", "ds", "dixon", "haste", "t2-space-coronar"],
            },
        }
    }

    return keys
