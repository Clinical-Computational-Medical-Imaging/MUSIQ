import shutil

import pytest

from musiq.series_selection import SeriesSelection

from ._dicom_builder import write_dicom_series


class DummyDS:
    def __init__(
        self,
        patient_id,
        study_date,
        modality,
        series_description,
        study_description,
        manufacturer,
        protocol_name=None,
    ):
        self.PatientID = patient_id
        self.StudyDate = study_date
        self.Modality = modality
        self.SeriesDescription = series_description
        self.StudyDescription = study_description
        self.Manufacturer = manufacturer
        self.ProtocolName = protocol_name


@pytest.fixture()
def dummy_dicom_dataset(make_dummy_ds):
    """A CT dataset for P001 and an MR dataset for P002, keyed by patient ID."""
    return {
        "P001": make_dummy_ds(
            patient_id="P001",
            study_date="20240101",
            modality="CT",
            series_description="series_desc",
            study_description="study_desc",
            manufacturer="manufacturer",
        ),
        "P002": make_dummy_ds(
            patient_id="P002",
            study_date="20240102",
            modality="MR",
            series_description="series_desc2",
            study_description="study_desc2",
            manufacturer="manufacturer2",
        ),
    }


@pytest.fixture()
def make_dummy_ds():
    """Factory for building a DummyDS with sensible defaults, overridable per test."""

    def _make(
        patient_id="P999",
        study_date="20240101",
        modality="CT",
        series_description="series_desc",
        study_description="study_desc",
        manufacturer="manufacturer",
        protocol_name=None,
    ):
        return DummyDS(
            patient_id, study_date, modality, series_description, study_description, manufacturer, protocol_name
        )

    return _make


@pytest.fixture()
def tmp_input_output(tmp_path):
    """A DICOM input tree for P001/P002 plus an empty output dir.

    P001's patient directory also holds one file per extension/name that `collect_series` is
    expected to skip (see the ignore list in `SeriesSelection.collect_series`), so tests can
    assert that none of them are mistaken for a series.
    """
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
def series_keywords():
    """Default PRIMARY/SECONDARY/EXCLUSION keywords per modality, mirroring config.yaml."""
    keys = {
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

    return keys


@pytest.fixture()
def collector(series_keywords):
    """A SeriesSelection instance with placeholder dirpaths, for tests that don't touch the filesystem."""
    return SeriesSelection(input_dirpath=".", output_dirpath=".", series_keywords=series_keywords)


@pytest.fixture()
def make_collector(series_keywords):
    """Factory for a SeriesSelection instance with overridable dirpaths/keywords, e.g. a real tmp_path."""
    default_series_keywords = series_keywords

    def _make(output_dirpath=".", input_dirpath=".", series_keywords=None):
        return SeriesSelection(
            input_dirpath=input_dirpath,
            output_dirpath=output_dirpath,
            series_keywords=series_keywords or default_series_keywords,
        )

    return _make


@pytest.fixture(scope="session")
def dcm2niix_available():
    """Skip a test when the real dcm2niix binary isn't on PATH.

    Installed via the ``dcm2niix`` PyPI package (pinned in pyproject.toml), which ships a
    console-script wrapper on Windows/Linux/macOS. Guarded rather than assumed so this test
    suite still runs (minus the real-conversion tests) in an environment without it.
    """
    if shutil.which("dcm2niix") is None:
        pytest.skip("dcm2niix binary not found on PATH; install the 'dcm2niix' package to run this test.")


@pytest.fixture()
def dicom_series_factory(tmp_path):
    """Factory writing a minimal-but-valid synthetic DICOM series to a tmp subdirectory.

    Wraps :func:`write_dicom_series`, defaulting ``out_dir`` to a fresh ``tmp_path`` subfolder
    per call so independent series don't collide.
    """
    counter = {"n": 0}

    def _make(modality, subdir=None, **kwargs):
        counter["n"] += 1
        out_dir = tmp_path / (subdir or f"series_{counter['n']}")
        return write_dicom_series(out_dir, modality, **kwargs)

    return _make
