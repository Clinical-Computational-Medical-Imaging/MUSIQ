"""Fixtures for real-DICOM integration tests against a locally downloaded TCIA cohort.

These tests exercise the real dcm2niix conversion path against genuine DICOM data (not
mocked), so they need the actual images on disk. Two ways to provide them:

1. Auto-download just the ~4 series these tests need, straight from the public TCIA REST API
   (no login, no Java NBIA Data Retriever), into a temp dir that is deleted again once the test
   session ends:

       pytest test/integration --download-integration-data

   Opt-in only: it hits an external network service and fetches real (de-identified) patient
   imaging data on every run, so it's never on by default.

2. Point at a manifest-1773751814915 cohort you already downloaded yourself (e.g. via the NBIA
   Data Retriever, using ``test/integration/manifest-1773751814915.tcia``):

       pytest test/integration --integration-data-dir "C:\\path\\to\\manifest-1773751814915"
       MUSIQ_INTEGRATION_DATA_DIR="/path/to/manifest-1773751814915" pytest test/integration

Tests in this directory skip automatically when none of the above is set.
"""

import io
import os
import pathlib as plb
import shutil
import urllib.request
import zipfile

import pytest

from musiq.series_selection import SeriesSelection

_TCIA_GET_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"

# The exact 4 series these tests need, identified by SeriesInstanceUID (see
# manifest-1773751814915.tcia, trimmed to just these). study_dir/series_dir only need to
# *contain* the substrings find_series_dir()'s tests glob for — they don't have to match TCIA's
# real (randomly-suffixed) folder names.
_DOWNLOAD_TARGETS = [
    {
        "patient_id": "TCGA-VP-A878",
        "study_dir": "Abdomen05CAP",
        "series_dir": "ChestAbdPel soft tissue",
        "uid": "1.3.6.1.4.1.14519.5.2.1.7777.4006.266822032522643745978907862000",
    },
    {
        "patient_id": "TCGA-VP-A878",
        "study_dir": "Abdomen05CAP",
        "series_dir": "ChestAbdPel LUNG",
        "uid": "1.3.6.1.4.1.14519.5.2.1.7777.4006.313444960025318623576650119009",
    },
    {
        "patient_id": "TCGA-EJ-5495",
        "study_dir": "MRI PELVIS",
        "series_dir": "T2 CORONAL",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6450.4006.236442968616827670504168629430",
    },
    {
        "patient_id": "TCGA-J4-A67O",
        "study_dir": "MRI PELVIS WWO C",
        "series_dir": "DYNAMIC SCAN",
        "uid": "1.3.6.1.4.1.14519.5.2.1.3983.4006.123592663440526734445424125161",
    },
]


def pytest_addoption(parser):
    parser.addoption(
        "--integration-data-dir",
        action="store",
        default=None,
        help="Path to the locally downloaded TCIA manifest-1773751814915 cohort root "
        "(the directory that contains TCGA-PRAD/). Falls back to the "
        "MUSIQ_INTEGRATION_DATA_DIR environment variable.",
    )
    parser.addoption(
        "--download-integration-data",
        action="store_true",
        default=False,
        help="Download just the ~4 series these tests need directly from the public TCIA REST "
        "API into a temp dir, and delete it again once the test session ends. Opt-in: hits an "
        "external network service and downloads real (de-identified) patient imaging data on "
        "every run. Takes priority over --integration-data-dir/MUSIQ_INTEGRATION_DATA_DIR.",
    )


def _download_series_zip(series_instance_uid: str, dest_dir: plb.Path) -> None:
    """Fetch one series as a zip from TCIA's public getImage endpoint and extract its DICOMs."""
    url = f"{_TCIA_GET_IMAGE_URL}?SeriesInstanceUID={series_instance_uid}"
    req = urllib.request.Request(url, headers={"User-Agent": "musiq-integration-tests"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        dcm_names = [n for n in zf.namelist() if n.lower().endswith(".dcm")]
        if not dcm_names:
            raise ValueError(f"No .dcm entries in TCIA response for series {series_instance_uid}")
        for name in dcm_names:
            dest_dir.joinpath(os.path.basename(name)).write_bytes(zf.read(name))


@pytest.fixture(scope="session")
def integration_data_dir(request, tmp_path_factory):
    if request.config.getoption("--download-integration-data"):
        cohort_root = tmp_path_factory.mktemp("tcia_download") / "TCGA-PRAD"
        try:
            for target in _DOWNLOAD_TARGETS:
                dest = cohort_root / target["patient_id"] / target["study_dir"] / target["series_dir"]
                _download_series_zip(target["uid"], dest)
        except Exception as e:
            shutil.rmtree(cohort_root.parent, ignore_errors=True)
            pytest.skip(f"Auto-download from TCIA failed ({e}); check network access and retry.")
        try:
            yield cohort_root.parent
        finally:
            shutil.rmtree(cohort_root.parent, ignore_errors=True)
        return

    raw = request.config.getoption("--integration-data-dir") or os.environ.get("MUSIQ_INTEGRATION_DATA_DIR")
    if not raw:
        pytest.skip(
            "No integration data configured; pass --download-integration-data to auto-fetch the "
            "~4 needed series from TCIA, or --integration-data-dir / MUSIQ_INTEGRATION_DATA_DIR "
            "to point at an existing download, to run these real-DICOM tests."
        )
    root = plb.Path(raw)
    if not (root / "TCGA-PRAD").is_dir():
        pytest.skip(f"{root} does not look like the manifest-1773751814915 cohort root (no TCGA-PRAD/ subfolder).")
    yield root


def find_series_dir(root, patient_id, study_glob, series_glob):
    """Locate a series directory by glob, tolerant of the random numeric suffix TCIA appends
    to study/series folder names (e.g. ``...MRI PELVIS-73360``)."""
    matches = sorted(root.glob(f"TCGA-PRAD/{patient_id}/*{study_glob}*/*{series_glob}*"))
    if not matches:
        pytest.skip(f"Series not found under {root}: {patient_id}/*{study_glob}*/*{series_glob}*")
    return matches[0]


@pytest.fixture(scope="session")
def dcm2niix_available():
    if shutil.which("dcm2niix") is None:
        pytest.skip("dcm2niix binary not found on PATH; install the 'dcm2niix' package to run this test.")


@pytest.fixture()
def collector():
    return SeriesSelection(input_dirpath=".", output_dirpath=".", series_keywords={})
