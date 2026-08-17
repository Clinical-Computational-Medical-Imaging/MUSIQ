"""Fixtures for real-DICOM integration tests against a locally downloaded TCIA cohort.

These tests exercise the real dcm2niix conversion path against genuine DICOM data (not
mocked), so they need the actual images on disk. Two ways to provide them:

1. Auto-download just the series these tests need, straight from the public TCIA REST API
   (no login, no Java NBIA Data Retriever), into a temp dir that is deleted again once the test
   session ends:

       pytest test/integration --download-integration-data

   Opt-in only: it hits an external network service and fetches real (de-identified) patient
   imaging data on every run, so it's never on by default.

2. Point at a cohort you already downloaded yourself via the NBIA Data Retriever, using the
   manifest files in this directory:

     - ``manifest-1773751814915.tcia`` — TCGA-PRAD series (CT/MR/PET conversion tests).
     - ``manifest-acrin-nsclc-fdg-pet.tcia`` — one ACRIN-NSCLC-FDG-PET CT series, used only by
       the irregular-slice-spacing affine-repair test.

   Download both into the same root directory (so it ends up containing both a ``TCGA-PRAD/``
   and an ``ACRIN-NSCLC-FDG-PET/`` subfolder), then point pytest at that root:

       pytest test/integration --integration-data-dir "C:\\path\\to\\that\\root"
       MUSIQ_INTEGRATION_DATA_DIR="/path/to/that/root" pytest test/integration

   Downloading only manifest-1773751814915.tcia still works — tests needing the ACRIN series
   just skip.

Tests in this directory skip automatically when none of the above is set.
"""

import io
import os
import pathlib as plb
import shutil
import urllib.request
import zipfile

import pydicom
import pytest

from musiq.series_selection import SeriesSelection

_TCIA_GET_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"

# The exact series these tests need, identified by SeriesInstanceUID. study_dir/series_dir only
# need to *contain* the substrings find_series_dir()'s tests glob for — they don't have to match
# TCIA's real (randomly-suffixed) folder names. collection is explicit per-target: most are
# TCGA-PRAD (manifest-1773751814915.tcia), one is from a different TCIA collection.
_DOWNLOAD_TARGETS = [
    {
        "collection": "TCGA-PRAD",
        "patient_id": "TCGA-VP-A878",
        "study_dir": "Abdomen05CAP",
        "series_dir": "ChestAbdPel soft tissue",
        "uid": "1.3.6.1.4.1.14519.5.2.1.7777.4006.266822032522643745978907862000",
    },
    {
        "collection": "TCGA-PRAD",
        "patient_id": "TCGA-VP-A878",
        "study_dir": "Abdomen05CAP",
        "series_dir": "ChestAbdPel LUNG",
        "uid": "1.3.6.1.4.1.14519.5.2.1.7777.4006.313444960025318623576650119009",
    },
    {
        "collection": "TCGA-PRAD",
        "patient_id": "TCGA-EJ-5495",
        "study_dir": "MRI PELVIS",
        "series_dir": "T2 CORONAL",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6450.4006.236442968616827670504168629430",
    },
    {
        # Same patient/study as T2 CORONAL — shared ProtocolName across series in this study.
        "collection": "TCGA-PRAD",
        "patient_id": "TCGA-EJ-5495",
        "study_dir": "MRI PELVIS",
        "series_dir": "T1 AXIAL",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6450.4006.692201102881185055002252472378",
    },
    {
        "collection": "TCGA-PRAD",
        "patient_id": "TCGA-EJ-5495",
        "study_dir": "MRI PELVIS",
        "series_dir": "T2 AXIAL",
        "uid": "1.3.6.1.4.1.14519.5.2.1.6450.4006.157052036872186731652264278303",
    },
    {
        "collection": "TCGA-PRAD",
        "patient_id": "TCGA-J4-A67O",
        "study_dir": "MRI PELVIS WWO C",
        "series_dir": "DYNAMIC SCAN",
        "uid": "1.3.6.1.4.1.14519.5.2.1.3983.4006.123592663440526734445424125161",
    },
    {
        # Real ORIGINAL/PRIMARY whole-body PET reconstruction with full dosimetry tags —
        # TCGA-VP-A878's own PET is only a derived MIP (no dosimetry tags).
        "collection": "TCGA-PRAD",
        "patient_id": "TCGA-VP-A879",
        "study_dir": "Prostate CA PET",
        "series_dir": "PET WB",
        "uid": "1.3.6.1.4.1.14519.5.2.1.7777.4006.245370922432097277313802824029",
    },
    {
        # One irregular interslice gap among otherwise-uniform spacing; reproduces dcm2niix's
        # "Interslice distance varies" warning and exercises its _Eq_1 resampling path. z_window
        # keeps only the slices around the gap instead of the full 770-slice series.
        "collection": "ACRIN-NSCLC-FDG-PET",
        "patient_id": "ACRIN-NSCLC-FDG-PET-114",
        "study_dir": "CT WB",
        "series_dir": "Recon 3",
        "uid": "1.3.6.1.4.1.14519.5.2.1.7009.2403.200599078995439563089201126426",
        "z_window": (41, 71),
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
        help="Download just the series these tests need directly from the public TCIA REST "
        "API into a temp dir, and delete it again once the test session ends. Opt-in: hits an "
        "external network service and downloads real (de-identified) patient imaging data on "
        "every run. Takes priority over --integration-data-dir/MUSIQ_INTEGRATION_DATA_DIR.",
    )


def _download_series_zip(series_instance_uid: str, dest_dir: plb.Path, z_window: tuple | None = None) -> None:
    """Fetch one series as a zip from TCIA's public getImage endpoint and extract its DICOMs.

    z_window, if given, is a (start, stop) index range into the files sorted by
    ImagePositionPatient z — only that slice range is written to dest_dir, so a test can target
    one part of a large series without downloading/converting all of it.
    """
    url = f"{_TCIA_GET_IMAGE_URL}?SeriesInstanceUID={series_instance_uid}"
    req = urllib.request.Request(url, headers={"User-Agent": "musiq-integration-tests"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        dcm_names = [n for n in zf.namelist() if n.lower().endswith(".dcm")]
        if not dcm_names:
            raise ValueError(f"No .dcm entries in TCIA response for series {series_instance_uid}")
        contents = {os.path.basename(n): zf.read(n) for n in dcm_names}

    if z_window is None:
        for name, content in contents.items():
            dest_dir.joinpath(name).write_bytes(content)
        return

    def z_position(content):
        return float(pydicom.dcmread(io.BytesIO(content), stop_before_pixels=True).ImagePositionPatient[2])

    ordered = sorted(contents.items(), key=lambda item: z_position(item[1]))
    start, stop = z_window
    for name, content in ordered[start:stop]:
        dest_dir.joinpath(name).write_bytes(content)


@pytest.fixture(scope="session")
def integration_data_dir(request, tmp_path_factory):
    if request.config.getoption("--download-integration-data"):
        cohort_root = tmp_path_factory.mktemp("tcia_download")
        try:
            for target in _DOWNLOAD_TARGETS:
                dest = (
                    cohort_root
                    / target["collection"]
                    / target["patient_id"]
                    / target["study_dir"]
                    / target["series_dir"]
                )
                _download_series_zip(target["uid"], dest, z_window=target.get("z_window"))
        except Exception as e:
            shutil.rmtree(cohort_root, ignore_errors=True)
            pytest.skip(f"Auto-download from TCIA failed ({e}); check network access and retry.")
        try:
            yield cohort_root
        finally:
            shutil.rmtree(cohort_root, ignore_errors=True)
        return

    raw = request.config.getoption("--integration-data-dir") or os.environ.get("MUSIQ_INTEGRATION_DATA_DIR")
    if not raw:
        pytest.skip(
            "No integration data configured; pass --download-integration-data to auto-fetch the "
            "needed series from TCIA, or --integration-data-dir / MUSIQ_INTEGRATION_DATA_DIR "
            "to point at an existing download, to run these real-DICOM tests."
        )
    root = plb.Path(raw)
    if not (root / "TCGA-PRAD").is_dir():
        pytest.skip(f"{root} does not look like the manifest-1773751814915 cohort root (no TCGA-PRAD/ subfolder).")
    yield root


def find_series_dir(root, collection, patient_id, study_glob, series_glob):
    """Locate a series directory by glob, tolerant of the random numeric suffix TCIA appends
    to study/series folder names (e.g. ``...MRI PELVIS-73360``)."""
    matches = sorted(root.glob(f"{collection}/{patient_id}/*{study_glob}*/*{series_glob}*"))
    if not matches:
        pytest.skip(f"Series not found under {root}: {collection}/{patient_id}/*{study_glob}*/*{series_glob}*")
    return matches[0]


@pytest.fixture(scope="session")
def dcm2niix_available():
    if shutil.which("dcm2niix") is None:
        pytest.skip("dcm2niix binary not found on PATH; install the 'dcm2niix' package to run this test.")


@pytest.fixture()
def collector():
    return SeriesSelection(input_dirpath=".", output_dirpath=".", series_keywords={})
