"""Fixtures shared across all unit-test subpackages (series_selection, utils, ...).

Anything here is generic infrastructure -- building synthetic DICOM series and guarding
real-binary-dependent tests -- with no dependency on a specific stage's classes.
"""

import shutil

import pytest

from ._dicom_builder import write_dicom_series


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
