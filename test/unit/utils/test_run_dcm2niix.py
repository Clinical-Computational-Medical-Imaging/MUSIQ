"""Unit tests for utils.run_dcm2niix: the dcm2niix subprocess wrapper every series_selection
conversion path calls.

series_selection's unit tests always mock this function out (see e.g. test_convert_dcm2nii_ct.py,
test_convert_dcm2nii_mr.py); its own command construction and error handling -- a failed
dcm2niix invocation must be logged, not raised, so one bad series doesn't crash a whole cohort
run -- had no direct coverage. The happy path against a real dcm2niix binary is exercised
end-to-end by the ``dcm2niix_available``-gated tests in test_convert_dcm2nii_pet.py, so it's not
repeated here; this focuses on command construction and the two exception branches.
"""

import logging
import subprocess

from musiq.utils import run_dcm2niix


def test_builds_expected_command_without_merge(mocker):
    run = mocker.patch("musiq.utils.subprocess.run")

    run_dcm2niix("/dicom/in", "/nifti/out")

    run.assert_called_once_with(
        ["dcm2niix", "-z", "y", "-f", "%p_%s", "-b", "y", "-ba", "n", "-o", "/nifti/out", "/dicom/in"],
        check=True,
    )


def test_merge_flag_adds_dcm2niix_merge_option(mocker):
    run = mocker.patch("musiq.utils.subprocess.run")

    run_dcm2niix("/dicom/in", "/nifti/out", merge=True)

    args, kwargs = run.call_args
    command = args[0]
    assert command == [
        "dcm2niix",
        "-z",
        "y",
        "-f",
        "%p_%s",
        "-b",
        "y",
        "-ba",
        "n",
        "-m",
        "y",
        "-o",
        "/nifti/out",
        "/dicom/in",
    ]


def test_called_process_error_is_logged_and_not_raised(mocker, caplog):
    mocker.patch(
        "musiq.utils.subprocess.run",
        side_effect=subprocess.CalledProcessError(returncode=1, cmd=["dcm2niix"]),
    )

    with caplog.at_level(logging.ERROR):
        run_dcm2niix("/dicom/in", "/nifti/out")  # must not raise

    assert "Error during dcm2niix" in caplog.text


def test_unexpected_exception_is_logged_and_not_raised(mocker, caplog):
    mocker.patch("musiq.utils.subprocess.run", side_effect=FileNotFoundError("dcm2niix not found"))

    with caplog.at_level(logging.ERROR):
        run_dcm2niix("/dicom/in", "/nifti/out")  # must not raise

    assert "Unexpected error" in caplog.text
