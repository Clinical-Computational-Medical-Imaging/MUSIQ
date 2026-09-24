"""Real dcm2niix conversion tests for convert_dcm2nii_PET.

The downloaded TCIA sample data (see test/integration) has no genuine quantifiable PET series
(only a derived MIP without RadiopharmaceuticalInformationSequence), so these build a minimal
synthetic PET DICOM series instead — real dcm2niix conversion, but with exactly the tags needed
to also verify the resulting SUV pixel values are mathematically correct, not just that a file
was produced.
"""

import logging

import nibabel as nib
import pytest

from musiq.utils import calculate_suv_factor

from .._dicom_builder import pet_radiopharm_tags

pytestmark = pytest.mark.usefixtures("dcm2niix_available")


def _make_pet_series(dicom_series_factory, pixel_value=100, weight=80.0, **extra_tags):
    tags = pet_radiopharm_tags()
    tags.update(extra_tags)
    return dicom_series_factory(
        "PT",
        subdir="pet_series",
        n_slices=3,
        rows=8,
        cols=8,
        pixel_value=lambda i: pixel_value,
        pixel_representation=0,
        extra_tags={
            "PatientWeight": weight,
            "SeriesTime": "120000",
            "DecayCorrection": "START",
            **tags,
        },
    )


def test_convert_pet_produces_mathematically_correct_suv_values(collector, tmp_path, dicom_series_factory):
    pet_dir = _make_pet_series(dicom_series_factory, pixel_value=100, weight=80.0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    dicom_tags = collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_dir, output_dirpath=out_dir)

    assert (out_dir / "PET.nii.gz").is_file()
    assert (out_dir / "SUV.nii.gz").is_file()

    expected_factor = calculate_suv_factor(
        total_dose=3e8, start_time="113000", half_life=6588.0, acq_time="120000", weight=80.0
    )
    assert dicom_tags["SUVFactor"] == pytest.approx(expected_factor)
    assert dicom_tags["PatientWeight"] == 80.0
    assert dicom_tags["DecayCorrectionReference"] == "START"

    suv = nib.load(str(out_dir / "SUV.nii.gz"))
    pet = nib.load(str(out_dir / "PET.nii.gz"))
    assert (pet.get_fdata() == 100).all()
    assert suv.get_fdata() == pytest.approx(100 * expected_factor, rel=1e-4)


def test_convert_pet_falls_back_to_study_weight_when_series_weight_is_zero(
    collector, tmp_path, dicom_series_factory, caplog
):
    pet_dir = _make_pet_series(dicom_series_factory, pixel_value=50, weight=0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with caplog.at_level(logging.WARNING):
        dicom_tags = collector.convert_dcm2nii_PET(
            PET_dcm_dirpath=pet_dir, output_dirpath=out_dir, fallback_weight=70.0
        )

    assert dicom_tags["PatientWeight"] == 70.0
    assert "using study-level weight 70.0" in caplog.text

    expected_factor = calculate_suv_factor(
        total_dose=3e8, start_time="113000", half_life=6588.0, acq_time="120000", weight=70.0
    )
    assert dicom_tags["SUVFactor"] == pytest.approx(expected_factor)


def test_convert_pet_logs_error_when_weight_invalid_and_no_fallback(collector, tmp_path, dicom_series_factory, caplog):
    pet_dir = _make_pet_series(dicom_series_factory, pixel_value=50, weight=0)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with caplog.at_level(logging.ERROR):
        dicom_tags = collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_dir, output_dirpath=out_dir)

    assert dicom_tags["PatientWeight"] == 0
    assert dicom_tags["SUVFactor"] == 0
    assert "No valid PatientWeight" in caplog.text


def test_convert_pet_skips_reconversion_when_pet_nifti_already_exists(collector, tmp_path, dicom_series_factory):
    pet_dir = _make_pet_series(dicom_series_factory)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_dir, output_dirpath=out_dir)
    pet_path = out_dir / "PET.nii.gz"
    original_mtime = pet_path.stat().st_mtime_ns

    dicom_tags = collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_dir, output_dirpath=out_dir)

    assert pet_path.stat().st_mtime_ns == original_mtime
    assert dicom_tags["Modality"] == "PT"


def test_convert_pet_skips_suv_regeneration_when_suv_already_exists(collector, tmp_path, dicom_series_factory):
    pet_dir = _make_pet_series(dicom_series_factory)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_dir, output_dirpath=out_dir)
    suv_path = out_dir / "SUV.nii.gz"
    original_mtime = suv_path.stat().st_mtime_ns
    original_values = nib.load(str(suv_path)).get_fdata().copy()

    # Force a different factor to prove a second run does NOT overwrite SUV.nii.gz.
    collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_dir, output_dirpath=out_dir, fallback_weight=999.0)

    assert suv_path.stat().st_mtime_ns == original_mtime
    assert (nib.load(str(suv_path)).get_fdata() == original_values).all()


def test_convert_pet_uses_series_time_not_varying_bed_acquisition_time_for_suv(
    collector, tmp_path, dicom_series_factory
):
    """Regression test for a real, previously observed bug: whole-body PET's AcquisitionTime
    varies per bed position over the scan, and naively using it (instead of SeriesTime) for the
    decay-correction reference inflated SUV by up to ~25% (see resolve_pet_decay_reference's
    docstring). Simulates that per-bed timing directly through the full convert_dcm2nii_PET path,
    not just the isolated resolve_pet_decay_reference unit -- the SUV factor must come out
    identical to using SeriesTime alone, regardless of the (deliberately later and varying)
    per-slice AcquisitionTime values."""
    tags = pet_radiopharm_tags()
    pet_dir = dicom_series_factory(
        "PT",
        subdir="pet_series_timing",
        n_slices=3,
        rows=8,
        cols=8,
        pixel_value=lambda i: 100,
        pixel_representation=0,
        extra_tags={
            "PatientWeight": 80.0,
            "SeriesTime": "120000",
            "DecayCorrection": "START",
            **tags,
        },
        # Later than SeriesTime and different per slice/bed position -- if this leaked into the
        # SUV factor instead of SeriesTime, the assertion below would fail.
        per_slice_tags=lambda i: {"AcquisitionTime": ["121500", "123000", "124500"][i]},
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    dicom_tags = collector.convert_dcm2nii_PET(PET_dcm_dirpath=pet_dir, output_dirpath=out_dir)

    expected_factor = calculate_suv_factor(
        total_dose=3e8, start_time="113000", half_life=6588.0, acq_time="120000", weight=80.0
    )
    assert dicom_tags["SUVFactor"] == pytest.approx(expected_factor)

    suv = nib.load(str(out_dir / "SUV.nii.gz"))
    assert suv.get_fdata() == pytest.approx(100 * expected_factor, rel=1e-4)


def test_convert_pet_raises_when_no_dicom_files_found(collector, tmp_path):
    empty_dir = tmp_path / "empty_pet"
    empty_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        collector.convert_dcm2nii_PET(PET_dcm_dirpath=empty_dir, output_dirpath=out_dir)
