"""Unit tests for utils.resolve_pet_decay_reference: which timestamp SUV/SUL decay-correct
against, resolved from DecayCorrection together with SeriesTime / per-slice AcquisitionTime.

Exercises every branch documented in the function's own docstring — most importantly the
START-with-varying-per-bed-AcquisitionTime case, which is a *real*, previously observed bug
(per-slice AcquisitionTime inflated SUV by up to ~25% — see the comment in
convert_dcm2nii_PET/resolve_pet_decay_reference), not a hypothetical edge case. A whole-body PET
acquires each bed position at a different AcquisitionTime over the ~10-20 min scan, so these
build synthetic series with genuinely varying per-slice AcquisitionTime (via
``per_slice_tags``) rather than one constant time for every slice.
"""

import logging
import pathlib as plb

import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from musiq.utils import resolve_pet_decay_reference


def _acq_times(times):
    """per_slice_tags callable assigning one AcquisitionTime per slice index, out of file order
    (the earliest is deliberately not slice 0) so a correct implementation must scan the whole
    series rather than trust the first file."""
    return lambda i: {"AcquisitionTime": times[i]}


@pytest.fixture()
def pet_series(dicom_series_factory):
    def _make(extra_tags=None, per_slice_tags=None):
        return dicom_series_factory(
            "PT",
            subdir="pet_timing",
            n_slices=3,
            rows=4,
            cols=4,
            extra_tags=extra_tags or {},
            per_slice_tags=per_slice_tags,
        )

    return _make


def test_start_uses_series_time_not_varying_bed_acquisition_time(pet_series):
    """The documented bug: whole-body PET has a different AcquisitionTime per bed position.
    START must decay-correct to SeriesTime (the true scan start), never to any of those."""
    pet_dir = pet_series(
        extra_tags={"SeriesTime": "110000", "DecayCorrection": "START"},
        per_slice_tags=_acq_times(["113000", "110500", "114500"]),
    )

    ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == "110000"
    assert decay_flag == "START"


def test_start_falls_back_to_earliest_acquisition_time_when_series_time_missing(pet_series):
    pet_dir = pet_series(
        extra_tags={"DecayCorrection": "START"},
        per_slice_tags=_acq_times(["113000", "110500", "114500"]),
    )

    ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == "110500"  # the true earliest, not slice 0's own AcquisitionTime ("113000")
    assert decay_flag == "START"


def test_start_falls_back_to_earliest_when_series_time_is_later_than_first_acquisition(pet_series, caplog):
    """A SeriesTime a vendor sets to a later (e.g. reconstruction) time rather than the true
    acquisition start would re-inflate SUV if trusted -- must be detected and discarded in favor
    of the earliest AcquisitionTime."""
    pet_dir = pet_series(
        extra_tags={"SeriesTime": "120000", "DecayCorrection": "START"},
        per_slice_tags=_acq_times(["113000", "110500", "114500"]),
    )

    with caplog.at_level(logging.WARNING):
        ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == "110500"
    assert decay_flag == "START"
    assert "unreliable" in caplog.text


def test_decay_flag_absent_behaves_like_start(pet_series):
    """No DecayCorrection tag at all (some scanners omit it) must fall into the same
    SeriesTime-preferring branch as an explicit START."""
    pet_dir = pet_series(
        extra_tags={"SeriesTime": "110000"},
        per_slice_tags=_acq_times(["113000", "110500", "114500"]),
    )

    ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == "110000"
    assert decay_flag is None


def test_admin_returns_radiopharmaceutical_start_time(pet_series):
    """ADMIN: pixels are already referenced to the injection time, so the returned reference must
    be the injection time itself (a no-op decay, Δt = 0), not SeriesTime or any acquisition time."""
    seq_item = Dataset()
    seq_item.RadiopharmaceuticalStartTime = "090000"
    pet_dir = pet_series(
        extra_tags={
            "SeriesTime": "110000",
            "DecayCorrection": "ADMIN",
            "RadiopharmaceuticalInformationSequence": Sequence([seq_item]),
        },
        per_slice_tags=_acq_times(["113000", "110500", "114500"]),
    )

    ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == "090000"
    assert decay_flag == "ADMIN"


def test_admin_without_injection_time_falls_back_to_earliest_and_warns(pet_series, caplog):
    pet_dir = pet_series(
        extra_tags={"SeriesTime": "110000", "DecayCorrection": "ADMIN"},  # no radiopharm sequence at all
        per_slice_tags=_acq_times(["113000", "110500", "114500"]),
    )

    with caplog.at_level(logging.WARNING):
        ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == "110500"
    assert decay_flag == "ADMIN"
    assert "no RadiopharmaceuticalStartTime" in caplog.text


def test_none_warns_and_falls_back_to_earliest_acquisition_time(pet_series, caplog):
    pet_dir = pet_series(
        extra_tags={"SeriesTime": "110000", "DecayCorrection": "NONE"},
        per_slice_tags=_acq_times(["113000", "110500", "114500"]),
    )

    with caplog.at_level(logging.WARNING):
        ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == "110500"
    assert decay_flag == "NONE"
    assert "not decay-corrected" in caplog.text


def test_unrecognized_decay_flag_warns_and_falls_back_to_earliest_acquisition_time(pet_series, caplog):
    pet_dir = pet_series(
        extra_tags={"SeriesTime": "110000", "DecayCorrection": "MANUAL"},
        per_slice_tags=_acq_times(["113000", "110500", "114500"]),
    )

    with caplog.at_level(logging.WARNING):
        ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == "110500"
    assert decay_flag == "MANUAL"
    assert "unrecognized" in caplog.text


def test_final_fallback_when_neither_series_time_nor_acquisition_time_present(pet_series, caplog):
    pet_dir = pet_series(extra_tags={"DecayCorrection": "START"})

    with caplog.at_level(logging.WARNING):
        ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == ""
    assert decay_flag == "START"
    assert "Could not resolve" in caplog.text


def test_no_dicom_files_raises_file_not_found(tmp_path):
    """With ds=None (the default, real-usage path), the reference dataset is read from the
    directory itself -- an empty/wrong directory must raise immediately rather than proceed with
    no dataset at all."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        resolve_pet_decay_reference(empty_dir)


def test_unreadable_file_among_the_series_is_skipped_when_scanning_for_the_earliest_time(pet_series):
    """The earliest-AcquisitionTime scan reads every file in the series; one that fails to parse
    (e.g. a stray non-DICOM sidecar) must be skipped, not crash the whole resolution."""
    pet_dir = pet_series(
        extra_tags={"DecayCorrection": "START"},
        per_slice_tags=_acq_times(["113000", "110500", "114500"]),
    )
    (plb.Path(pet_dir) / "zzz_garbage.dcm").write_bytes(b"not a real dicom file at all")

    ref_time, decay_flag = resolve_pet_decay_reference(pet_dir)

    assert ref_time == "110500"  # the garbage file is skipped, real earliest is still found
    assert decay_flag == "START"
