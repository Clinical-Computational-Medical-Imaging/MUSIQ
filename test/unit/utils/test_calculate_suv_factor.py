"""Unit tests for utils.calculate_suv_factor: turning an injected PET dose into the scalar
factor that converts raw (Bq/mL) PET pixel values into SUV (g/mL) via ``convert_pet``.

The factor decays ``total_dose`` (Bq, at ``start_time``) forward to ``acq_time`` — the reference
resolved by ``resolve_pet_decay_reference`` — using the tracer's physical ``half_life`` (seconds),
then divides the patient's ``weight`` (kg, converted to grams via the leading ``1000``) by that
decayed activity: ``suv_factor = 1000 * weight / (total_dose * 0.5 ** (time_diff / half_life))``.
Round half-life/elapsed-time combinations are used below so every assertion can be hand-verified
against that formula, plus one realistic FDG scenario (300 MBq, 6588s = 109.8min half-life)
mirroring the values used in test/unit/series_selection/test_convert_dcm2nii_pet.py.
"""

import pytest

from musiq.utils import calculate_suv_factor


def test_zero_elapsed_time_leaves_dose_undecayed():
    """acq_time == start_time -> 0.5 ** 0 == 1, so the factor is just 1000 * weight / total_dose."""
    factor = calculate_suv_factor(
        total_dose=2_000_000, start_time="100000", half_life=3600, acq_time="100000", weight=80
    )

    assert factor == pytest.approx(0.04)


def test_one_half_life_elapsed_doubles_the_factor():
    """3600s elapsed at a 3600s half-life halves the decayed dose, so the factor (inversely
    proportional to dose) doubles relative to the zero-elapsed baseline."""
    factor = calculate_suv_factor(
        total_dose=2_000_000, start_time="100000", half_life=3600, acq_time="110000", weight=80
    )

    assert factor == pytest.approx(0.08)


def test_two_half_lives_elapsed_quadruples_the_factor():
    factor = calculate_suv_factor(
        total_dose=2_000_000, start_time="100000", half_life=3600, acq_time="120000", weight=80
    )

    assert factor == pytest.approx(0.16)


def test_colon_separated_time_strings_are_supported():
    """time_to_seconds accepts both HHMMSS and HH:MM:SS -- the decay math must agree either way."""
    factor = calculate_suv_factor(
        total_dose=1_000_000, start_time="10:00:00", half_life=3600, acq_time="11:00:00", weight=70
    )

    assert factor == pytest.approx(0.14)


def test_factor_scales_linearly_with_weight():
    baseline = calculate_suv_factor(
        total_dose=1_000_000, start_time="100000", half_life=3600, acq_time="103000", weight=70
    )
    doubled_weight = calculate_suv_factor(
        total_dose=1_000_000, start_time="100000", half_life=3600, acq_time="103000", weight=140
    )

    assert doubled_weight == pytest.approx(2 * baseline)


def test_factor_scales_inversely_with_total_dose():
    baseline = calculate_suv_factor(
        total_dose=1_000_000, start_time="100000", half_life=3600, acq_time="103000", weight=70
    )
    doubled_dose = calculate_suv_factor(
        total_dose=2_000_000, start_time="100000", half_life=3600, acq_time="103000", weight=70
    )

    assert doubled_dose == pytest.approx(baseline / 2)


def test_acquisition_before_start_time_decays_backwards():
    """A negative time_diff (acq_time earlier than start_time) is not expected in practice, but the
    formula handles it symmetrically: the dose is decayed *backwards* (grows), so the factor
    shrinks -- exercised here so the behavior is pinned down rather than left implicit."""
    factor = calculate_suv_factor(
        total_dose=1_000_000, start_time="120000", half_life=3600, acq_time="110000", weight=70
    )

    assert factor == pytest.approx(0.035)


def test_realistic_fdg_clinical_values():
    """300 MBq injected dose, F-18 FDG half-life (6588s), 80kg patient, 30min uptake -- the same
    magnitude of values series_selection.py's convert_dcm2nii_PET passes in practice."""
    factor = calculate_suv_factor(total_dose=3e8, start_time="113000", half_life=6588.0, acq_time="120000", weight=80.0)

    assert factor == pytest.approx(0.0003222681344017752)


def test_zero_weight_gives_zero_suv_factor():
    """weight is the numerator (via the leading 1000), so a zero weight -- e.g. a not-yet-recorded
    PatientWeight -- silently collapses the factor to 0 rather than raising; pinned here so the
    behavior is explicit rather than only discovered downstream in a suspiciously all-zero SUV
    volume."""
    factor = calculate_suv_factor(
        total_dose=2_000_000, start_time="100000", half_life=3600, acq_time="103000", weight=0
    )

    assert factor == 0.0


def test_zero_total_dose_raises_zero_division_error():
    """A zero (missing/unrecorded) RadionuclideTotalDose decays to a zero act_dose, and the final
    ``1000 * weight / act_dose`` divides by that zero -- calculate_suv_factor has no guard against
    this, so it raises rather than returning e.g. inf or nan. Pinned as a regression test: this is
    the medically most sensitive formula in the pipeline and this edge case is never exercised by
    the gated end-to-end PET conversion test (which only feeds it real, non-zero doses)."""
    with pytest.raises(ZeroDivisionError):
        calculate_suv_factor(total_dose=0.0, start_time="100000", half_life=3600, acq_time="103000", weight=80)
