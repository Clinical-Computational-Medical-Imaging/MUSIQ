"""Unit tests for utils.is_mr_filename: deciding whether a NIfTI filename is an MR series worth
segmenting, used by totalsegmentator_inference.py and totalsegmentator_muscle_fat.py to pick MR
files out of a patient's mixed-modality output tree via load_mr_keywords()'s PRIMARY/SECONDARY/
EXCLUSION lists (see config.yaml).
"""

import pytest

from musiq.utils import is_mr_filename

KEYWORDS = {
    "PRIMARY": ["t1_tse", "t2_tse", "dwi", "adc"],
    "SECONDARY": ["dyn"],
    "EXCLUSION": ["localizer", "survey"],
}


@pytest.mark.parametrize("filename", ["t1_tse_tra_5.nii.gz", "T1_TSE_TRA_5.NII.GZ", "patient_t2_tse_3.nii.gz"])
def test_primary_keyword_match_is_case_insensitive(filename):
    assert is_mr_filename(filename, KEYWORDS) is True


def test_secondary_keyword_also_matches():
    assert is_mr_filename("dyn_2.nii.gz", KEYWORDS) is True


def test_no_keyword_present_does_not_match():
    assert is_mr_filename("ct_series_1.nii.gz", KEYWORDS) is False


def test_exclusion_keyword_overrides_an_otherwise_matching_primary_keyword():
    """A localizer/survey scan can still carry an inclusion keyword (e.g. a generic protocol
    name); EXCLUSION must win regardless of order or which inclusion keyword also matched."""
    assert is_mr_filename("t1_tse_localizer_1.nii.gz", KEYWORDS) is False


def test_exclusion_keyword_alone_without_any_inclusion_match_does_not_match():
    assert is_mr_filename("survey_1.nii.gz", KEYWORDS) is False
