"""Unit tests for utils.mr_nifti_exists: the boolean wrapper around find_mr_niftis used by
convert_dcm2nii_MR to decide whether a series was already converted. A thin wrapper (``bool(
find_mr_niftis(...))``), so only its own boundary -- some matches vs. none -- needs covering;
find_mr_niftis' own matching logic is exercised in test_find_mr_niftis.py.
"""

from musiq.utils import mr_nifti_exists


def test_returns_true_when_a_matching_nifti_exists(tmp_path):
    study_dir = tmp_path / "TESTPAT" / "20240101"
    study_dir.mkdir(parents=True)
    (study_dir / "t1_tse_tra_5.nii.gz").touch()

    assert mr_nifti_exists(study_dir, protocol_name="T1 TSE TRA") is True


def test_returns_false_when_no_matching_nifti_exists(tmp_path):
    study_dir = tmp_path / "TESTPAT" / "20240101"
    study_dir.mkdir(parents=True)

    assert mr_nifti_exists(study_dir, protocol_name="T1 TSE TRA") is False
