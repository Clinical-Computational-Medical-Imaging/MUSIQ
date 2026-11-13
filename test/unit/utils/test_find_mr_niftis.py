"""Unit tests for utils.find_mr_niftis: locating an already-converted MR NIfTI on disk from a
series' ProtocolName (falling back to SeriesDescription), so convert_dcm2nii_MR can skip
reconversion. Pure filename matching against files already on disk -- no dcm2niix/DICOM
required -- so every branch is reachable directly with empty placeholder ``.nii.gz`` files.

"""

from musiq.utils import find_mr_niftis


def _study_dir(tmp_path, patient_id="TESTPAT", study_date="20240101"):
    d = tmp_path / patient_id / study_date
    d.mkdir(parents=True)
    return d


def test_matches_using_protocol_name(tmp_path):
    study_dir = _study_dir(tmp_path)
    (study_dir / "t1_tse_tra_5.nii.gz").touch()

    matches = find_mr_niftis(study_dir, protocol_name="T1 TSE TRA", series_description="ignored")

    assert [f.name for f in matches] == ["t1_tse_tra_5.nii.gz"]


def test_falls_back_to_series_description_when_protocol_name_is_absent(tmp_path):
    """Real MR DICOMs in this cohort don't carry (0018,1030) ProtocolName -- dcm2niix's `%p` token
    then falls back to SeriesDescription, so the match must too."""
    study_dir = _study_dir(tmp_path)
    (study_dir / "t2_axial_4.nii.gz").touch()

    matches = find_mr_niftis(study_dir, protocol_name=None, series_description="T2 AXIAL")

    assert [f.name for f in matches] == ["t2_axial_4.nii.gz"]


def test_falls_back_to_series_description_when_protocol_name_is_empty_string(tmp_path):
    """An empty-but-present tag must be treated the same as an absent one, not matched literally."""
    study_dir = _study_dir(tmp_path)
    (study_dir / "t2_axial_4.nii.gz").touch()

    matches = find_mr_niftis(study_dir, protocol_name="", series_description="T2 AXIAL")

    assert [f.name for f in matches] == ["t2_axial_4.nii.gz"]


def test_no_protocol_name_or_series_description_returns_empty_list(tmp_path):
    study_dir = _study_dir(tmp_path)
    (study_dir / "t2_axial_4.nii.gz").touch()

    assert find_mr_niftis(study_dir, protocol_name=None, series_description=None) == []


def test_unrelated_niftis_in_the_study_dir_are_not_matched(tmp_path):
    study_dir = _study_dir(tmp_path)
    (study_dir / "t1_tse_tra_5.nii.gz").touch()
    (study_dir / "t2_axial_4.nii.gz").touch()

    matches = find_mr_niftis(study_dir, protocol_name="T1 TSE TRA", series_description=None)

    assert [f.name for f in matches] == ["t1_tse_tra_5.nii.gz"]


def test_excludes_niftis_named_after_the_patient_id(tmp_path):
    """Side-project artifacts sharing the study dir (e.g. a derived perfusion map) can be named
    ``<patient_id>_...`` -- real dcm2niix MR outputs never start with the patient_id, so these
    must be excluded even when they would otherwise satisfy the stem/digit regex."""
    study_dir = _study_dir(tmp_path, patient_id="flair")
    (study_dir / "flair_tracew_2.nii.gz").touch()  # starts with patient_id "flair" -> excluded

    matches = find_mr_niftis(study_dir, protocol_name="FLAIR TRACEW", series_description=None)

    assert matches == []


def test_returns_shortest_matching_name_first(tmp_path):
    """A derived/secondary file sharing the source's prefix (longer name) must sort after the
    source NIfTI itself."""
    study_dir = _study_dir(tmp_path)
    (study_dir / "flair_2_derived.nii.gz").touch()
    (study_dir / "flair_2.nii.gz").touch()

    matches = find_mr_niftis(study_dir, protocol_name="FLAIR", series_description=None)

    assert [f.name for f in matches] == ["flair_2.nii.gz", "flair_2_derived.nii.gz"]


def test_series_sharing_a_protocol_name_are_not_confused_with_each_other(tmp_path):
    study_dir = _study_dir(tmp_path)
    (study_dir / "female_pelvis_5.nii.gz").touch()  # T1 AXIAL, SeriesNumber 5
    (study_dir / "female_pelvis_4.nii.gz").touch()  # T2 AXIAL, SeriesNumber 4 -- same ProtocolName

    t1_matches = find_mr_niftis(
        study_dir, protocol_name="female Pelvis/", series_description="T1 AXIAL", series_number=5
    )
    t2_matches = find_mr_niftis(
        study_dir, protocol_name="female Pelvis/", series_description="T2 AXIAL", series_number=4
    )

    assert [f.name for f in t1_matches] == ["female_pelvis_5.nii.gz"]
    assert [f.name for f in t2_matches] == ["female_pelvis_4.nii.gz"]


def test_without_series_number_any_trailing_digit_still_matches(tmp_path):
    """Backward-compatible fallback for callers that don't have SeriesNumber on hand: without it,
    matching reverts to the old (bug-prone) "any trailing digit" behavior."""
    study_dir = _study_dir(tmp_path)
    (study_dir / "female_pelvis_4.nii.gz").touch()

    matches = find_mr_niftis(study_dir, protocol_name="female Pelvis/", series_description="T2 AXIAL")

    assert [f.name for f in matches] == ["female_pelvis_4.nii.gz"]


def test_series_number_does_not_match_as_a_prefix_of_a_longer_number(tmp_path):
    """SeriesNumber 4 must not match a file actually numbered 40 -- the exact-match anchor needs
    a negative lookahead, not just a literal digit prefix."""
    study_dir = _study_dir(tmp_path)
    (study_dir / "female_pelvis_40.nii.gz").touch()

    matches = find_mr_niftis(study_dir, protocol_name="female Pelvis/", series_description="T2 AXIAL", series_number=4)

    assert matches == []
