"""Unit tests for utils.select_dominant_ct_acquisition: filtering out a minority acquisition
bundled under the same SeriesInstanceUID with mismatched slice spacing (see the function's own
docstring for the real-world scenario -- a whole-body stack plus coarser head/feet "end-cap"
blocks).

test_convert_dcm2nii_ct.py mocks this function out entirely and says why in its own module
docstring: "symlink-based dominant-acquisition filtering isn't reliably provokable through real
DICOM data ... orthogonal to what this test is verifying". That's true for the *symlinking*
wired up in convert_dcm2nii_CT, but the filtering logic itself only reads DICOM tags
(ImagePositionPatient/ImageOrientationPatient/AcquisitionNumber) and needs no dcm2niix at all --
so it's tested directly here with a synthetic multi-acquisition series.
"""

import pathlib as plb

from musiq.utils import select_dominant_ct_acquisition


def _mixed_spacing_tags(i):
    """4 slices of AcquisitionNumber 1 at 10mm spacing (z=0,10,20,30), then 2 slices of
    AcquisitionNumber 2 at 5mm spacing (z=100,105) -- mismatched, so acquisition 2 (the minority)
    must be dropped."""
    if i < 4:
        return {"AcquisitionNumber": 1, "ImagePositionPatient": [0.0, 0.0, float(i * 10)]}
    j = i - 4
    return {"AcquisitionNumber": 2, "ImagePositionPatient": [0.0, 0.0, float(100 + j * 5)]}


def _uniform_spacing_tags(i):
    """Two acquisitions, both at 10mm spacing -- a coherent multi-part volume, not mismatched."""
    if i < 4:
        return {"AcquisitionNumber": 1, "ImagePositionPatient": [0.0, 0.0, float(i * 10)]}
    j = i - 4
    return {"AcquisitionNumber": 2, "ImagePositionPatient": [0.0, 0.0, float(40 + j * 10)]}


def test_mixed_spacing_keeps_only_the_dominant_acquisition(dicom_series_factory):
    dcm_dir = dicom_series_factory("CT", subdir="mixed", n_slices=6, rows=4, cols=4, per_slice_tags=_mixed_spacing_tags)

    kept = select_dominant_ct_acquisition(dcm_dir)

    assert kept is not None
    kept_names = sorted(plb.Path(p).name for p in kept)
    assert kept_names == ["slice_000.dcm", "slice_001.dcm", "slice_002.dcm", "slice_003.dcm"]


def test_uniform_spacing_across_acquisitions_returns_none(dicom_series_factory):
    """Multiple acquisitions but consistent spacing = a genuine multi-part volume -> convert the
    whole directory as-is, don't filter anything."""
    dcm_dir = dicom_series_factory(
        "CT", subdir="uniform", n_slices=6, rows=4, cols=4, per_slice_tags=_uniform_spacing_tags
    )

    assert select_dominant_ct_acquisition(dcm_dir) is None


def test_single_acquisition_returns_none(dicom_series_factory):
    dcm_dir = dicom_series_factory("CT", subdir="single", n_slices=4, rows=4, cols=4)

    assert select_dominant_ct_acquisition(dcm_dir) is None


def test_empty_directory_returns_none(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert select_dominant_ct_acquisition(empty_dir) is None


def test_non_file_directory_entries_are_skipped(dicom_series_factory):
    """os.scandir also yields subdirectories -- a stray one (e.g. a DICOMDIR-adjacent folder)
    must not break the tag-reading loop or affect the result."""
    dcm_dir = dicom_series_factory(
        "CT", subdir="mixed_with_subdir", n_slices=6, rows=4, cols=4, per_slice_tags=_mixed_spacing_tags
    )
    (plb.Path(dcm_dir) / "subdir").mkdir()

    kept = select_dominant_ct_acquisition(dcm_dir)

    assert kept is not None
    assert sorted(plb.Path(p).name for p in kept) == [
        "slice_000.dcm",
        "slice_001.dcm",
        "slice_002.dcm",
        "slice_003.dcm",
    ]


def test_unreadable_file_in_the_directory_is_skipped(dicom_series_factory):
    """A non-DICOM file living alongside the series (dcmread raises) must be skipped, not crash
    the whole scan or get counted into either acquisition."""
    dcm_dir = dicom_series_factory(
        "CT", subdir="mixed_with_garbage", n_slices=6, rows=4, cols=4, per_slice_tags=_mixed_spacing_tags
    )
    (plb.Path(dcm_dir) / "garbage.dcm").write_bytes(b"not a real dicom file at all")

    kept = select_dominant_ct_acquisition(dcm_dir)

    assert kept is not None
    assert sorted(plb.Path(p).name for p in kept) == [
        "slice_000.dcm",
        "slice_001.dcm",
        "slice_002.dcm",
        "slice_003.dcm",
    ]


def test_degenerate_orientation_returns_none(dicom_series_factory):
    """A zero-length cross product of the row/column direction cosines (e.g. both pointing the
    same way) means the slice normal can't be determined at all -- must bail out immediately
    rather than dividing by a zero norm."""
    dcm_dir = dicom_series_factory(
        "CT",
        subdir="degenerate_orientation",
        n_slices=4,
        rows=4,
        cols=4,
        extra_tags={"ImageOrientationPatient": [1, 0, 0, 1, 0, 0]},
    )

    assert select_dominant_ct_acquisition(dcm_dir) is None


def _one_multi_slice_and_one_single_slice_acquisition(i):
    """2 acquisitions (so the "single acquisition" short-circuit doesn't apply), but only one of
    them has >=2 slices -- there's nothing to compare its spacing against."""
    if i < 4:
        return {"AcquisitionNumber": 1, "ImagePositionPatient": [0.0, 0.0, float(i * 10)]}
    return {"AcquisitionNumber": 2, "ImagePositionPatient": [0.0, 0.0, 100.0]}


def test_fewer_than_two_multi_slice_acquisitions_returns_none(dicom_series_factory):
    dcm_dir = dicom_series_factory(
        "CT",
        subdir="one_single_slice_acq",
        n_slices=5,
        rows=4,
        cols=4,
        per_slice_tags=_one_multi_slice_and_one_single_slice_acquisition,
    )

    assert select_dominant_ct_acquisition(dcm_dir) is None


def _globally_spread_but_individually_within_tolerance_tags(i):
    """3 acquisitions whose spacings (10, 10.9, 9.1) span a >10% global min/max ratio -- so the
    "one consistent spacing" short-circuit does NOT apply -- yet every single one of them is
    individually within the default 10% tolerance of the dominant (5-slice) acquisition's 10mm
    spacing. Nothing ends up actually dropped despite the series looking "mixed" at a glance."""
    if i < 5:
        return {"AcquisitionNumber": 1, "ImagePositionPatient": [0.0, 0.0, float(i * 10.0)]}
    if i < 8:
        j = i - 5
        return {"AcquisitionNumber": 2, "ImagePositionPatient": [0.0, 0.0, 100.0 + j * 10.9]}
    j = i - 8
    return {"AcquisitionNumber": 3, "ImagePositionPatient": [0.0, 0.0, 200.0 + j * 9.1]}


def test_nothing_actually_dropped_despite_mixed_spacing_returns_none(dicom_series_factory):
    dcm_dir = dicom_series_factory(
        "CT",
        subdir="all_within_tolerance",
        n_slices=10,
        rows=4,
        cols=4,
        per_slice_tags=_globally_spread_but_individually_within_tolerance_tags,
    )

    assert select_dominant_ct_acquisition(dcm_dir) is None
