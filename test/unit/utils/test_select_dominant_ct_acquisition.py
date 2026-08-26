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
