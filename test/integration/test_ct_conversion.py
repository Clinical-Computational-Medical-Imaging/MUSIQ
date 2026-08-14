"""Real DICOM -> NIfTI conversion tests for convert_dcm2nii_CT, against genuine downloaded
TCIA data (see conftest.py for how to point pytest at the local download).

TCGA-VP-A878's abdomen study has two separate CT series (different SeriesInstanceUID, i.e.
different reconstructions of the same acquisition: a soft-tissue and a lung kernel), so it also
exercises real-world directory layout/tag quirks that a synthetic series wouldn't.
"""

import logging
import os

import nibabel as nib
import numpy as np
import pydicom
import pytest

import musiq.series_selection

from .conftest import find_series_dir

pytestmark = [pytest.mark.usefixtures("dcm2niix_available")]


@pytest.fixture()
def ct_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-VP-A878",
        study_glob="Abdomen05CAP",
        series_glob="ChestAbdPel soft tissue",
    )


def test_real_ct_series_converts_to_a_plausible_ct_volume(collector, tmp_path, ct_series_dir):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    dicom_tags = collector.convert_dcm2nii_CT(CT_dcm_dirpath=ct_series_dir, output_dirpath=out_dir)

    out_fpath = out_dir / "CT.nii.gz"
    assert out_fpath.is_file()
    assert dicom_tags["Modality"] == "CT"

    img = nib.load(str(out_fpath))
    assert img.ndim == 3
    n_dicom_slices = len(list(ct_series_dir.glob("*.dcm"))) or sum(1 for _ in ct_series_dir.iterdir())
    assert img.shape[2] == n_dicom_slices

    data = img.get_fdata()
    # A real CT volume must show actual anatomy: Hounsfield range roughly air (~-1000) to
    # dense bone/contrast (up to a few thousand), and not be a degenerate constant/blank volume.
    assert data.min() < -500
    assert data.max() > 100
    assert data.std() > 50


def test_real_ct_conversion_is_skipped_when_output_already_exists(collector, tmp_path, ct_series_dir, caplog):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    collector.convert_dcm2nii_CT(CT_dcm_dirpath=ct_series_dir, output_dirpath=out_dir)
    out_fpath = out_dir / "CT.nii.gz"
    original_bytes = out_fpath.read_bytes()

    with caplog.at_level(logging.INFO):
        dicom_tags = collector.convert_dcm2nii_CT(CT_dcm_dirpath=ct_series_dir, output_dirpath=out_dir)

    assert out_fpath.read_bytes() == original_bytes
    assert "already exists" in caplog.text
    assert dicom_tags["Modality"] == "CT"


def test_second_reconstruction_of_the_same_study_does_not_overwrite_the_first(
    collector, tmp_path, integration_data_dir, ct_series_dir
):
    """Two CT series in one study (soft-tissue and lung kernel) share the fixed CT.nii.gz output
    name; per the documented single-series-per-modality-per-study assumption, the first
    conversion wins and the second is treated as already-converted rather than overwriting it."""
    lung_series_dir = find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-VP-A878",
        study_glob="Abdomen05CAP",
        series_glob="ChestAbdPel LUNG",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    collector.convert_dcm2nii_CT(CT_dcm_dirpath=ct_series_dir, output_dirpath=out_dir)
    first_bytes = (out_dir / "CT.nii.gz").read_bytes()

    collector.convert_dcm2nii_CT(CT_dcm_dirpath=lung_series_dir, output_dirpath=out_dir)

    assert (out_dir / "CT.nii.gz").read_bytes() == first_bytes


def test_real_ct_affine_matches_dicom_slice_spacing(collector, tmp_path, ct_series_dir):
    """Sanity-checks the NIfTI geometry (not just its existence) against the DICOM's own
    PixelSpacing, i.e. real image functionality rather than a bare file-exists check."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    collector.convert_dcm2nii_CT(CT_dcm_dirpath=ct_series_dir, output_dirpath=out_dir)

    first_dcm = next(f for f in sorted(ct_series_dir.iterdir()) if f.is_file())
    ds = pydicom.dcmread(str(first_dcm), stop_before_pixels=True)
    expected_spacing = [float(x) for x in ds.PixelSpacing]

    img = nib.load(str(out_dir / "CT.nii.gz"))
    voxel_sizes = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    assert voxel_sizes[0] == pytest.approx(expected_spacing[0], abs=0.05)
    assert voxel_sizes[1] == pytest.approx(expected_spacing[1], abs=0.05)


@pytest.fixture()
def irregular_spacing_ct_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="ACRIN-NSCLC-FDG-PET",
        patient_id="ACRIN-NSCLC-FDG-PET-114",
        study_glob="CT WB",
        series_glob="Recon 3",
    )


def test_real_irregular_slice_spacing_is_repaired_to_the_dicom_derived_value(
    mocker, collector, tmp_path, irregular_spacing_ct_series_dir
):
    """This series has one irregular interslice gap among otherwise-uniform spacing, which makes
    dcm2niix emit an _Eq_1 output whose z-spacing/sign it derives wrong from SliceThickness.
    repair_ct_affine_from_dicom must correct it to the value actually implied by
    ImagePositionPatient — computed here independently of that function, not by calling it.

    repair_spy confirms the repair path is taken; it does not by itself confirm the repaired
    value is correct (see BUGREPORT_ct_pt_volume_and_sidecar_selection.md re: input-file-count
    vs. output-slice-count when Eq_1 resampling changes the slice count)."""
    repair_spy = mocker.spy(musiq.series_selection, "repair_ct_affine_from_dicom")

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    collector.convert_dcm2nii_CT(CT_dcm_dirpath=irregular_spacing_ct_series_dir, output_dirpath=out_dir)

    assert repair_spy.spy_return is True

    positions = []
    for name in os.listdir(irregular_spacing_ct_series_dir):
        ds = pydicom.dcmread(
            str(irregular_spacing_ct_series_dir / name), stop_before_pixels=True, specific_tags=["ImagePositionPatient"]
        )
        positions.append(float(ds.ImagePositionPatient[2]))
    positions.sort()
    # The uniform spacing implied by the full z-extent and slice count — not a median of the
    # per-gap diffs, which the one irregular gap here would skew toward the wrong value.
    true_spacing = (positions[-1] - positions[0]) / (len(positions) - 1)

    img = nib.load(str(out_dir / "CT.nii.gz"))
    z_spacing = float(np.linalg.norm(img.affine[:3, 2]))
    assert z_spacing == pytest.approx(true_spacing, abs=0.05)
