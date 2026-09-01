"""Real DICOM -> NIfTI conversion tests for convert_dcm2nii_CT, against genuine downloaded
TCIA data (see conftest.py for how to point pytest at the local download).

TCGA-VP-A878's abdomen study has two separate CT series (different SeriesInstanceUID, i.e.
different reconstructions of the same acquisition: a soft-tissue and a lung kernel), so it also
exercises real-world directory layout/tag quirks that a synthetic series wouldn't.
"""

import json
import logging
import os

import nibabel as nib
import numpy as np
import pydicom
import pytest

from musiq.utils import run_dcm2niix

from ..conftest import find_series_dir

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
    """Verifies the converted NIfTI's in-plane voxel spacing matches the DICOM PixelSpacing."""
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
    collector, tmp_path, irregular_spacing_ct_series_dir
):
    """This series has one irregular interslice gap among otherwise-uniform spacing, which can make
    dcm2niix derive an incorrect slice spacing for its _Eq_1 output. Verifies that the converted
    NIfTI's slice spacing, multiplied by its own slice count, reproduces the physical extent
    implied by the DICOM ImagePositionPatient values — computed here independently of the
    conversion code, so the check holds regardless of whether a correction was applied."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    collector.convert_dcm2nii_CT(CT_dcm_dirpath=irregular_spacing_ct_series_dir, output_dirpath=out_dir)

    positions = []
    for name in os.listdir(irregular_spacing_ct_series_dir):
        ds = pydicom.dcmread(
            str(irregular_spacing_ct_series_dir / name), stop_before_pixels=True, specific_tags=["ImagePositionPatient"]
        )
        positions.append(float(ds.ImagePositionPatient[2]))
    positions.sort()
    dicom_extent = positions[-1] - positions[0]

    img = nib.load(str(out_dir / "CT.nii.gz"))
    z_spacing = float(np.linalg.norm(img.affine[:3, 2]))
    nifti_extent = (img.shape[2] - 1) * z_spacing

    assert nifti_extent == pytest.approx(dicom_extent, rel=0.01, abs=0.5)


def test_real_eq1_volume_is_chosen_without_any_imagetype_check(collector, tmp_path, irregular_spacing_ct_series_dir):
    """Regression test for BUGREPORT_ct_pt_volume_and_sidecar_selection.md's first finding:
    _select_ct_volume's ``if eq: return eq[0]`` returns before _rank() -- the only PRIMARY/
    SECONDARY ImageType check -- ever runs. The slice-spacing test above already uses this same
    real series (it is dcm2niix's own trigger for an _Eq_1 output) but only checks the resulting
    physical extent, not which file was picked or why -- it would not notice if the ImageType
    check were skipped. This test makes that mechanism itself the assertion, using real
    dcm2niix output rather than a synthetic directory listing:

    1. dcm2niix really does emit two candidates for this series -- a plain volume and its
       gantry-tilt-corrected ``_Eq_1`` sibling.
    2. dcm2niix writes a JSON sidecar (with a valid ``PRIMARY`` ImageType) only for the plain
       volume, never for the ``_Eq_1`` one -- so the chosen file's ImageType is not merely
       unchecked, it is *undiscoverable* to the selection code even if it tried.
    3. `_select_ct_volume` still picks the ``_Eq_1`` file.

    This particular series' `_Eq_1` output happens to be the physically correct volume to keep
    (see the slice-spacing test above and the bug report's "Verified against real data" section),
    so this test does not assert a wrong final image -- there is no competing SECONDARY
    reconstruction here to be wrongly preferred over. It pins down, with real (not synthetic)
    dcm2niix output, that the ImageType check is structurally bypassed whenever an `_Eq_1`
    candidate exists at all, exactly as the bug report describes.
    """
    tmp_out = tmp_path / "dcm2niix_out"
    tmp_out.mkdir()
    run_dcm2niix(irregular_spacing_ct_series_dir, tmp_out)

    nii_files = sorted(tmp_out.glob("*.nii.gz"))
    eq1_files = [f for f in nii_files if f.name.endswith("_Eq_1.nii.gz")]
    plain_files = [f for f in nii_files if f not in eq1_files]
    assert len(eq1_files) == 1, f"expected dcm2niix to also emit an _Eq_1 file, got: {[f.name for f in nii_files]}"
    assert len(plain_files) == 1

    eq1_sidecar = eq1_files[0].with_suffix("").with_suffix(".json")
    plain_sidecar = plain_files[0].with_suffix("").with_suffix(".json")
    assert not eq1_sidecar.is_file(), "expected dcm2niix to NOT write its own sidecar for the _Eq_1 file"
    assert plain_sidecar.is_file()
    with open(plain_sidecar) as f:
        plain_image_type = json.load(f)["ImageType"]
    assert "PRIMARY" in plain_image_type  # a legitimate, ImageType-tagged PRIMARY alternative does exist

    chosen = collector._select_ct_volume(tmp_out, irregular_spacing_ct_series_dir)

    # Chosen despite having strictly less ImageType information available than the alternative
    # (none at all, vs. a confirmed PRIMARY) -- proof the ranking never ran, not that it ran and
    # correctly preferred the Eq_1 file.
    assert chosen == eq1_files[0]
