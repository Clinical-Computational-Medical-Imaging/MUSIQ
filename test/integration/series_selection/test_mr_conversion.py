"""Real DICOM -> NIfTI conversion tests for convert_dcm2nii_MR, against genuine downloaded
TCIA data (see conftest.py for how to point pytest at the local download).
"""

import os
import pathlib as plb

import nibabel as nib
import numpy as np
import pydicom
import pytest

from ..conftest import find_series_dir

pytestmark = [pytest.mark.usefixtures("dcm2niix_available")]


def _dominant_orientation_axis(series_dir) -> int:
    """0/1/2 (X/Y/Z) for the physical axis the slice-normal of the first DICOM in ``series_dir``
    is most aligned with -- i.e. whether the series is sagittal/coronal/axial, read from the
    source DICOM's own ``ImageOrientationPatient`` rather than assumed from the series name."""
    first_file = plb.Path(series_dir) / sorted(os.listdir(series_dir))[0]
    ds = pydicom.dcmread(str(first_file), stop_before_pixels=True)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    return int(np.argmax(np.abs(normal)))


@pytest.fixture()
def t2_coronal_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-EJ-5495",
        study_glob="MRI PELVIS",
        series_glob="T2 CORONAL",
    )


def test_real_mr_series_converts_to_a_plausible_volume(collector, tmp_path, t2_coronal_series_dir):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=t2_coronal_series_dir, output_dirpath=out_dir)

    assert nii_path
    assert str(nii_path).startswith(str(out_dir))
    assert dicom_tags["Modality"] == "MR"

    img = nib.load(str(nii_path))
    assert img.ndim == 3
    data = img.get_fdata()
    assert data.max() > 0  # a real, non-blank MR volume
    assert data.std() > 0


def test_real_mr_conversion_is_skipped_when_output_already_exists(mocker, collector, tmp_path, t2_coronal_series_dir):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    first_path, _ = collector.convert_dcm2nii_MR(MR_dcm_dirpath=t2_coronal_series_dir, output_dirpath=out_dir)
    with open(first_path, "rb") as f:
        original_bytes = f.read()

    run_dcm2niix = mocker.patch("musiq.series_selection.run_dcm2niix")

    second_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=t2_coronal_series_dir, output_dirpath=out_dir)

    run_dcm2niix.assert_not_called()
    assert str(second_path) == str(first_path)
    with open(second_path, "rb") as f:
        assert f.read() == original_bytes
    assert dicom_tags["Modality"] == "MR"


@pytest.fixture()
def t1_axial_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-EJ-5495",
        study_glob="MRI PELVIS",
        series_glob="T1 AXIAL",
    )


@pytest.fixture()
def t2_axial_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-EJ-5495",
        study_glob="MRI PELVIS",
        series_glob="T2 AXIAL",
    )


@pytest.mark.parametrize("contrast", ["t1", "t2"])
def test_real_t1_and_t2_series_both_convert_to_plausible_volumes(
    collector, tmp_path, contrast, t1_axial_series_dir, t2_axial_series_dir
):
    """T1 AXIAL and T2 AXIAL (same patient/study as T2 CORONAL above) each convert
    independently to a real, non-blank 3D volume — covering both major MR contrast types."""
    series_dir = t1_axial_series_dir if contrast == "t1" else t2_axial_series_dir
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=series_dir, output_dirpath=out_dir)

    assert nii_path
    assert dicom_tags["Modality"] == "MR"
    img = nib.load(str(nii_path))
    assert img.ndim == 3
    data = img.get_fdata()
    assert data.max() > 0
    assert data.std() > 0


@pytest.fixture()
def t2_sagittal_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-EJ-5495",
        study_glob="MRI PELVIS",
        series_glob="T2 SAGITTAL",
    )


@pytest.mark.parametrize(
    "view, expected_axis, fixture_name",
    [
        ("axial", 2, "t2_axial_series_dir"),
        ("coronal", 1, "t2_coronal_series_dir"),
        ("sagittal", 0, "t2_sagittal_series_dir"),
    ],
)
def test_real_axial_coronal_and_sagittal_mr_all_convert_correctly(
    request, collector, tmp_path, view, expected_axis, fixture_name
):
    """All three MR planes (same patient/study as the T1/T2 AXIAL and T2 CORONAL tests above)
    must convert to a real, non-blank volume. Also self-checks that each fixture really is the
    claimed view via the source DICOM's own ImageOrientationPatient, so a mislabeled fixture
    wouldn't silently pass as "coverage" of a view it doesn't actually exercise."""
    series_dir = request.getfixturevalue(fixture_name)
    assert _dominant_orientation_axis(series_dir) == expected_axis, f"{series_dir} is not really {view}"

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=series_dir, output_dirpath=out_dir)

    assert nii_path
    assert dicom_tags["Modality"] == "MR"
    img = nib.load(str(nii_path))
    assert img.ndim == 3
    data = img.get_fdata()
    assert data.max() > 0
    assert data.std() > 0


@pytest.mark.xfail(
    reason=(
        "BUG: find_mr_niftis (utils.py) matches an existing NIfTI by ProtocolName + 'any digit' "
        "suffix, not the exact SeriesNumber. TCGA-EJ-5495 sets the same ProtocolName for every "
        "series in the study, so after T1 AXIAL converts, T2 AXIAL matches T1's file via the same "
        "regex and is wrongly treated as already converted — its tags end up attached to T1's "
        "image data. Remove once find_mr_niftis checks the exact SeriesNumber."
    ),
    strict=True,
)
def test_mr_series_sharing_a_protocol_name_are_not_confused_with_each_other(
    collector, tmp_path, t1_axial_series_dir, t2_axial_series_dir
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    t1_path, t1_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=t1_axial_series_dir, output_dirpath=out_dir)
    t2_path, t2_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=t2_axial_series_dir, output_dirpath=out_dir)

    assert t1_tags["SeriesDescription"] == "T1 AXIAL"
    assert t2_tags["SeriesDescription"] == "T2 AXIAL"
    # The actual bug: without the fix, t2_path == t1_path (T2's tags end up pointing at T1's file).
    assert t2_path != t1_path


@pytest.fixture()
def dynamic_scan_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-J4-A67O",
        study_glob="MRI PELVIS WWO C",
        series_glob="DYNAMIC SCAN",
    )


def test_real_multi_timepoint_dynamic_series_merges_into_one_4d_volume(collector, tmp_path, dynamic_scan_series_dir):
    """936 DICOM files for one series (a dynamic/DCE acquisition stored under a single
    SeriesInstanceUID rather than as separate per-timepoint series, with repeated slice
    positions across timepoints) — exercises dcm2niix's own merge (-m y), which reliably
    combines this layout into a single 4D NIfTI rather than one 3D file per timepoint."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=dynamic_scan_series_dir, output_dirpath=out_dir)

    assert nii_path
    img = nib.load(str(nii_path))
    assert img.ndim == 4
    assert img.shape[3] > 1  # genuinely multiple timepoints, not a degenerate singleton 4th axis
    assert dicom_tags["Modality"] == "MR"


@pytest.fixture()
def ax_dwi_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-J4-A67O",
        study_glob="MRI PELVIS WWO C",
        series_glob="Ax DWI B400",
    )


def test_real_diffusion_weighted_mr_series_converts_to_a_plausible_volume(collector, tmp_path, ax_dwi_series_dir):
    """DWI is a materially different MR sequence type from the T1/T2 anatomical contrasts covered
    above (different contrast mechanism, lower in-plane resolution).

    This particular series bundles two b-values (b=0 and b=400) under one SeriesInstanceUID —
    52 files for 26 spatial locations — so dcm2niix itself produces a 4D volume here, not 3D;
    the ``convert_dcm2nii_MR`` ranking (most dimensions wins) must keep that 4D volume rather
    than picking a degenerate 3D fragment of it."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=ax_dwi_series_dir, output_dirpath=out_dir)

    assert nii_path
    assert dicom_tags["Modality"] == "MR"
    img = nib.load(str(nii_path))
    assert img.ndim == 4
    assert img.shape[3] == 2  # the two b-values, not a degenerate singleton 4th axis
    data = img.get_fdata()
    assert data.max() > 0
    assert data.std() > 0


@pytest.fixture()
def adc_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="TCGA-PRAD",
        patient_id="TCGA-J4-A67O",
        study_glob="MRI PELVIS WWO C",
        series_glob="Apparent Diffusion Coefficient",
    )


def test_real_adc_map_converts_to_a_plausible_volume(collector, tmp_path, adc_series_dir):
    """ADC is a scanner-derived quantitative map (mm^2/s), computed from the DWI b-values rather
    than acquired directly -- a different conversion case again from the raw DWI series above:
    a single 3D volume (unlike DWI's bundled-b-values 4D output), reconstructed at the same
    26-slice geometry."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=adc_series_dir, output_dirpath=out_dir)

    assert nii_path
    assert dicom_tags["Modality"] == "MR"
    img = nib.load(str(nii_path))
    assert img.ndim == 3
    data = img.get_fdata()
    assert data.max() > 0
    assert data.std() > 0


@pytest.fixture()
def flair_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="ReMIND",
        patient_id="ReMIND-048",
        study_glob="Preop",
        series_glob="2D_AX_T2_FLAIR",
    )


def test_real_flair_mr_series_converts_to_a_plausible_volume(collector, tmp_path, flair_series_dir):
    """FLAIR is a brain MR contrast none of the TCGA-PRAD (pelvis) series above cover -- a
    different collection (ReMIND, neurosurgical MRI) since no pelvis protocol includes it."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=flair_series_dir, output_dirpath=out_dir)

    assert nii_path
    assert dicom_tags["Modality"] == "MR"
    img = nib.load(str(nii_path))
    assert img.ndim == 3
    data = img.get_fdata()
    assert data.max() > 0
    assert data.std() > 0


@pytest.fixture()
def tracew_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="ACRIN-6698",
        patient_id="ACRIN-6698-207837",
        study_glob="ACRIN-6698_ISPY2_MRI_T0",
        series_glob="TRACEW_DFC",
    )


def test_real_tracew_dwi_series_converts_to_a_plausible_volume(collector, tmp_path, tracew_series_dir):
    """TRACEW (trace-weighted DWI) from a different collection/vendor than the Ax DWI B400 series
    above. This particular series bundles 4 b-values under one SeriesInstanceUID (144 files for
    36 spatial locations), so dcm2niix produces a 4D volume -- same ranking case as DWI
    B400, from an independent real-world example."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=tracew_series_dir, output_dirpath=out_dir)

    assert nii_path
    assert dicom_tags["Modality"] == "MR"
    img = nib.load(str(nii_path))
    assert img.ndim == 4
    assert img.shape[3] == 4  # the four b-values, not a degenerate singleton 4th axis
    data = img.get_fdata()
    assert data.max() > 0
    assert data.std() > 0


@pytest.fixture()
def ideal_fat_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="ISPY2",
        patient_id="ISPY2-100899",
        study_glob="ISPY2_MRI_T0",
        series_glob="FAT",
    )


@pytest.fixture()
def ideal_water_series_dir(integration_data_dir):
    return find_series_dir(
        integration_data_dir,
        collection="ISPY2",
        patient_id="ISPY2-100899",
        study_glob="ISPY2_MRI_T0",
        series_glob="WATER",
    )


@pytest.mark.parametrize("component", ["fat", "water"])
def test_real_dixon_style_water_fat_series_convert_to_plausible_volumes(
    request, collector, tmp_path, component, ideal_fat_series_dir, ideal_water_series_dir
):
    """GE's IDEAL water-fat separation is the vendor-specific name for the same underlying
    technique as Siemens' DIXON -- exercises exactly the real-world case the synthetic
    water/fat ranking test in test_convert_dcm2nii_mr.py models: a study yielding multiple
    related component volumes from one acquisition, each of which must convert independently
    to a real, non-blank volume."""
    series_dir = ideal_fat_series_dir if component == "fat" else ideal_water_series_dir
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    nii_path, dicom_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=series_dir, output_dirpath=out_dir)

    assert nii_path
    assert dicom_tags["Modality"] == "MR"
    img = nib.load(str(nii_path))
    assert img.ndim == 3
    data = img.get_fdata()
    assert data.max() > 0
    assert data.std() > 0
