"""Real DICOM -> NIfTI conversion tests for convert_dcm2nii_MR, against genuine downloaded
TCIA data (see conftest.py for how to point pytest at the local download).
"""

import nibabel as nib
import pytest

from .conftest import find_series_dir

pytestmark = [pytest.mark.usefixtures("dcm2niix_available")]


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
