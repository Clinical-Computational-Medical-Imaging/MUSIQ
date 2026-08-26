"""Fixtures shared across the utils/ unit tests."""

import nibabel as nib
import numpy as np
import pytest


@pytest.fixture()
def ct_nifti_factory(tmp_path):
    """Factory writing a synthetic CT-like NIfTI with a controllable slice-axis affine column
    and origin, for exercising ``repair_slice_direction_from_dicom``/``repair_slice_spacing_from_dicom``
    against a known-good/known-bad affine without needing a real dcm2niix conversion.
    """
    counter = {"n": 0}

    def _make(shape=(4, 4, 4), slice_col=(0.0, 0.0, 10.0), origin=(0.0, 0.0, 0.0)):
        counter["n"] += 1
        affine = np.eye(4)
        affine[:3, 2] = slice_col
        affine[:3, 3] = origin
        data = np.zeros(shape, dtype=np.int16)
        path = tmp_path / f"ct_{counter['n']}.nii.gz"
        nib.save(nib.Nifti1Image(data, affine), str(path))
        return str(path)

    return _make
