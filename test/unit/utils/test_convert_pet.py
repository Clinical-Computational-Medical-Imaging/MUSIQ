"""Unit tests for utils.convert_pet: applying a precomputed SUV/SUL factor (from
calculate_suv_factor) to a loaded PET NIfTI to produce the quantitative SUV/SUL image that
series_selection.py and sul_computation.py write to SUV.nii.gz / SUL.nii.gz.

The function itself is a pure, three-line transform -- scale the voxel data by ``suv_factor``,
cast to float32, and rewrap with the original affine -- so these tests pin down each of those
three properties directly rather than only exercising it indirectly through a full conversion
pipeline.
"""

import nibabel as nib
import numpy as np
import pytest

from musiq.utils import convert_pet


def test_scales_pet_data_by_suv_factor():
    data = np.array([[[10, 20], [30, 40]]], dtype=np.int16)
    pet = nib.Nifti1Image(data, affine=np.eye(4))

    result = convert_pet(pet, suv_factor=2.5)

    assert result.get_fdata() == pytest.approx(data.astype(np.float32) * 2.5)


def test_preserves_the_original_affine():
    affine = np.array(
        [
            [1.0, 0.0, 0.0, -50.0],
            [0.0, 1.0, 0.0, -60.0],
            [0.0, 0.0, 2.0, -80.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pet = nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), affine=affine)

    result = convert_pet(pet, suv_factor=1.0)

    assert result.affine == pytest.approx(affine)


def test_output_data_is_cast_to_float32_regardless_of_input_dtype():
    """PET pixels loaded from DICOM are integer-typed; the SUV-converted image must come back
    float32 (precision after a typically sub-unity suv_factor multiply, and consistency for
    downstream radiomics/metrics code), not left in the source dtype."""
    pet = nib.Nifti1Image(np.array([[[100]]], dtype=np.int16), affine=np.eye(4))

    result = convert_pet(pet, suv_factor=0.0005)

    assert np.asanyarray(result.dataobj).dtype == np.float32


def test_suv_factor_of_one_leaves_values_numerically_unchanged():
    data = np.array([[[1, 2], [3, 4]]], dtype=np.int16)
    pet = nib.Nifti1Image(data, affine=np.eye(4))

    result = convert_pet(pet, suv_factor=1.0)

    assert result.get_fdata() == pytest.approx(data.astype(np.float32))


def test_zero_suv_factor_zeroes_out_the_data():
    data = np.array([[[5, 10], [15, 20]]], dtype=np.int16)
    pet = nib.Nifti1Image(data, affine=np.eye(4))

    result = convert_pet(pet, suv_factor=0.0)

    assert result.get_fdata() == pytest.approx(np.zeros_like(data, dtype=np.float32))


def test_does_not_mutate_the_input_image():
    """The scaling must produce a new array (pet.get_fdata() * suv_factor), never write back
    into the source image -- a caller re-reading ``pet`` after conversion must still see the
    original, unscaled data."""
    data = np.array([[[10, 20]]], dtype=np.int16)
    pet = nib.Nifti1Image(data.copy(), affine=np.eye(4))

    convert_pet(pet, suv_factor=3.0)

    assert pet.get_fdata() == pytest.approx(data.astype(np.float64))
