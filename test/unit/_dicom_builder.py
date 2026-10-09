"""Helper for writing minimal-but-valid synthetic DICOM series to disk.

Used by tests that exercise the real ``dcm2niix`` binary (installed as the ``dcm2niix``
PyPI package, see pyproject.toml) end-to-end, instead of mocking ``run_dcm2niix``. Real DICOM
downloads (e.g. from TCIA) rarely carry all the tags a specific test branch needs (a
Radiopharmaceutical sequence, a particular DecayCorrection flag, ...), so tests that need exact
control over those tags build a tiny synthetic series here rather than mocking the conversion
away entirely.
"""

import os

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

_SOP_CLASS_BY_MODALITY = {
    "CT": pydicom.uid.CTImageStorage,
    "MR": pydicom.uid.MRImageStorage,
    "PT": pydicom.uid.PositronEmissionTomographyImageStorage,
}


def write_dicom_series(
    out_dir,
    modality,
    n_slices=5,
    rows=8,
    cols=8,
    pixel_value=lambda i: i * 10,
    slice_spacing=2.0,
    pixel_spacing=(1.0, 1.0),
    series_uid=None,
    study_uid=None,
    patient_id="TESTPAT",
    series_description="synthetic series",
    protocol_name=None,
    series_number=1,
    rescale_slope=1,
    rescale_intercept=0,
    pixel_representation=1,
    extra_tags=None,
    per_slice_tags=None,
):
    """Write ``n_slices`` DICOM files for one series into ``out_dir`` and return it.

    ``extra_tags``: dict of tag -> value applied to every slice (e.g. PET radiopharmaceutical
    info). ``per_slice_tags``: callable ``(index) -> dict`` for values that vary by slice
    (e.g. AcquisitionTime for a dynamic series).
    """
    os.makedirs(out_dir, exist_ok=True)
    series_uid = series_uid or generate_uid()
    study_uid = study_uid or generate_uid()

    for i in range(n_slices):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = _SOP_CLASS_BY_MODALITY[modality]
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.Modality = modality
        ds.PatientName = "Test^Pat"
        ds.PatientID = patient_id
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.StudyDate = "20240101"
        ds.SeriesDate = "20240101"
        ds.StudyID = "1"
        ds.SeriesNumber = series_number
        ds.SeriesDescription = series_description
        if protocol_name is not None:
            ds.ProtocolName = protocol_name
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(i * slice_spacing)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = list(pixel_spacing)
        ds.SliceThickness = slice_spacing
        ds.SpacingBetweenSlices = slice_spacing
        ds.Rows = rows
        ds.Columns = cols
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = pixel_representation
        ds.RescaleSlope = rescale_slope
        ds.RescaleIntercept = rescale_intercept

        if extra_tags:
            for tag, value in extra_tags.items():
                setattr(ds, tag, value)
        if per_slice_tags:
            for tag, value in per_slice_tags(i).items():
                setattr(ds, tag, value)

        dtype = np.int16 if pixel_representation == 1 else np.uint16
        arr = np.full((rows, cols), pixel_value(i), dtype=dtype)
        ds.PixelData = arr.tobytes()

        ds.save_as(os.path.join(str(out_dir), f"slice_{i:03d}.dcm"), enforce_file_format=True)

    return out_dir


def pet_radiopharm_tags(total_dose=300000000.0, start_time="113000", half_life=6588.0):
    """Build the ``extra_tags`` dict for a minimal PET RadiopharmaceuticalInformationSequence."""
    seq_item = Dataset()
    seq_item.RadionuclideTotalDose = total_dose
    seq_item.RadiopharmaceuticalStartTime = start_time
    seq_item.RadionuclideHalfLife = half_life
    return {"RadiopharmaceuticalInformationSequence": Sequence([seq_item])}
