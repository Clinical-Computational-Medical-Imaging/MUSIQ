import logging
import os
import pathlib as plb
import subprocess
from datetime import datetime
from typing import Any

import dicom2nifti
import nibabel as nib
import nilearn.image
import numpy as np
import pydicom
import SimpleITK as sitk
import yaml
from pydicom.multival import MultiValue
from pydicom.uid import UID
from pydicom.valuerep import IS, DSfloat, PersonName
from skimage.measure import label

from . import metrics

logger = logging.getLogger(__name__)


def setup_series_keywords(
    ct_primary_keywords: list[str] | None = None,
    ct_secondary_keywords: list[str] | None = None,
    ct_exclusion_keywords: list[str] | None = None,
    pt_primary_keywords: list[str] | None = None,
    pt_secondary_keywords: list[str] | None = None,
    pt_exclusion_keywords: list[str] | None = None,
    mr_primary_keywords: list[str] | None = None,
    mr_secondary_keywords: list[str] | None = None,
    mr_exclusion_keywords: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    """Setup series keywords for CT, PT, and MR modalities in order to preselect suiting series for further processing.
    If no keywords are provided, it will use the default keywords from the config.yaml file.

    Args:
        ct_primary_keywords (list[str] | None): Keywords for primary selection of CT series.
        ct_secondary_keywords (list[str] | None): Keywords for secondary selection of CT series.
        ct_exclusion_keywords (list[str] | None): Keywords to exclude CT series.
        pt_primary_keywords (list[str] | None): Keywords for primary selection of PT series.
        pt_secondary_keywords (list[str] | None): Keywords for secondary selection of PT series.
        pt_exclusion_keywords (list[str] | None): Keywords to exclude PT series.
        mr_primary_keywords (list[str] | None): Keywords for primary selection of MR series.
        mr_secondary_keywords (list[str] | None): Keywords for secondary selection of MR series.
        mr_exclusion_keywords (list[str] | None): Keywords to exclude MR series.
    Returns:
        dict[str, dict[str, list[str]]]: A dictionary containing the series keywords for CT, PT, and MR modalities.
    """

    if not any(
        [
            ct_primary_keywords,
            ct_secondary_keywords,
            ct_exclusion_keywords,
            pt_primary_keywords,
            pt_secondary_keywords,
            pt_exclusion_keywords,
            mr_primary_keywords,
            mr_secondary_keywords,
            mr_exclusion_keywords,
        ]
    ):
        logger.warning("No series keywords provided. Using default keywords from config.yaml.")
        config_path = plb.Path(__file__).parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file {config_path} does not exist.")

        with open(config_path) as file:
            default_keywords = yaml.safe_load(file)

        default_keywords = default_keywords["SERIE_KEYWORDS"]

    return {
        "CT": {
            "PRIMARY": ct_primary_keywords if ct_primary_keywords else default_keywords["CT"]["PRIMARY"],
            "SECONDARY": ct_secondary_keywords if ct_secondary_keywords else default_keywords["CT"]["SECONDARY"],
            "EXCLUSION": ct_exclusion_keywords if ct_exclusion_keywords else default_keywords["CT"]["EXCLUSION"],
        },
        "PT": {
            "PRIMARY": pt_primary_keywords if pt_primary_keywords else default_keywords["PT"]["PRIMARY"],
            "SECONDARY": pt_secondary_keywords if pt_secondary_keywords else default_keywords["PT"]["SECONDARY"],
            "EXCLUSION": pt_exclusion_keywords if pt_exclusion_keywords else default_keywords["PT"]["EXCLUSION"],
        },
        "MR": {
            "PRIMARY": mr_primary_keywords if mr_primary_keywords else default_keywords["MR"]["PRIMARY"],
            "SECONDARY": mr_secondary_keywords if mr_secondary_keywords else default_keywords["MR"]["SECONDARY"],
            "EXCLUSION": mr_exclusion_keywords if mr_exclusion_keywords else default_keywords["MR"]["EXCLUSION"],
        },
    }


def conv_time(time_str: str) -> float:
    # function for time conversion in DICOM tag
    return float(time_str[:2]) * 3600 + float(time_str[2:4]) * 60 + float(time_str[4:13])


def create_logger() -> logging.Logger:
    """Instantiates a logger with two h andlers: one for file output and one for console output."""
    os.makedirs("./logger", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"./logger/ccmir-radiomics_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def get_spacing_from_niftipath(path: str) -> tuple[float, float, float]:
    """Get the spacing of a NIfTI image."""
    spacing = sitk.ReadImage(path).GetSpacing()
    return spacing


def agnostic_path(*args) -> plb.Path:
    """Create a path that is agnostic to the operating system."""
    raw_path = os.path.join(*map(str, args)).replace("\\", "/")
    return plb.Path(raw_path)


def run_dicom2nifti(input_folder: str | os.PathLike, output_folder: str | os.PathLike) -> None:
    """Convert DICOM files in a directory to NIfTI format using dicom2nifti.

    Args:
        input_folder (str | os.PathLike): Input directory containing DICOM files.
        output_folder (str | os.PathLike): Output directory for NIfTI files.
    """
    ## If dcm2niix fails can try with this library
    try:
        dicom2nifti.convert_directory(input_folder, str(output_folder), compression=True, reorient=True)
        logger.info(f"Converted {input_folder} to {output_folder}")
    except Exception as e:
        logger.error(f"Error converting {input_folder}: {e}")


def run_dcm2niix(input_folder: str | os.PathLike, output_folder: str | os.PathLike) -> None:
    try:
        # Construct the nnUNet predict command
        command = ["dcm2niix", "-z", "y", "-f", "%p_%s", "-b", "y", "-ba", "n", "-o", output_folder, input_folder]

        # Execute the command
        subprocess.run(command, check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during dcm2niix: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


def is_preselected(series) -> bool:
    """Check if the series is preselected based on its description and modality."""
    desc = series["SeriesDescription"]
    modality = series["Modality"]
    return (modality == "CT" and "knochen" in desc) or (
        modality == "PT" and any(x in desc for x in ["pet gk ctac", "qc fx"])
    )


def dcm2nii_mask(mask_dcm_path: str | os.PathLike, nii_output_dirpath: str | os.PathLike) -> None:
    """Convert a DICOM mask file to NIfTI format.

    Args:
        mask_dcm_path (str | os.PathLike): Path to the directory containing the mask DICOM files.
        nii_output_dirpath (str | os.PathLike): Path to the output directory where the NIfTI mask will be saved.
    """
    # conversion of the mask dicom file to nifti (not directly possible with dicom2nifti)
    mask_dcm = list(mask_dcm_path.glob("*.dcm"))[0]
    mask = pydicom.read_file(str(mask_dcm))
    mask_array = mask.pixel_array

    # get mask array to correct orientation (this procedure is dataset specific)
    mask_array = np.transpose(mask_array, (2, 1, 0))
    mask_orientation = mask[0x5200, 0x9229][0].PlaneOrientationSequence[0].ImageOrientationPatient
    if mask_orientation[4] == 1:
        mask_array = np.flip(mask_array, 1)

    # get affine matrix from the corresponding pet
    pet = nib.load(os.path.join(nii_output_dirpath, "PET.nii.gz"))
    pet_affine = pet.affine

    # return mask as nifti object
    mask_out = nib.Nifti1Image(mask_array, pet_affine)
    nib.save(mask_out, os.path.join(nii_output_dirpath, "SEG.nii.gz"))


def resample_image(
    source_img: str | os.PathLike,
    target_img: str | os.PathLike,
    nii_output_dirpath: str | os.PathLike,
    output_fname: str,
    interpolation: str,
    fill_value: int,
) -> None:
    """Resample a source resolution and directly save under a given path.
    For example, to resample a CT to the resolution of a PET.

    Args:
        source_img (str | os.PathLike): Path to the source NIfTI image to be resampled.
        target_img (str | os.PathLike): Path to the target NIfTI image that defines the resolution.
        nii_output_dirpath (str | os.PathLike): Output directory where the resampled CT will be saved.
        output_fname (str): Filename of the resampled CT file.
        interpolation (str): Interpolation method for nilearn. E.g. "nearest", "continuous", etc.
        fill_value (int): Fill value for points outside of the input volume.
    """
    resampled = nilearn.image.resample_to_img(
        source_img=nib.load(source_img),
        target_img=nib.load(target_img),
        fill_value=fill_value,
        interpolation=interpolation,
        force_resample=True,
        copy_header=True,
    )
    nib.save(resampled, os.path.join(nii_output_dirpath, output_fname))


def load_nifti_as_array(path: str) -> tuple[np.ndarray, sitk.Image]:
    """Load a NIfTI file as a numpy array."""
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    return array, image


def compute_connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Perform connected component analysis on a binary mask."""
    labeled_mask = label(mask, connectivity=3)
    return labeled_mask


def compute_tumor_organ_overlap(tumor_mask: np.ndarray, organ_mask: np.ndarray, organ_labels: dict) -> dict:
    """Determine which organs the tumor overlaps with."""
    overlap = {}
    for organ_id, organ_name in organ_labels.items():
        organ_region = organ_mask == int(organ_id)
        if np.any(np.logical_and(tumor_mask, organ_region)):
            overlap[organ_name] = True
    return overlap


def compute_pet_metrics(tumor_mask: np.ndarray, pet_array: np.ndarray, spacing: tuple[float]) -> dict:
    """Compute standard PET metrics for a tumor."""
    tumor_voxels = pet_array[tumor_mask > 0]
    if tumor_voxels.size == 0:
        return {"SUVmean": None, "SUVmax": None, "SUVstd": None}
    return {
        "SUVmean": float(np.mean(tumor_voxels)),
        "SUVmax": float(np.max(tumor_voxels)),
        "SUVstd": float(np.std(tumor_voxels)),
        "SUVpeak": metrics.calculate_suvpeak_median(pet_array, tumor_mask, spacing),
        "SurfaceArea_mm2": metrics.calculate_patient_level_surface_area(tumor_mask, spacing),
    }


def get_modalities_in_dir(directory: str | os.PathLike) -> set:
    """
    Scans the specified directory (and its subdirectories) for DICOM files and
    collects the modalities found. It stops searching early if both "CT" and "PT" are found.
    """
    modalities = set()
    directory = plb.Path(directory)
    for root, _dirs, files in os.walk(directory):
        for file in files:
            file_path = plb.Path(root) / file
            try:
                ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)
                mod = getattr(ds, "Modality", None)
                if mod:
                    modalities.add(mod)
                # If both CT and PT are found, we can stop early.
                if "CT" in modalities and "PT" in modalities:
                    return modalities
            except Exception:
                continue
    return modalities


def extract_dicom_data(dirpath: plb.Path, tags: dict) -> dict[Any, Any]:
    dicom_files = [
        f
        for f in dirpath.iterdir()
        if f.is_file()
        and f.name.lower() != "dicomdir"
        and f.suffix.lower()
        not in [".zip", ".inf", ".jar", ".icns", ".info", ".exe", ".pdf", ".txt", ".ini", ".xml", ".bmp", ".sh"]
        and f.name != ".DS_Store"
    ]
    if not dicom_files:
        return {}

    first_file = dicom_files[0]

    try:
        dcm = pydicom.dcmread(first_file)
        info = {}
        for key, (grp, elem) in tags.items():
            tag = (int(grp, 16), int(elem, 16))
            if tag in dcm:
                value = dcm.get(tag).value
                # Convert PersonName to string
                if hasattr(value, "family_name") or hasattr(value, "given_name"):
                    value = str(value)
                # Convert Patient Age
                if "patientage" in key.lower():
                    value = "".join(filter(str.isdigit, value))
                    value = value.lstrip("0")
                if "radiopharmaceutical" in key.lower() and value in [
                    "FDG",
                    "Fluorodeoxyglucose",
                    "FDG -- Fluorodeoxyglucose",
                    "FDG -- fluorodeoxyglucose",
                ]:
                    value = "FDG"
                info[key] = value
        return info
    except Exception as e:
        logger.error(f"Error processing file {first_file}: {e}\n")
        return {}


def make_json_safe(obj: Any) -> Any:
    """Convert a DICOM or NumPy object to a JSON-safe format."""
    non_serializable_types = MultiValue | PersonName | DSfloat | IS | UID

    if isinstance(obj, MultiValue):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, non_serializable_types):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj  # basic type
