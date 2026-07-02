import logging
import os
import pathlib as plb
import re
import subprocess
import sys
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
from pydicom.valuerep import IS, DSdecimal, DSfloat, PersonName
from skimage.measure import label

from . import metrics

logger = logging.getLogger(__name__)


def natural_key(s: str):
    """Helper function for natural sorting, e.g. mp_2 before mp_10"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


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

    provided = [
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

    # An explicitly empty keyword list (argparse nargs="*" yields [] when a keyword flag is
    # passed without values — distinct from None = flag absent) means "disable keyword
    # filtering and select every series". Returning all-empty lists makes find_default_indices
    # take its "use all series" branch. Useful for anonymized cohorts whose Series/Study
    # Description tags are empty, so keyword matching can never select anything.
    if any(isinstance(k, list) and len(k) == 0 for k in provided):
        logger.warning("Empty keyword(s) provided — disabling keyword filtering; all series will be selected.")
        return {m: {"PRIMARY": [], "SECONDARY": [], "EXCLUSION": []} for m in ("CT", "PT", "MR")}

    # Always load config defaults so partially-specified keywords can be backfilled
    # (referencing default_keywords below would otherwise be undefined when only some are given).
    config_path = plb.Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    with open(config_path) as file:
        default_keywords = yaml.safe_load(file)["SERIES_KEYWORDS"]

    if not any(provided):
        logger.warning("No series keywords provided. Using default keywords from config.yaml.")

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


def load_mr_keywords() -> dict[str, list[str]]:
    """Load MR PRIMARY, SECONDARY, and EXCLUSION keywords from config.yaml."""
    config_path = plb.Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)["SERIES_KEYWORDS"]["MR"]


def is_mr_filename(filename: str, mr_keywords: dict[str, list[str]]) -> bool:
    """Return True if filename matches MR inclusion keywords and no exclusion keywords."""
    fname_lower = filename.lower()
    inclusion = mr_keywords["PRIMARY"] + mr_keywords["SECONDARY"]
    return any(kw in fname_lower for kw in inclusion) and not any(kw in fname_lower for kw in mr_keywords["EXCLUSION"])


def conv_time(time_str: str) -> float:
    # function for time conversion in DICOM tag
    return float(time_str[:2]) * 3600 + float(time_str[2:4]) * 60 + float(time_str[4:13])


def time_to_seconds(t: str | float | int) -> float:
    """
    Converts time as str to float to seconds after 00:00,

    Args:
        t (str | float | int): Time as str in the formate of HHMMSS.MS or HH:MM:SS.MS.

    Returns:
        float: Time in seconds.
    """
    t = str(t).strip()

    if ":" in t:
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    h = int(t[0:2])
    m = int(t[2:4])
    s = float(t[4:])
    return h * 3600 + m * 60 + s


def create_logger(name=None) -> logging.Logger:
    """Instantiates a logger with two h andlers: one for file output and one for console output."""
    os.makedirs("./logger", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"./logger/musiq_{datetime.now().strftime('%Y-%m-%d-%H-%M')}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(name)


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


def repair_ct_affine_from_dicom(
    nifti_path: str | os.PathLike,
    dicom_dirpath: str | os.PathLike,
    rel_tol: float = 0.01,
) -> bool:
    """Fix a CT NIfTI whose through-plane (slice) geometry was mis-derived by dcm2niix.

    Some CT series omit ``SpacingBetweenSlices`` (0018,0088) — observed on Siemens NAEOTOM
    Alpha photon-counting VMI reconstructions. dcm2niix then falls back to ``SliceThickness``
    (0018,0050) for the slice spacing and can also pick the wrong superior-inferior sign,
    producing a volume that is both stretched and flipped head-for-feet ("upside down").

    The DICOM ``ImagePositionPatient`` values are reliable, so this recomputes the slice-axis
    column of the affine directly from them, leaves the (correct) in-plane axes and the voxel
    data untouched, and rewrites the file only when the existing affine actually disagrees —
    wrong sign or spacing off by more than ``rel_tol``. Oblique / gantry-tilted series (where
    dcm2niix's slice vector is legitimately not along the pure slice normal) are left alone.

    Returns True if the file was repaired, False if it was already consistent or unverifiable.
    """
    nifti_path = str(nifti_path)
    img = nib.load(nifti_path)
    if img.ndim < 3 or img.shape[2] < 2:
        return False  # nothing to verify for a single slice

    # Collect each slice's position projected onto the slice normal (in DICOM LPS space).
    normal_lps = None
    projections = []
    for entry in os.scandir(dicom_dirpath):
        if not entry.is_file():
            continue
        try:
            ds = pydicom.dcmread(entry.path, stop_before_pixels=True)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        except Exception:
            continue
        if normal_lps is None:
            normal_lps = np.cross(iop[:3], iop[3:6])
            norm = np.linalg.norm(normal_lps)
            if norm == 0:
                return False
            normal_lps = normal_lps / norm
        projections.append(float(ipp @ normal_lps))

    if normal_lps is None or len(projections) < 2:
        return False

    proj_min, proj_max = min(projections), max(projections)
    spacing_geom = (proj_max - proj_min) / (len(projections) - 1)
    if spacing_geom <= 0:
        return False

    # dcm2niix stores LPS as RAS by negating x and y; the slice normal transforms the same way.
    normal_ras = np.array([-normal_lps[0], -normal_lps[1], normal_lps[2]])
    current_col = np.asarray(img.affine[:3, 2], dtype=float)
    current_norm = np.linalg.norm(current_col)
    if current_norm == 0:
        return False
    # Only touch plain axial-along-normal series; skip oblique/sheared geometry.
    if abs(float(current_col @ normal_ras) / current_norm) < 0.999:
        return False

    # The affine origin is voxel (0,0,0) = the slice at array index 0. Whichever geometric
    # end it sits at tells us which way the slice axis runs.
    origin_ras = np.asarray(img.affine[:3, 3], dtype=float)
    proj0 = float(np.array([-origin_ras[0], -origin_ras[1], origin_ras[2]]) @ normal_lps)
    direction = 1.0 if abs(proj0 - proj_min) <= abs(proj0 - proj_max) else -1.0

    slice_vec_lps = direction * spacing_geom * normal_lps
    slice_vec_ras = np.array([-slice_vec_lps[0], -slice_vec_lps[1], slice_vec_lps[2]])

    if np.allclose(current_col, slice_vec_ras, rtol=rel_tol, atol=1e-3):
        return False  # geometry already correct

    new_affine = np.array(img.affine, dtype=float)
    new_affine[:3, 2] = slice_vec_ras
    logger.warning(
        f"Repairing CT affine for {nifti_path}: slice axis {current_col.round(3).tolist()} "
        f"-> {slice_vec_ras.round(3).tolist()} (dcm2niix used SliceThickness/wrong sign; "
        f"true slice spacing {spacing_geom:.3f} mm derived from ImagePositionPatient)."
    )
    repaired = nib.Nifti1Image(np.asanyarray(img.dataobj), new_affine, img.header)
    repaired.set_sform(new_affine, code=1)
    repaired.set_qform(new_affine, code=1)
    nib.save(repaired, nifti_path)
    return True


def run_dcm2niix(input_folder: str | os.PathLike, output_folder: str | os.PathLike, merge: bool = False) -> None:
    """Run dcm2niix.

    merge=True adds ``-m y``, which tells dcm2niix to merge slices it would otherwise split
    into separate files (e.g. the timepoints of a dynamic/DCE series), yielding a single 4D
    NIfTI instead of one 3D file per timepoint. Used for MR; left off for CT/PET to preserve
    their existing single-volume behavior.
    """
    try:
        command = ["dcm2niix", "-z", "y", "-f", "%p_%s", "-b", "y", "-ba", "n"]
        if merge:
            command += ["-m", "y"]
        command += ["-o", output_folder, input_folder]

        # Execute the command
        subprocess.run(command, check=True)

    except subprocess.CalledProcessError as e:
        logger.error(f"Error during dcm2niix: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


def normalize_dcm2niix_name(name: str | None) -> str:
    """Normalize a string the way dcm2niix sanitizes filenames.

    dcm2niix builds output filenames from the `%p` token by replacing characters that are
    not filename-safe with underscores. Its exact rules vary across versions, so we
    normalize defensively: lowercase and collapse every run of non-alphanumeric characters
    into a single underscore. Applying this to both the source tag and to candidate
    filenames makes matching robust to spaces, parentheses, slashes, dots, etc.
    """
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def find_mr_niftis(
    study_dir: plb.Path, protocol_name: str | None, series_description: str | None = None
) -> list[plb.Path]:
    """Find NIfTIs already produced from an MR series by dcm2niix.

    dcm2niix is run with `-f %p_%s` (see run_dcm2niix), so an MR series is written as
    ``{%p}_{SeriesNumber}.nii.gz``. The `%p` token is ProtocolName, but dcm2niix falls back
    to SeriesDescription when ProtocolName is absent/empty — which is the case for our MR
    DICOMs (the 0018,1030 tag is not present, yet filenames clearly track SeriesDescription).
    So the source stem is ``ProtocolName if non-empty else SeriesDescription``. We match that
    normalized stem followed by the series-number token (``_<digits>``).

    Side-project artifacts that share these study dirs (NIfTIs whose name starts with the
    patient_id, e.g. ``mp_0008_ttp.nii.gz``) are excluded — real dcm2niix MR outputs are
    named from the series tag and never start with the patient_id. Returns matches
    shortest-name-first so the source NIfTI is preferred over derived files that share its
    prefix.
    """
    stem = normalize_dcm2niix_name(protocol_name) or normalize_dcm2niix_name(series_description)
    if not stem:
        return []
    patient_id = study_dir.parent.name
    matches = [
        f
        for f in study_dir.glob("*.nii.gz")
        if not f.name.startswith(patient_id)
        and re.match(rf"^{re.escape(stem)}_\d", normalize_dcm2niix_name(f.name.removesuffix(".nii.gz")))
    ]
    return sorted(matches, key=lambda f: len(f.name))


def mr_nifti_exists(study_dir: plb.Path, protocol_name: str | None, series_description: str | None = None) -> bool:
    """Return True if an MR NIfTI for this series already exists in study_dir."""
    return bool(find_mr_niftis(study_dir, protocol_name, series_description))


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


def calculate_suv_factor(total_dose: float, start_time: str, half_life: float, acq_time: str, weight: float) -> float:
    """Calculation of the SUV conversion factor"""
    time_diff = time_to_seconds(acq_time) - time_to_seconds(start_time)
    act_dose = total_dose * 0.5 ** (time_diff / half_life)
    return 1000 * weight / act_dose


def convert_pet(pet, suv_factor) -> nib.Nifti1Image:
    """Conversion of PET values to SUV (should work on Siemens PET/CT)"""
    pet_suv_data = (pet.get_fdata() * suv_factor).astype(np.float32)
    return nib.Nifti1Image(pet_suv_data, pet.affine)  # type: ignore


def make_json_safe(obj: Any) -> Any:
    """Convert a DICOM or NumPy object to a JSON-safe format."""
    string_types = PersonName | UID

    if isinstance(obj, MultiValue):
        return [make_json_safe(item) for item in obj]
    # DICOM DS (Decimal String) and IS (Integer String) are numeric subclasses (float/int).
    # Keep them numeric instead of stringifying, so downstream arithmetic (e.g. PatientWeight
    # used to compute LBM) doesn't break on values like "80.0".
    elif isinstance(obj, IS):
        return int(obj)
    elif isinstance(obj, DSfloat | DSdecimal):
        return float(obj)
    elif isinstance(obj, string_types):
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
