import json
import logging
import os

import cc3d

from . import metrics, utils
from .utils import load_response_config, make_json_safe

logger = logging.getLogger(__name__)

# recip categories
CR, PR, SD, PD = 1, 2, 3, 4
RECIP_LABELS = {CR: "CR", PR: "PR", SD: "SD", PD: "PD"}

# CADS v1.0.0 combined label map (labelmap_all_structure). CTcads.nii.gz stores these integer
# labels; the map is needed to resolve them to structure names for T/N/M classification. It is a
# fixed property of the model — keep the clinical T/N/M grouping in config.yaml, not here.
CADS_LABELS = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "lung_upper_lobe_left",
    14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right",
    16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right",
    18: "vertebrae_L5",
    19: "vertebrae_L4",
    20: "vertebrae_L3",
    21: "vertebrae_L2",
    22: "vertebrae_L1",
    23: "vertebrae_T12",
    24: "vertebrae_T11",
    25: "vertebrae_T10",
    26: "vertebrae_T9",
    27: "vertebrae_T8",
    28: "vertebrae_T7",
    29: "vertebrae_T6",
    30: "vertebrae_T5",
    31: "vertebrae_T4",
    32: "vertebrae_T3",
    33: "vertebrae_T2",
    34: "vertebrae_T1",
    35: "vertebrae_C7",
    36: "vertebrae_C6",
    37: "vertebrae_C5",
    38: "vertebrae_C4",
    39: "vertebrae_C3",
    40: "vertebrae_C2",
    41: "vertebrae_C1",
    42: "esophagus",
    43: "trachea",
    44: "heart_myocardium",
    45: "heart_atrium_left",
    46: "heart_ventricle_left",
    47: "heart_atrium_right",
    48: "heart_ventricle_right",
    49: "pulmonary_artery",
    50: "brain",
    51: "iliac_artery_left",
    52: "iliac_artery_right",
    53: "iliac_vena_left",
    54: "iliac_vena_right",
    55: "small_bowel",
    56: "duodenum",
    57: "colon",
    58: "urinary_bladder",
    59: "face",
    60: "humerus_left",
    61: "humerus_right",
    62: "scapula_left",
    63: "scapula_right",
    64: "clavicula_left",
    65: "clavicula_right",
    66: "femur_left",
    67: "femur_right",
    68: "hip_left",
    69: "hip_right",
    70: "sacrum",
    71: "gluteus_maximus_left",
    72: "gluteus_maximus_right",
    73: "gluteus_medius_left",
    74: "gluteus_medius_right",
    75: "gluteus_minimus_left",
    76: "gluteus_minimus_right",
    77: "autochthon_left",
    78: "autochthon_right",
    79: "iliopsoas_left",
    80: "iliopsoas_right",
    81: "rib_left_1",
    82: "rib_left_2",
    83: "rib_left_3",
    84: "rib_left_4",
    85: "rib_left_5",
    86: "rib_left_6",
    87: "rib_left_7",
    88: "rib_left_8",
    89: "rib_left_9",
    90: "rib_left_10",
    91: "rib_left_11",
    92: "rib_left_12",
    93: "rib_right_1",
    94: "rib_right_2",
    95: "rib_right_3",
    96: "rib_right_4",
    97: "rib_right_5",
    98: "rib_right_6",
    99: "rib_right_7",
    100: "rib_right_8",
    101: "rib_right_9",
    102: "rib_right_10",
    103: "rib_right_11",
    104: "rib_right_12",
    105: "spinal_canal",
    106: "larynx",
    107: "heart",
    108: "bowel_bag",  # bowel space
    109: "sigmoid",
    110: "rectum",
    111: "prostate",
    112: "seminal_vesicle",
    113: "left_mammary_gland",
    114: "right_mammary_gland",
    115: "sternum",
    116: "right psoas major",
    117: "left psoas major",
    118: "right rectus abdominis",
    119: "left rectus abdominis",
    120: "white matter",
    121: "gray matter",
    122: "csf",
    123: "scalp",
    124: "eye balls",
    125: "compact bone",
    126: "spongy bone",
    127: "blood",
    128: "head muscles",
    129: "OAR_A_Carotid_L",
    130: "OAR_A_Carotid_R",
    131: "OAR_Arytenoid",
    132: "OAR_Bone_Mandible",
    133: "OAR_Brainstem",
    134: "OAR_BuccalMucosa",
    135: "OAR_Cavity_Oral",
    136: "OAR_Cochlea_L",
    137: "OAR_Cochlea_R",
    138: "OAR_Cricopharyngeus",
    139: "OAR_Esophagus_S",
    140: "OAR_Eye_AL",
    141: "OAR_Eye_AR",
    142: "OAR_Eye_PL",
    143: "OAR_Eye_PR",
    144: "OAR_Glnd_Lacrimal_L",
    145: "OAR_Glnd_Lacrimal_R",
    146: "OAR_Glnd_Submand_L",
    147: "OAR_Glnd_Submand_R",
    148: "OAR_Glnd_Thyroid",
    149: "OAR_Glottis",
    150: "OAR_Larynx_SG",
    151: "OAR_Lips",
    152: "OAR_OpticChiasm",
    153: "OAR_OpticNrv_L",
    154: "OAR_OpticNrv_R",
    155: "OAR_Parotid_L",
    156: "OAR_Parotid_R",
    157: "OAR_Pituitary",
    158: "subcutaneous_tissue",
    159: "muscle",
    160: "abdominal_cavity",
    161: "thoracic_cavity",
    162: "bones",
    163: "glands",
    164: "pericardium",
    165: "breast_implant",
    166: "mediastinum",
    167: "spinal_cord",
}


def _num(value) -> float | None:
    """Coerce a stored metric to float, mapping None/""/"NAN" to None."""
    if value is None:
        return None
    if isinstance(value, str) and (value.strip() == "" or value.strip().upper() == "NAN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_overlap_tnm(overlap: dict, tnm_cfg: dict) -> str:
    """Classify one lesion as 'T', 'N', or 'M' from the CADS structures it overlaps.

    - T (primary) if it overlaps any structure in the configured T list (takes precedence).
    - M (distant) if it overlaps any solid organ or bone, i.e. any structure that is NOT in
      the NODAL_PASSTHROUGH list (vessels, muscle, body cavities, unspecific soft tissue).
    - N (nodal) otherwise: a lesion overlapping only pass-through structures, or nothing
      segmented at all. CADS has no lymph-node labels, so this is how nodal disease is caught.

    Matching is case-insensitive substring matching of the configured lists against the
    overlapping structure names.
    """
    organs = {name.lower() for name, present in (overlap or {}).items() if present}
    t_subs = [s.lower() for s in (tnm_cfg.get("T") or [])]
    if any(sub in organ for organ in organs for sub in t_subs):
        return "T"
    passthrough = [s.lower() for s in (tnm_cfg.get("NODAL_PASSTHROUGH") or [])]
    overlaps_organ_or_bone = any(not any(sub in organ for sub in passthrough) for organ in organs)
    return "M" if overlaps_organ_or_bone else "N"


def classify_study_tnm(study_dir: str | os.PathLike, petseg_fname: str, tnm_cfg: dict) -> dict[str, int | None]:
    """Count lesions per metabolic T/N/M class for a single timepoint, using CADS.

    Requires CTcads.nii.gz (CADS segmentation) in the study directory — T/N/M classification
    is defined against CADS structures, not TotalSegmentator. Resamples CADS into PET space,
    runs connected-component analysis on the PETseg mask, and classifies each lesion by the
    CADS structures it overlaps. Returns Nones if CADS or the PETseg mask is unavailable.
    """
    ctcads_path = os.path.join(study_dir, "CTcads.nii.gz")
    petseg_path = os.path.join(study_dir, petseg_fname)
    pet_path = os.path.join(study_dir, "PET.nii.gz")
    null = {"T": None, "N": None, "M": None}

    if not os.path.isfile(ctcads_path):
        logger.warning(
            "CTcads.nii.gz missing in %s — T/N/M requires CADS. miT/miN/miM set to null; run the 'cads' task first.",
            study_dir,
        )
        return null
    if not os.path.isfile(petseg_path) or not os.path.isfile(pet_path):
        logger.warning(
            "%s or PET.nii.gz missing in %s; cannot classify lesions. miT/miN/miM set to null.",
            petseg_fname,
            study_dir,
        )
        return null

    # Resample CADS into PET space (cached) so it aligns with the PETseg lesion mask.
    ctcadsres_path = os.path.join(study_dir, "CTcadsres.nii.gz")
    if not os.path.isfile(ctcadsres_path):
        utils.resample_image(
            source_img=ctcads_path,
            target_img=pet_path,
            nii_output_dirpath=study_dir,
            output_fname="CTcadsres.nii.gz",
            interpolation="nearest",
            fill_value=0,
        )

    petseg_array = metrics.get_3darray_from_niftipath(petseg_path)
    cads_array = metrics.get_3darray_from_niftipath(ctcadsres_path)
    labeled, num_lesions = cc3d.connected_components(petseg_array, connectivity=26, return_N=True)

    counts = {"T": 0, "N": 0, "M": 0}
    for i in range(1, num_lesions + 1):
        tumor_mask = labeled == i
        overlap = utils.compute_tumor_organ_overlap(tumor_mask, cads_array, CADS_LABELS)
        cls = classify_overlap_tnm(overlap, tnm_cfg)
        counts[cls if cls in counts else "M"] += 1
    return counts


def count_new_lesions(study_dir_tp1: str | os.PathLike, study_dir_tp2: str | os.PathLike) -> int | None:
    """Count lesions present at follow-up (TP2) but absent at baseline (TP1).

    TODO: implement registration-based lesion matching. The two PET/CTs are acquired with
    the patient positioned differently, so the TP1 and TP2 PETseg masks do NOT share a
    coordinate frame and cannot simply be overlaid. The correct approach is to (deformably)
    register TP2 to TP1 (e.g. CT->CT), warp the TP2 lesion mask into TP1 space, and flag
    TP2 lesions that have no spatial correspondence in TP1. Until that registration step
    exists, this returns None and recip will not reflect new-lesion progression.
    """
    # Placeholder: requires cross-timepoint registration (see docstring). Intentionally
    # not the earlier organ-overlap heuristic, which is too coarse to be trusted.
    return None


def percist_response(
    sulpeak1: float | None,
    sulpeak2: float | None,
    new_lesions: int | None,
    tmtv2: float | None,
    lesion_count2: float | None,
    tlg1: float | None,
    tlg2: float | None,
    cfg: dict,
) -> int | None:
    """Apply PERCIST 1.0 to assign a recip category (1=CR, 2=PR, 3=SD, 4=PD).

    SULpeak of the most active lesion drives the assessment. New lesions or a marked TLG
    increase count as progression. Returns None when SULpeak is unavailable and the
    response cannot be assessed.
    """
    pct = cfg.get("percent_threshold", 30.0)
    abs_th = cfg.get("absolute_threshold", 0.8)
    tlg_pct = cfg.get("tlg_progression_pct", 75.0)

    no_disease_tp2 = (lesion_count2 in (0, None)) and (tmtv2 in (0, None))

    # New lesions are unequivocal progression.
    if new_lesions:
        return PD
    # Complete metabolic response: no residual measurable disease and no new lesions.
    if no_disease_tp2:
        return CR
    # Beyond this point SULpeak at both timepoints is required.
    if not sulpeak1 or sulpeak2 is None:
        return None

    delta = sulpeak2 - sulpeak1
    delta_pct = delta / sulpeak1 * 100.0
    tlg_increase_pct = ((tlg2 - tlg1) / tlg1 * 100.0) if (tlg1 and tlg2 is not None) else None

    # Progressive metabolic disease.
    if (delta_pct >= pct and delta >= abs_th) or (tlg_increase_pct is not None and tlg_increase_pct >= tlg_pct):
        return PD
    # Partial metabolic response.
    if delta_pct <= -pct and -delta >= abs_th:
        return PR
    # Stable metabolic disease.
    return SD


class ResponseExtractor:
    def __init__(self, input_dirpath_processed: str | os.PathLike, pet_metric: str = "SUL") -> None:
        """Longitudinal PET response assessment across exactly two PET/CT timepoints per patient.

        Identifies the two earliest studies that carry tumor statistics (baseline = TP1,
        follow-up = TP2), classifies lesions into metabolic T/N/M against CADS, applies
        PERCIST 1.0, and writes a patient-level "ResponseAssessment" block back into
        patient_info.json. T/N/M classification requires CADS (CTcads.nii.gz).

        Args:
            input_dirpath_processed (str | os.PathLike): Processed output tree containing the
                per-patient patient_info.json files. Can be nested.
            pet_metric (str): Metric selecting the PETseg mask and stats feeding the PERCIST
                extent checks. Accepts "SUV" or "SUL" (default "SUL"). recip always uses
                SULpeak (PERCIST is lean-body-mass based) and requires the SUL stats regardless.
        """
        if pet_metric not in ("SUV", "SUL"):
            raise ValueError(f"pet_metric must be 'SUV' or 'SUL', got '{pet_metric}'")
        self.input_dirpath = input_dirpath_processed
        self.pet_metric = pet_metric
        self.petseg_fname = "PETseg.nii.gz" if pet_metric == "SUV" else "PETsegSUL.nii.gz"
        cfg = load_response_config()
        self.tnm_cfg = cfg.get("TNM_CLASSIFICATION", {})
        self.percist_cfg = cfg.get("PERCIST", {})

    def run(self) -> None:
        patient_info_paths = [
            os.path.join(dirpath, "patient_info.json")
            for dirpath, _, filenames in os.walk(self.input_dirpath)
            if "patient_info.json" in filenames
        ]
        if not patient_info_paths:
            logger.warning("No patient_info.json files found under %s. Nothing to assess.", self.input_dirpath)
            return
        logger.info("Running longitudinal response assessment on %d patients.", len(patient_info_paths))
        for patient_info_path in sorted(patient_info_paths):
            try:
                self.process_patient(patient_info_path)
            except Exception as e:
                logger.error("Failed response assessment for %s: %s", patient_info_path, e)

    def process_patient(self, patient_info_path: str | os.PathLike) -> None:
        with open(patient_info_path) as f:
            patient_info = json.load(f)

        patient_dir = os.path.dirname(patient_info_path)
        patient_id = patient_info.get("PatientID", "Unknown")
        stats_key = "TumorStats" if self.pet_metric == "SUV" else "TumorStatsSUL"

        studies = patient_info.get("Studies", {})
        dated = sorted((date, study) for date, study in studies.items() if stats_key in study)
        if len(dated) < 2:
            logger.warning(
                "Patient %s: found %d study/studies with %s; need 2. Skipping. "
                "Run the 'radiomics' (and 'tumor') tasks first.",
                patient_id,
                len(dated),
                stats_key,
            )
            return
        if len(dated) > 2:
            logger.warning(
                "Patient %s: %d studies carry %s; using the two earliest (%s, %s).",
                patient_id,
                len(dated),
                stats_key,
                dated[0][0],
                dated[1][0],
            )

        (date1, study1), (date2, study2) = dated[0], dated[1]
        s1, s2 = study1[stats_key], study2[stats_key]
        study_dir1, study_dir2 = os.path.join(patient_dir, date1), os.path.join(patient_dir, date2)

        tnm1 = classify_study_tnm(study_dir1, self.petseg_fname, self.tnm_cfg)
        tnm2 = classify_study_tnm(study_dir2, self.petseg_fname, self.tnm_cfg)
        new_lesions = count_new_lesions(study_dir1, study_dir2)

        # recip is PERCIST and therefore always SUL-based, independent of self.pet_metric.
        sul1 = study1.get("TumorStatsSUL", {})
        sul2 = study2.get("TumorStatsSUL", {})
        sulpeak1, sulpeak2 = _num(sul1.get("SUVpeak")), _num(sul2.get("SUVpeak"))
        if not sul1 or not sul2:
            logger.warning(
                "Patient %s: SUL stats missing for a timepoint; recip cannot follow PERCIST. "
                "Run the muscle_fat + autopet + radiomics tasks with SUL.",
                patient_id,
            )
        if new_lesions is None:
            logger.info(
                "Patient %s: new-lesion count unavailable; recip will not reflect new-lesion progression.",
                patient_id,
            )

        recip = percist_response(
            sulpeak1=sulpeak1,
            sulpeak2=sulpeak2,
            new_lesions=new_lesions,
            tmtv2=_num(s2.get("TMTV")),
            lesion_count2=_num(s2.get("LesionCount")),
            tlg1=_num(s1.get("TLG")),
            tlg2=_num(s2.get("TLG")),
            cfg=self.percist_cfg,
        )

        # Descriptive per-timepoint metrics (ttv/suvmax/suvmean/TLA/SULpeak) are NOT duplicated
        # here — they already live in each study's TumorStats(SUL) from the radiomics task. This
        # block only holds the genuinely longitudinal/derived values.
        response_assessment = {
            "criteria": "PERCIST 1.0",
            "pet_metric": self.pet_metric,
            "tp1_date": date1,
            "tp2_date": date2,
            "miT1": tnm1["T"],
            "miN1": tnm1["N"],
            "miM1": tnm1["M"],
            "miT2": tnm2["T"],
            "miN2": tnm2["N"],
            "miM2": tnm2["M"],
            "newlesions2": new_lesions,
            "recip": recip,
            "recip_label": RECIP_LABELS.get(recip),
        }

        patient_info["ResponseAssessment"] = response_assessment
        with open(patient_info_path, "w") as f:
            json.dump(make_json_safe(patient_info), f)

        logger.info(
            "Patient %s: %s vs %s -> recip=%s (%s), newlesions2=%s.",
            patient_id,
            date1,
            date2,
            recip,
            RECIP_LABELS.get(recip),
            new_lesions,
        )


def response_extraction_entrypoint() -> None:
    """Entry point to run the longitudinal response assessment without the full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.response_extraction")

    import argparse

    parser = argparse.ArgumentParser(
        description="Longitudinal PET response assessment (PERCIST 1.0) across two timepoints per patient."
    )
    parser.add_argument(
        "--input-dirpath-processed",
        type=str,
        help="Path to the processed output tree containing patient_info.json files.",
        required=True,
    )
    parser.add_argument(
        "--pet-metric",
        type=str,
        choices=["SUV", "SUL"],
        default="SUL",
        help="Metric selecting the PETseg mask and stats for the PERCIST extent checks "
        "(default: SUL). recip always uses SULpeak per PERCIST.",
    )
    args = parser.parse_args()

    ResponseExtractor(
        input_dirpath_processed=args.input_dirpath_processed,
        pet_metric=args.pet_metric,
    ).run()


if __name__ == "__main__":
    response_extraction_entrypoint()
