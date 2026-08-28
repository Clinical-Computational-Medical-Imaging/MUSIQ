"""PROMISE V2 prostate cancer PSMA-PET staging for MUSIQ.

Classifies per-lesion data from the 'tumor' task into miTNM categories per
Seifert et al., Eur Urol 2023;83:405-412, and writes PromiseStats to
patient_info.json.

Segmentation source (in preference order):
  1. CTcadsres.nii.gz (CADS) — preferred: has seminal_vesicle separately (→ miT3b),
     granular bone labels, and iliac arteries for pelvic LN laterality.
  2. CTsegres.nii.gz (TotalSegmentator) — fallback when CTcads hasn't been run.

Remaining limitations vs. the full PROMISE V2 spec:
- Parotid SUV reference unavailable (not in TotalSegmentator 'total' task);
  PSMA-expression score 2 means ">liver" and may be 2 or 3.
- Pelvic LN sub-region (II/EI/OB/PS) is approximate from centroid position.
- PRIMARY score (intraprostatic pattern) is not computed (requires pattern reading).
"""

import json
import logging
import os
import pathlib as plb

import cc3d
import nibabel as nib
import numpy as np
from scipy import ndimage

from . import metrics, utils
from .tumor_info_extraction import _CADS_LABEL_NAMES

logger = logging.getLogger(__name__)

# ─── CADS integer label IDs relevant for PROMISE staging ──────────────────────
# (from labelmap_all_structure / _CADS_LABEL_NAMES in tumor_info_extraction.py)

_CADS_PROSTATE = 111
_CADS_SEMINAL_VESICLE = 112
_CADS_BLADDER = 58
_CADS_RECTUM = 110
_CADS_SIGMOID = 109
_CADS_LIVER = 5
_CADS_AORTA = 7
_CADS_SACRUM = 70
_CADS_HIP_L = 68
_CADS_HIP_R = 69
_CADS_ILIAC_A_L = 51
_CADS_ILIAC_A_R = 52
_CADS_VERTEBRAE_L4 = 19
_CADS_VERTEBRAE_T12 = 23

# Pelvic/lower bone labels that can legitimately sit below the sacrum inferior tip.
# All other bone labels (vertebrae, ribs, sternum, skull, clavicle, humerus, …) are upper-body
# and a centroid below sacrum_z_min means CADS mislabeled the region.
_LOWER_BONE_LABEL_NAMES: frozenset[str] = frozenset({"sacrum", "hip_left", "hip_right", "femur_left", "femur_right"})

# Visceral M1c label IDs in CADS (liver, lungs, adrenals; brain=50 excluded from CTcads)
_CADS_VISCERAL_LABELS: frozenset[int] = frozenset(
    {
        _CADS_LIVER,
        13,
        14,
        15,
        16,
        17,  # lung lobes
        11,
        12,  # adrenal glands
    }
)

# T4-extension CADS label IDs (when also overlapping prostate)
_CADS_T4_LABELS: frozenset[int] = frozenset({_CADS_BLADDER, _CADS_RECTUM, _CADS_SIGMOID})

# Minimum overlap for T4-driving organs (bladder/rectum/sigmoid) after 1-voxel erosion.
_T4_MIN_VOXELS = 3
_T4_MIN_FRACTION = 0.05  # fraction of lesion voxels

# Seminal vesicle (T3b) uses a higher threshold without erosion: the SV is small so
# eroding it removes too much volume and creates false negatives.
_SV_MIN_VOXELS = 10
_SV_MIN_FRACTION = 0.15

# CADS name → ID for T4 organs AND seminal vesicle (built at first instantiation; used by overlap filter)
_CADS_T4_NAME_TO_ID: dict[str, int] = {}

# ─── TotalSegmentator label name sets (fallback when no CTcads) ───────────────

_TS_BONE_LABEL_NAMES: frozenset[str] = frozenset(
    {
        "sacrum",
        "skull",
        "sternum",
        "costal_cartilages",
        "humerus_left",
        "humerus_right",
        "scapula_left",
        "scapula_right",
        "clavicula_left",
        "clavicula_right",
        "femur_left",
        "femur_right",
        "hip_left",
        "hip_right",
        *[
            f"vertebrae_{v}"
            for v in (
                "S1",
                "L5",
                "L4",
                "L3",
                "L2",
                "L1",
                "T12",
                "T11",
                "T10",
                "T9",
                "T8",
                "T7",
                "T6",
                "T5",
                "T4",
                "T3",
                "T2",
                "T1",
                "C7",
                "C6",
                "C5",
                "C4",
                "C3",
                "C2",
                "C1",
            )
        ],
        *[f"rib_{side}_{i}" for side in ("left", "right") for i in range(1, 13)],
    }
)

_TS_VISCERAL_LABEL_NAMES: frozenset[str] = frozenset(
    {
        "liver",
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
        "adrenal_gland_right",
        "adrenal_gland_left",
        "brain",
    }
)

_TS_T4_LABEL_NAMES: frozenset[str] = frozenset({"urinary_bladder", "colon", "small_bowel", "duodenum", "rectum"})

_VERTEBRAL_PREFIXES = ("vertebrae_T", "vertebrae_L", "vertebrae_C", "vertebrae_S")


# ─── Geometry helpers ─────────────────────────────────────────────────────────


def _voxel_to_world(vox_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    hom = np.column_stack([vox_coords, np.ones(len(vox_coords))])
    return (affine @ hom.T).T[:, :3]


def _label_z_range(seg: np.ndarray, label_id: int, affine: np.ndarray) -> tuple[float, float] | None:
    coords = np.argwhere(seg == label_id)
    if not len(coords):
        return None
    zs = _voxel_to_world(coords, affine)[:, 2]
    return float(zs.min()), float(zs.max())


def _label_centroid(seg: np.ndarray, label_id: int, affine: np.ndarray) -> np.ndarray | None:
    coords = np.argwhere(seg == label_id)
    if not len(coords):
        return None
    return _voxel_to_world(coords.mean(axis=0, keepdims=True), affine)[0]


def _build_name_to_id(organ_labels: dict) -> dict[str, int]:
    return {v: int(k) for k, v in organ_labels.items()}


# ─── CADS-based organ overlap ─────────────────────────────────────────────────


def _cads_overlap_names(tumor_mask: np.ndarray, cadsres_crop: np.ndarray) -> set[str]:
    """Return set of CADS label name strings that overlap the lesion mask."""
    present_ids = set(int(v) for v in np.unique(cadsres_crop[tumor_mask.astype(bool)])) - {0}
    return {_CADS_LABEL_NAMES[i] for i in present_ids if i in _CADS_LABEL_NAMES}


def _filter_t4_overlap(
    lesion_mask: np.ndarray,
    seg_crop: np.ndarray,
    organ_overlap: set[str],
    t4_label_ids: dict[str, int],
) -> set[str]:
    """Remove invasion-trigger organ names that don't meet minimum overlap thresholds.

    T4 organs (bladder/rectum/sigmoid): erode by 1 voxel before measuring overlap —
    only voxels past the organ surface count.  Threshold: _T4_MIN_VOXELS / _T4_MIN_FRACTION.

    Seminal vesicle (T3b): no erosion (SV is small; erosion causes false negatives),
    but uses a stricter threshold: _SV_MIN_VOXELS / _SV_MIN_FRACTION.
    """
    _invasion_names = _TS_T4_LABEL_NAMES | frozenset({"seminal_vesicle"})
    t4_candidates = organ_overlap & _invasion_names
    if not t4_candidates:
        return organ_overlap

    total = int(lesion_mask.sum())
    if total == 0:
        return organ_overlap - _invasion_names

    to_remove = set()
    for name in t4_candidates:
        lid = t4_label_ids.get(name)
        if lid is None:
            continue
        organ_mask = seg_crop == lid
        if name == "seminal_vesicle":
            n_overlap = int(np.sum(lesion_mask & organ_mask))
            if n_overlap < _SV_MIN_VOXELS or n_overlap / total < _SV_MIN_FRACTION:
                to_remove.add(name)
        else:
            organ_eroded = ndimage.binary_erosion(organ_mask, iterations=1)
            n_overlap = int(np.sum(lesion_mask & organ_eroded))
            if n_overlap < _T4_MIN_VOXELS or n_overlap / total < _T4_MIN_FRACTION:
                to_remove.add(name)

    return organ_overlap - to_remove


def _cads_landmarks(
    seg: np.ndarray, affine: np.ndarray
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    """Return (pelvic_brim_z, sacrum_z_min, L4_z, T12_z, hip_z_min, midline_x) from CADS seg."""
    pelvic_brim_z = sacrum_z_min = L4_z = T12_z = hip_z_min = midline_x = None

    zr = _label_z_range(seg, _CADS_SACRUM, affine)
    if zr:
        sacrum_z_min, pelvic_brim_z = zr

    c = _label_centroid(seg, _CADS_VERTEBRAE_L4, affine)
    if c is not None:
        L4_z = float(c[2])

    c = _label_centroid(seg, _CADS_VERTEBRAE_T12, affine)
    if c is not None:
        T12_z = float(c[2])

    hip_zs = []
    for hid in (_CADS_HIP_L, _CADS_HIP_R):
        zr = _label_z_range(seg, hid, affine)
        if zr:
            hip_zs.append(zr[0])
    if hip_zs:
        hip_z_min = min(hip_zs)

    # Midline from iliac arteries (more pelvic-specific than aorta)
    xs = []
    for aid in (_CADS_ILIAC_A_L, _CADS_ILIAC_A_R):
        c = _label_centroid(seg, aid, affine)
        if c is not None:
            xs.append(float(c[0]))
    if xs:
        midline_x = sum(xs) / len(xs)
    else:
        c = _label_centroid(seg, _CADS_AORTA, affine)
        if c is not None:
            midline_x = float(c[0])

    return pelvic_brim_z, sacrum_z_min, L4_z, T12_z, hip_z_min, midline_x


def _ts_landmarks(
    seg: np.ndarray, affine: np.ndarray, name_to_id: dict[str, int]
) -> tuple[float | None, float | None, float | None, float | None, float | None, float | None]:
    """Same outputs as _cads_landmarks but from TotalSegmentator seg + its label id map."""
    pelvic_brim_z = sacrum_z_min = L4_z = T12_z = hip_z_min = midline_x = None

    if (sid := name_to_id.get("sacrum")) is not None:
        zr = _label_z_range(seg, sid, affine)
        if zr:
            sacrum_z_min, pelvic_brim_z = zr

    if (l4 := name_to_id.get("vertebrae_L4")) is not None:
        c = _label_centroid(seg, l4, affine)
        if c is not None:
            L4_z = float(c[2])

    if (t12 := name_to_id.get("vertebrae_T12")) is not None:
        c = _label_centroid(seg, t12, affine)
        if c is not None:
            T12_z = float(c[2])

    for hname in ("hip_left", "hip_right"):
        if (hid := name_to_id.get(hname)) is not None:
            zr = _label_z_range(seg, hid, affine)
            if zr:
                hip_z_min = min(hip_z_min, zr[0]) if hip_z_min is not None else zr[0]

    if (aoid := name_to_id.get("aorta")) is not None:
        c = _label_centroid(seg, aoid, affine)
        if c is not None:
            midline_x = float(c[0])

    return pelvic_brim_z, sacrum_z_min, L4_z, T12_z, hip_z_min, midline_x


# ─── PSMA-expression score ────────────────────────────────────────────────────


def _reference_suv(suv: np.ndarray, seg: np.ndarray, label_id: int) -> float | None:
    mask = seg == label_id
    return float(suv[mask].mean()) if mask.any() else None


def _psma_expression_score(suvmax: float | None, bp_suv: float | None, liver_suv: float | None) -> int | None:
    """PROMISE V2 PSMA-expression score 0-3 (score 2 means '>liver'; parotid absent)."""
    if suvmax is None:
        return None
    if bp_suv is not None and suvmax <= bp_suv:
        return 0
    if liver_suv is not None and suvmax <= liver_suv:
        return 1
    if liver_suv is not None:
        return 2  # ≥2; can't distinguish 2 vs 3 without parotid reference
    return None


# ─── miTNM helpers ────────────────────────────────────────────────────────────


def _bone_pattern(n_bone: int, n_vertebral: int) -> str:
    if n_bone == 1:
        return "uni"
    if n_bone <= 3:
        return "oligo"
    if n_vertebral >= 5 and n_vertebral / n_bone >= 0.5:
        return "dmi"
    return "diss"


def _pelvic_subregion(lesion_x: float, lesion_z: float, midline_x: float, pelvic_brim_z: float) -> str:
    """Approximate pelvic LN sub-region from lesion centroid.

    Laterality from x-offset vs. midline (RAS: positive x = patient left).
    Height within pelvis: near brim → EI, mid-pelvis → II, deep → OB.
    Near-midline → PS (presacral).
    """
    offset_x = lesion_x - midline_x
    lat = "L" if offset_x > 0 else "R"

    if abs(offset_x) < 15:
        return "PS"

    pelvic_height = 120.0
    frac = (pelvic_brim_z - lesion_z) / pelvic_height  # 0 = near brim, 1 = deep pelvis

    if frac < 0.4:
        return f"EI_{lat}"
    if frac < 0.7:
        return f"II_{lat}"
    return f"OB_{lat}"


def _promise_stats_key(pet_metric: str, mask_source: str) -> str:
    key = "PromiseStats"
    if mask_source == "revised":
        key += "Revised"
    elif mask_source == "promise":
        key += "Promise"
    if pet_metric == "SUL":
        key += "SUL"
    return key


# ─── Main class ───────────────────────────────────────────────────────────────


class PromiseStaging:
    """PROMISE V2 whole-body miTNM staging from PSMA-PET lesion data.

    Prefers CTcadsres.nii.gz (CADS) for organ overlap — gives seminal vesicle
    as a distinct structure enabling miT3b detection.  Falls back to CTsegres.nii.gz
    (TotalSegmentator) when CTcads hasn't been run.

    Depends on: the 'tumor' task (TumorStats in patient_info.json).
    Reads: PETseg.nii.gz, CTcadsres.nii.gz or CTsegres.nii.gz, SUV.nii.gz.
    Writes: PromiseStats (or PromiseStatsRevised/SUL) to patient_info.json.
    """

    def __init__(
        self,
        input_dirpath_processed: str | os.PathLike,
        pet_metric: str = "SUV",
        mask_source: str = "auto",
        exclude_patients: list[str] | None = None,
    ) -> None:
        if pet_metric not in ("SUV", "SUL"):
            raise ValueError(f"pet_metric must be 'SUV' or 'SUL', got '{pet_metric}'")
        if mask_source not in ("auto", "revised", "promise"):
            raise ValueError(f"mask_source must be 'auto', 'revised', or 'promise', got '{mask_source}'")
        self.input_dirpath = input_dirpath_processed
        self.pet_metric = pet_metric
        self.mask_source = mask_source
        self.exclude_patients: frozenset[str] = frozenset(exclude_patients or [])

        # Populate the CADS T4 name→ID map on first instantiation (needs _CADS_LABEL_NAMES loaded)
        if not _CADS_T4_NAME_TO_ID:
            _CADS_T4_NAME_TO_ID.update(
                {_CADS_LABEL_NAMES[lid]: lid for lid in _CADS_T4_LABELS if lid in _CADS_LABEL_NAMES}
            )
            # Also include seminal vesicle so boundary-bleed T3b is suppressed by the same erosion filter
            _CADS_T4_NAME_TO_ID[_CADS_LABEL_NAMES[_CADS_SEMINAL_VESICLE]] = _CADS_SEMINAL_VESICLE

    def run(self) -> None:
        if not os.path.isdir(self.input_dirpath):
            logger.error("Not a directory: %s", self.input_dirpath)
            return
        for top_dir in utils.list_patient_dirs(self.input_dirpath):
            if top_dir in self.exclude_patients:
                logger.info("Excluding patient %s.", top_dir)
                continue
            top_path = os.path.join(self.input_dirpath, top_dir)
            for dirpath, dirnames, _ in os.walk(top_path):
                rel = plb.Path(os.path.relpath(dirpath, self.input_dirpath)).parts
                if len(rel) != 2:
                    continue
                dirnames.clear()
                self._process_study(dirpath)

    def _process_study(self, study_dirpath: str) -> None:
        patient_dirpath = os.path.dirname(study_dirpath)
        study_date = os.path.basename(study_dirpath)
        pinfo_path = os.path.join(patient_dirpath, "patient_info.json")

        if not os.path.isfile(pinfo_path):
            logger.warning("Missing patient_info.json in %s; skipping.", patient_dirpath)
            return

        with open(pinfo_path) as f:
            patient_info = json.load(f)

        study = patient_info.get("Studies", {}).get(study_date)
        if study is None:
            return

        if self.mask_source == "promise":
            # Prefer revised lesion stats (source of the promise mask for most patients)
            ts_key = "TumorStatsRevised" if study.get("TumorStatsRevised") else "TumorStats"
        elif self.mask_source == "revised":
            ts_key = "TumorStatsRevised"
        elif self.pet_metric == "SUL":
            ts_key = "TumorStatsSUL"
        else:
            ts_key = "TumorStats"
        tumor_stats = study.get(ts_key)
        if not tumor_stats or not tumor_stats.get("Tumors"):
            logger.debug("No %s in %s/%s; skipping.", ts_key, patient_dirpath, study_date)
            return

        promise_key = _promise_stats_key(self.pet_metric, self.mask_source)
        if promise_key in study:
            logger.info("%s already present for %s; skipping.", promise_key, study_dirpath)
            return

        # Resolve PETseg mask
        if self.mask_source == "revised":
            from .radiomics_extraction import DEFAULT_LABEL_GLOB, resolve_mask

            petseg_path, _ = resolve_mask(study_dirpath, self.pet_metric, "revised", None, DEFAULT_LABEL_GLOB)
            if petseg_path is None:
                logger.warning("No revised mask in %s; skipping.", study_dirpath)
                return
        elif self.mask_source == "promise":
            petseg_path = os.path.join(study_dirpath, "PETseg_revised_promise.nii.gz")
        else:
            fname = "PETseg.nii.gz" if self.pet_metric == "SUV" else "PETsegSUL.nii.gz"
            petseg_path = os.path.join(study_dirpath, fname)

        suv_path = os.path.join(study_dirpath, f"{self.pet_metric}.nii.gz")
        for p in (petseg_path, suv_path):
            if not os.path.isfile(p):
                logger.warning("Missing %s for %s; skipping.", p, study_dirpath)
                return

        logger.info("Computing PROMISE staging for %s", study_dirpath)

        petseg_img = nib.load(petseg_path)
        petseg_arr = np.asanyarray(petseg_img.dataobj).astype(np.uint8)
        affine = petseg_img.affine
        suv_arr = metrics.get_3darray_from_niftipath(suv_path)

        # Revised masks saved by external viewers sometimes have a flipped or shifted affine
        # relative to the PET/CTcadsres grid. Resample to PET space so voxel indices align
        # with CTcadsres (same grid) — mirrors the check in tumor_info_extraction.
        if self.mask_source == "revised":
            suv_img = nib.load(suv_path)
            if not np.allclose(affine, suv_img.affine, atol=1e-3):
                from .radiomics_extraction import resample_label_to_image_grid

                logger.info("Revised PETseg affine mismatch in %s; resampling to PET grid.", study_dirpath)
                resampled = resample_label_to_image_grid(petseg_path, suv_path, study_dirpath)
                if resampled is not None and resampled.any():
                    petseg_arr = resampled
                    affine = suv_img.affine
                else:
                    logger.warning("Resample failed for %s; skipping.", study_dirpath)
                    return

        # ── Choose segmentation source: CADS > TotalSegmentator ─────────────
        ctcadsres_path = os.path.join(study_dirpath, "CTcadsres.nii.gz")
        ctsegres_path = os.path.join(study_dirpath, "CTsegres.nii.gz")

        use_cads = os.path.isfile(ctcadsres_path)
        seg_arr = None
        name_to_id: dict[str, int] = {}

        if use_cads:
            seg_arr = metrics.get_3darray_from_niftipath(ctcadsres_path)
            logger.debug("Using CTcadsres for PROMISE organ overlap in %s", study_dirpath)
        else:
            # Create CTsegres on demand if absent
            if not os.path.isfile(ctsegres_path):
                ctseg_path = os.path.join(study_dirpath, "CTseg.nii.gz")
                pet_path = os.path.join(study_dirpath, "PET.nii.gz")
                if os.path.isfile(ctseg_path) and os.path.isfile(pet_path):
                    utils.resample_image(
                        source_img=ctseg_path,
                        target_img=pet_path,
                        nii_output_dirpath=study_dirpath,
                        interpolation="nearest",
                        fill_value=0,
                        output_fname="CTsegres.nii.gz",
                    )
            if os.path.isfile(ctsegres_path):
                seg_arr = metrics.get_3darray_from_niftipath(ctsegres_path)
                try:
                    ct0 = study["Modalities"]["CT"][0]
                    sname = next(iter(ct0))
                    organ_labels = ct0[sname].get("CTseg_metadata", {}).get("labels", {})
                    name_to_id = _build_name_to_id(organ_labels)
                except (KeyError, IndexError, StopIteration):
                    pass
            logger.debug("Using CTsegres (TS fallback) for PROMISE organ overlap in %s", study_dirpath)

        # ── PSMA reference SUVs ──────────────────────────────────────────────
        blood_pool_suv = liver_suv = None
        if seg_arr is not None:
            aorta_id = _CADS_AORTA if use_cads else name_to_id.get("aorta")
            liver_id = _CADS_LIVER if use_cads else name_to_id.get("liver")
            if aorta_id is not None:
                blood_pool_suv = _reference_suv(suv_arr, seg_arr, aorta_id)
            if liver_id is not None:
                liver_suv = _reference_suv(suv_arr, seg_arr, liver_id)

        # ── Anatomical landmarks for spatial LN classification ───────────────
        if seg_arr is not None:
            if use_cads:
                pelvic_brim_z, sacrum_z_min, L4_z, T12_z, hip_z_min, midline_x = _cads_landmarks(seg_arr, affine)
            else:
                pelvic_brim_z, sacrum_z_min, L4_z, T12_z, hip_z_min, midline_x = _ts_landmarks(
                    seg_arr, affine, name_to_id
                )
        else:
            pelvic_brim_z = sacrum_z_min = L4_z = T12_z = hip_z_min = midline_x = None

        # ── Is prostate segmented? (absent after prostatectomy) ─────────────
        prostate_present = False
        if seg_arr is not None:
            pid = _CADS_PROSTATE if use_cads else name_to_id.get("prostate")
            if pid is not None:
                prostate_present = bool((seg_arr == pid).any())

        # ── Per-lesion classification ────────────────────────────────────────
        labeled, n_lesions = cc3d.connected_components(petseg_arr, connectivity=18, return_N=True)
        bboxes = ndimage.find_objects(labeled)
        stored_by_id = {t["TumorID"]: t for t in tumor_stats["Tumors"]}
        mx = midline_x if midline_x is not None else 0.0
        # TS fallback: map integer label IDs → name strings (computed once)
        ts_id_to_name = {v: k for k, v in name_to_id.items()} if (not use_cads and name_to_id) else {}
        # T4 threshold: label name → integer ID, for the chosen seg source
        if use_cads:
            t4_label_ids = _CADS_T4_NAME_TO_ID
        else:
            t4_label_ids = {name: name_to_id[name] for name in _TS_T4_LABEL_NAMES if name in name_to_id}

        lesion_entries = []
        for i in range(1, n_lesions + 1):
            bbox = bboxes[i - 1]
            if bbox is None:
                continue
            crop = tuple(slice(s.start, s.stop) for s in bbox)
            local_coords = np.argwhere(labeled[crop] == i)
            centroid_vox = local_coords.mean(axis=0) + np.array([s.start for s in bbox])
            centroid = _voxel_to_world(centroid_vox[np.newaxis], affine)[0]

            stored = stored_by_id.get(i, {})
            suvmax = stored.get("PETMetrics", {}).get("SUVmax")
            volume_cm3 = stored.get("Volume_cm3", 0.0)
            psma_score = _psma_expression_score(suvmax, blood_pool_suv, liver_suv)

            # Compute organ overlap from the chosen seg source
            lesion_mask = labeled[crop] == i
            seg_crop = seg_arr[crop] if seg_arr is not None else None
            if seg_arr is not None and use_cads:
                # CADS: _CADS_LABEL_NAMES maps int IDs → name strings
                organ_overlap = _cads_overlap_names(lesion_mask, seg_crop)
            elif seg_arr is not None and ts_id_to_name:
                # TotalSegmentator: remap via the label dict stored in patient_info
                present_ids = set(int(v) for v in np.unique(seg_crop[lesion_mask])) - {0}
                organ_overlap = {ts_id_to_name[lid] for lid in present_ids if lid in ts_id_to_name}
            else:
                # No seg array: fall back to stored OrganOverlap (TS labels from tumor stage)
                organ_overlap = {k for k, v in stored.get("OrganOverlap", {}).items() if v}

            # Apply T4 minimum overlap threshold: remove T4-driving organs with only
            # boundary-bleed voxels (prostate is directly adjacent to rectum/bladder in CADS)
            if seg_crop is not None:
                organ_overlap = _filter_t4_overlap(lesion_mask, seg_crop, organ_overlap, t4_label_ids)

            cat, detail, bone_sites, organ_sites = self._classify(
                organ_overlap,
                centroid,
                pelvic_brim_z,
                sacrum_z_min,
                L4_z,
                T12_z,
                hip_z_min,
                mx,
            )

            lesion_entries.append(
                {
                    "TumorID": i,
                    "Volume_cm3": volume_cm3,
                    "SUVmax": suvmax,
                    "PSMA_expression_score": psma_score,
                    "category": cat,
                    "detail": detail,
                    "bone_sites": sorted(bone_sites),
                    "organ_sites": sorted(organ_sites),
                    "centroid_world_mm": centroid.tolist(),
                    "centroid_voxel_ijk": [round(float(v)) for v in centroid_vox],
                }
            )

        seg_source = "CTcads" if use_cads else "CTseg"
        promise_stats = self._assemble(lesion_entries, prostate_present, blood_pool_suv, liver_suv)
        promise_stats["seg_source"] = seg_source
        promise_stats["mask_source"] = self.mask_source
        promise_stats["pet_metric"] = self.pet_metric

        patient_info["Studies"][study_date][promise_key] = promise_stats
        with open(pinfo_path, "w") as f:
            json.dump(patient_info, f)
        logger.info("Wrote %s [%s] for %s", promise_key, seg_source, study_dirpath)

    def _classify(
        self,
        organ_overlap: set[str],
        centroid: np.ndarray,
        pelvic_brim_z: float | None,
        sacrum_z_min: float | None,
        L4_z: float | None,
        T12_z: float | None,
        hip_z_min: float | None,
        midline_x: float,
    ) -> tuple[str, str, list[str], list[str]]:
        """Return (category, detail, bone_sites, organ_sites).  Priority: T > M1c > M1b > N/M1a."""
        if "prostate" in organ_overlap:
            sv = "seminal_vesicle" in organ_overlap
            ext = organ_overlap & _TS_T4_LABEL_NAMES
            if ext:
                detail = "prostate+" + "|".join(sorted(ext))
                return "T", detail, [], []
            if sv:
                return "T", "prostate+seminal_vesicle", [], []
            return "T", "prostate", [], []

        visceral = sorted(organ_overlap & _TS_VISCERAL_LABEL_NAMES)
        if visceral:
            return "M1c", visceral[0], [], visceral

        bones = sorted(organ_overlap & _TS_BONE_LABEL_NAMES)
        if bones:
            lesion_z = float(centroid[2])
            # Guard: upper-body bone labels (vertebrae, ribs, …) cannot have a centroid
            # below the sacrum inferior tip — that indicates a CADS mislabelling.
            # Pelvic/lower bones (sacrum, hip, femur) may legitimately sit lower.
            only_upper = not (set(bones) & _LOWER_BONE_LABEL_NAMES)
            if only_upper and sacrum_z_min is not None and lesion_z < sacrum_z_min:
                logger.debug(
                    "Ignoring bone overlap %s at z=%.1f (below sacrum_z_min=%.1f): CADS mislabel",
                    bones,
                    lesion_z,
                    sacrum_z_min,
                )
                bones = []
            else:
                primary = next((b for b in bones if b.startswith("vertebrae_")), bones[0])
                return "M1b", primary, bones, []

        # Soft tissue / lymph node — classify spatially
        lesion_z = float(centroid[2])
        lesion_x = float(centroid[0])

        if pelvic_brim_z is not None:
            if lesion_z > pelvic_brim_z:
                if T12_z is not None and lesion_z > T12_z:
                    return "M1a", "SD", [], []
                if L4_z is not None and lesion_z > L4_z:
                    return "M1a", "RP", [], []
                return "M1a", "CI", [], []
            if hip_z_min is not None and lesion_z < hip_z_min:
                return "M1a", "OE", [], []
            subregion = _pelvic_subregion(lesion_x, lesion_z, midline_x, pelvic_brim_z)
            return "N", subregion, [], []

        return "N", "OP", [], []

    def _assemble(
        self,
        lesions: list[dict],
        prostate_present: bool,
        blood_pool_suv: float | None,
        liver_suv: float | None,
    ) -> dict:
        t = [les for les in lesions if les["category"] == "T"]
        n = [les for les in lesions if les["category"] == "N"]
        m1a = [les for les in lesions if les["category"] == "M1a"]
        m1b = [les for les in lesions if les["category"] == "M1b"]
        m1c = [les for les in lesions if les["category"] == "M1c"]

        # T stage
        if t:
            has_t4 = any("+" in les["detail"] and "seminal_vesicle" not in les["detail"] for les in t)
            has_t3b = any("seminal_vesicle" in les["detail"] for les in t)
            if has_t4:
                miT = "miT4"
            elif has_t3b:
                miT = "miT3b"
            else:
                miT = "miT2u" if len(t) == 1 else "miT2m"
        else:
            miT = "miT0"

        # N stage
        n_regions = sorted({les["detail"] for les in n})
        miN = "miN0" if not n else ("miN1" if len(n_regions) == 1 else "miN2")

        # M stage
        m1b_all_bones = [s for les in m1b for s in les["bone_sites"]]
        n_vertebral = sum(1 for s in m1b_all_bones if any(s.startswith(p) for p in _VERTEBRAL_PREFIXES))
        pattern = _bone_pattern(len(m1b), n_vertebral) if m1b else None

        miM = "miM1c" if m1c else ("miM1b" if m1b else ("miM1a" if m1a else "miM0"))

        # Code string — report all present M categories
        n_code = miN + (f"({'|'.join(n_regions)})" if n_regions else "")
        m_parts = []
        if m1a:
            m_parts.append(f"M1a({'|'.join(sorted({les['detail'] for les in m1a}))})")
        if m1b:
            m_parts.append(f"M1b({pattern})")
        if m1c:
            m_parts.append(f"M1c({'|'.join(sorted({s for les in m1c for s in les['organ_sites']}))})")
        m_code = " ".join(m_parts) if m_parts else "M0"
        code = f"{miT} {n_code} {m_code}"

        return {
            "miTNM": {
                "T": miT,
                "N": miN,
                "M": miM,
                "M_pattern": pattern,
                "code": code,
            },
            "PSMA_reference": {
                "blood_pool_SUV": blood_pool_suv,
                "liver_SUV": liver_suv,
                "parotid_SUV": None,
                "note": "Parotid unavailable; score 2 means >liver (may be 2 or 3 per PROMISE V2).",
            },
            "prostate_present_in_seg": prostate_present,
            "summary": {
                "T_count": len(t),
                "N_count": len(n),
                "M1a_count": len(m1a),
                "M1b_count": len(m1b),
                "M1c_count": len(m1c),
                "total_PSMA_VOL_ml": sum(les["Volume_cm3"] for les in lesions),
                "T_PSMA_VOL_ml": sum(les["Volume_cm3"] for les in t),
                "N_PSMA_VOL_ml": sum(les["Volume_cm3"] for les in n),
                "M1a_PSMA_VOL_ml": sum(les["Volume_cm3"] for les in m1a),
                "M1b_PSMA_VOL_ml": sum(les["Volume_cm3"] for les in m1b),
                "M1c_PSMA_VOL_ml": sum(les["Volume_cm3"] for les in m1c),
                "M1b_pattern": pattern,
                "M1b_bone_sites": sorted({s for les in m1b for s in les["bone_sites"]}),
                "M1a_regions": sorted({les["detail"] for les in m1a}),
                "M1c_organ_sites": sorted({s for les in m1c for s in les["organ_sites"]}),
                "N_regions": n_regions,
            },
            "Lesions": lesions,
        }


def _strip_promise_stats(input_dirpath_processed: str, pet_metric: str, mask_source: str) -> int:
    """Remove existing PromiseStats keys from all patient_info.json files. Returns count removed."""
    promise_key = _promise_stats_key(pet_metric, mask_source)
    n = 0
    for dirpath, _, filenames in os.walk(input_dirpath_processed):
        if "patient_info.json" not in filenames:
            continue
        pinfo_path = os.path.join(dirpath, "patient_info.json")
        with open(pinfo_path) as f:
            patient_info = json.load(f)
        changed = False
        for study in patient_info.get("Studies", {}).values():
            if promise_key in study:
                del study[promise_key]
                changed = True
                n += 1
        if changed:
            with open(pinfo_path, "w") as f:
                json.dump(patient_info, f)
    return n


def promise_staging_entrypoint() -> None:
    from .utils import create_logger

    global logger
    logger = create_logger("musiq.promise_staging")

    import argparse

    parser = argparse.ArgumentParser(
        description="Compute PROMISE V2 miTNM staging from TumorStats in the processed tree. "
        "Uses CTcadsres.nii.gz when available (seminal vesicle → miT3b); falls back to CTsegres."
    )
    parser.add_argument("--input-dirpath-processed", required=True, help="Processed output tree root.")
    parser.add_argument("--pet-metric", choices=["SUV", "SUL"], default="SUV")
    parser.add_argument(
        "--mask-source",
        choices=["auto", "revised", "promise"],
        default="auto",
        help="'auto' → PromiseStats, 'revised' → PromiseStatsRevised, 'promise' → PromiseStatsPromise. Default: auto.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Strip existing PromiseStats entries before running (forces full recompute).",
    )
    parser.add_argument(
        "--exclude-patients",
        nargs="+",
        default=[],
        metavar="PATIENT_ID",
        help="Patient directory names to skip entirely (e.g. mp_0030 mp_0126).",
    )
    args = parser.parse_args()

    if args.rerun:
        n = _strip_promise_stats(args.input_dirpath_processed, args.pet_metric, args.mask_source)
        logger.info("Stripped %d existing %s entries.", n, _promise_stats_key(args.pet_metric, args.mask_source))

    PromiseStaging(
        input_dirpath_processed=args.input_dirpath_processed,
        pet_metric=args.pet_metric,
        mask_source=args.mask_source,
        exclude_patients=args.exclude_patients,
    ).run()


if __name__ == "__main__":
    promise_staging_entrypoint()
