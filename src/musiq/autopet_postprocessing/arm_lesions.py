"""
Flag/remove above-shoulder ARM lesions, using CADS labels.

Pipeline order:  lacrimalRemoval.py  ->  arm_lesions.py  ->  smallVoxRemoval.py
This script reads <mask>_postPro.nii.gz (the lacrimal-cleaned mask) and updates it IN PLACE
(pass --no-save to only log). It never touches the original mask.

Choose which mask(s) to process by passing --masks:
  --masks PETseg                            # SUV only
  --masks PETseg_revised                    # Helen's only
  --masks PETseg PETsegSUL                  # both model masks
  --masks PETseg PETsegSUL PETseg_revised   # all three
If --masks is not given, it defaults to PETseg PETsegSUL (SUV + SUL).

Intended for arms-up acquisitions where the forearm/hand fold over the head, so
physiologic arm/skin uptake lands in the same axial slices as the head.

Decision per lesion:
  flag if the lesion sits ABOVE the humerus (on the head side) AND is dominantly
  subcutaneous tissue / background.

Orientation is NOT assumed from the axis convention. It is derived from the data:
  - the head anchor (cerebrospinal fluid; falls back to gray matter) is known head,
  - the humerus is known arm,
  - the axis along which those two are most separated is the superior-inferior
    axis, and the sign of (head_centroid - humerus_centroid) along that axis
    tells us which direction is "toward the head".
This makes the rule robust to RAS/LPS/flipped volumes.

CAUTION: the axilla (armpit) sits next to the proximal humerus and can contain
real nodal disease. This rule targets tissue ABOVE the humerus (folded forearm),
not lateral to it. It will also catch head subcutaneous tissue (e.g. scalp) in those
slices, so check the flagged lesions in the log (use --no-save for a log-only look first).

Per session and mask (skipped with --no-save):
  <mask>_postPro.nii.gz          updated IN PLACE: the flagged arm lesions are removed from it
  <mask>_postProRemoved.nii.gz  the arm lesions are ADDED to it, so it holds every lesion removed so far
                                 (lacrimal + arm), labels kept, 0 elsewhere. No separate arm file is made.
Rerunning on the same _postPro is safe: nothing more is removed and the removed-lesions file is unchanged.
If lacrimalRemoval.py is rerun (it rebuilds _postPro and _postProRemoved from the original mask), run
this script and smallVoxRemoval.py again afterwards.

Remove + save:  python3 -u arm_lesions.py --input-dirpath /path/to/processed --masks PETseg_revised
Log only:       python3 -u arm_lesions.py --input-dirpath /path/to/processed --masks PETseg_revised --no-save
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
import cc3d  # pip install connected-components-3d
from scipy.ndimage import distance_transform_edt

# data discovery / config / helpers shared with the lacrimal step
from lacrimalRemoval import (
    DATA_DIR, MASK_CHOICES, DEFAULT_MASKS, OUT_SUFFIX, REMOVED_SUFFIX,
    NAMES, resample_cads_to_grid,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
HUMERUS_LABELS      = [60, 61]        # CADS: Humerus L / R  (arm reference)
CSF_LABEL           = [122]           # CADS: Cerebrospinal fluid -- head-position anchor AND skull-ignore
                                       # reference. Large structure (unlike the lacrimal gland), so it
                                       # survives resampling reliably -- no native-resolution work needed.
GRAY_MATTER_FALLBACK = [121]          # CADS: Gray matter (used if CSF absent)
SUBCUTANEOUS_LABEL  = 158             # CADS: Subcutaneous tissue
ALLOWED_DOMINANT    = {0, SUBCUTANEOUS_LABEL}   # background(0) / subcutaneous -> flaggable when above humerus
CONNECTIVITY        = 18              # keep consistent with your metrics/compare stage

SKULL_IGNORE_NEAR_MM = 24.0   # ignore a lesion inside OR within this many mm of CSF_LABEL
                               # (distance-based, not voxel dilation -- real lesions can sit just outside
                               # the skull in the outer subcutaneous/scalp layer and must not be suppressed)

# Patients to SKIP for arm processing (folder names). Add up to 8 here.
ARM_EXCLUDED_PATIENTS = {
    "mp_0009",
    "mp_0030",
    "mp_0035",
    "mp_0037",
    "mp_0079",
    "mp_0086",
    "mp_0088",
    "mp_0093",
    "mp_0115"
}

CADS_FILENAME   = "CTcadsres.nii.gz"   # already resampled to match the PET grid -- no native-res work needed


def head_direction(cads_arr):
    """Return (si_axis, head_dir, hum_edge, ok, anchor_name).

    si_axis  : the array axis separating head from humerus (superior-inferior).
    head_dir : +1 if the head is at a larger index along si_axis, else -1.
    hum_edge : the humerus coordinate (along si_axis) on the head-facing side.
    ok       : False if humerus or head anchor is missing -> can't orient.
    """
    hum = np.isin(cads_arr, HUMERUS_LABELS)
    head = np.isin(cads_arr, CSF_LABEL)
    anchor = "CSF"
    if not head.any():
        head = np.isin(cads_arr, GRAY_MATTER_FALLBACK)
        anchor = "gray matter"
    if not (hum.any() and head.any()):
        return None, None, None, False, "none"

    hum_idx  = np.argwhere(hum)
    head_idx = np.argwhere(head)
    diff = head_idx.mean(axis=0) - hum_idx.mean(axis=0)   # head_centroid - hum_centroid
    si_axis = int(np.argmax(np.abs(diff)))                # most-separated axis = SI
    head_dir = 1 if diff[si_axis] > 0 else -1
    hum_coords = hum_idx[:, si_axis]
    hum_edge = int(hum_coords.max()) if head_dir > 0 else int(hum_coords.min())
    return si_axis, head_dir, hum_edge, True, anchor


def save_atomic(arr, ref_img, out_file):
    """Write to a temp name then swap in, so a rerun fully replaces the old file."""
    tmp_file = out_file.with_name(out_file.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(nib.Nifti1Image(arr, ref_img.affine, ref_img.header), str(tmp_file))
    tmp_file.replace(out_file)


class ArmLesionRemover:
    def __init__(self, input_dirpath, multiprocessing=False, max_workers=30,
                 limit=None, patient_id=None, save_mask=True, masks=None):
        self.input_dirpath = input_dirpath
        self.multiprocessing = multiprocessing
        self.max_workers = max_workers
        self.limit = limit
        self.patient_id = patient_id
        self.save_mask = save_mask
        self.masks = list(masks) if masks else list(DEFAULT_MASKS)

    # ---- discovery + dispatch ------------------------------------------- #
    def run(self):
        sub_dirs = sorted(
            dirpath
            for dirpath, _, filenames in os.walk(self.input_dirpath)
            if CADS_FILENAME in filenames
            and any(f"{m}{OUT_SUFFIX}.nii.gz" in filenames for m in self.masks)
        )
        print(f"Found {len(sub_dirs)} session(s) with {CADS_FILENAME} + <mask>{OUT_SUFFIX} for {self.masks}",
              flush=True)

        if ARM_EXCLUDED_PATIENTS:
            before = len(sub_dirs)
            sub_dirs = [d for d in sub_dirs if Path(d).parent.name not in ARM_EXCLUDED_PATIENTS]
            print(f"Arm processing: excluded {before - len(sub_dirs)} session(s) "
                  f"for {len(ARM_EXCLUDED_PATIENTS)} listed patient(s)", flush=True)

        if self.patient_id is not None:
            sub_dirs = [d for d in sub_dirs if Path(d).parent.name == self.patient_id]
            print(f"Filtering to patient_id={self.patient_id}: {len(sub_dirs)} session(s)", flush=True)
        if self.limit is not None:
            sub_dirs = sub_dirs[:self.limit]
            print(f"Limiting to first {len(sub_dirs)} session(s)", flush=True)
        if not sub_dirs:
            return

        mode = "updating _postPro in place" if self.save_mask else "--no-save (no mask writing)"
        print(f"Mode: {mode}", flush=True)

        if self.multiprocessing:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self.process_wrapper, sub_dirs))
        else:
            results = [self.process_wrapper(d) for d in sub_dirs]

        rows = [r for res in results if res for r in res]
        verb = "Removed" if self.save_mask else "Flagged (--no-save)"
        print(f"\n{verb} {len(rows)} arm lesion(s) across {len(sub_dirs)} session(s):", flush=True)
        for r in rows:
            print(f"  {r['session']} [{r['mask_name']}] lesion {r['component_id']}: {r['voxels']} vox, "
                  f"SI {r['centroid_si']} (humerus edge {r['humerus_edge_si']}), "
                  f"bg {r['background_fraction']:.0%}, organs {r['organs']}", flush=True)
        print("\nArm lesion processing done.", flush=True)

    def process_wrapper(self, dirpath):
        try:
            return self.process_patient(dirpath)
        except Exception as e:
            print(f"  !! ERROR processing {dirpath}: {e}", flush=True)
            return []

    # ---- per-session work ----------------------------------------------- #
    def process_patient(self, dirpath):
        session = Path(dirpath)
        print(f"\n=== {session.parent.name}/{session.name} ===", flush=True)

        # CTcadsres.nii.gz is already resampled to match the PET grid, and
        # CSF/gray-matter are large enough to survive that resampling, so we
        # can load it once and reuse it directly -- no native-resolution work
        cads_img = nib.squeeze_image(nib.load(str(session / CADS_FILENAME)))
        cads_arr = np.rint(cads_img.get_fdata()).astype(np.int32)

        rows = []
        for mask_name in self.masks:
            rows.extend(self.process_mask(session, mask_name, cads_img, cads_arr))
        return rows

    def process_mask(self, session, mask_name, cads_img, cads_arr_native):
        tag = f"{session.parent.name}/{session.name}"
        in_path = session / f"{mask_name}{OUT_SUFFIX}.nii.gz"
        if not in_path.exists():
            print(f"  [{mask_name}] no {in_path.name} (run lacrimalRemoval.py first) - skipped", flush=True)
            return []

        model_img = nib.squeeze_image(nib.load(str(in_path)))
        data = model_img.get_fdata()
        cads_arr = resample_cads_to_grid(cads_img, cads_arr_native, model_img)

        csf = np.isin(cads_arr, CSF_LABEL)
        if csf.any():
            spacing = np.sqrt((model_img.affine[:3, :3] ** 2).sum(axis=0))
            skull_ignore_dist = distance_transform_edt(~csf, sampling=spacing)
        else:
            skull_ignore_dist = np.full(cads_arr.shape, np.inf, dtype=np.float32)

        # establish head/arm orientation from the anchors (label-guided)
        si_axis, head_dir, hum_edge, ok, anchor = head_direction(cads_arr)
        if not ok:
            print(f"  [{mask_name}] humerus and/or head anchor missing - skipping arm rule", flush=True)
            return []
        print(f"  [{mask_name}] input {in_path.name} | anchor={anchor} | SI axis={si_axis} | "
              f"head_dir={'+' if head_dir > 0 else '-'} | humerus edge={hum_edge}", flush=True)

        cc, n = cc3d.connected_components((data > 0).astype(np.int32),
                                          connectivity=CONNECTIVITY, return_N=True)
        stats = cc3d.statistics(cc)
        voxel_counts = stats["voxel_counts"]
        bboxes = stats["bounding_boxes"]
        centroids = stats["centroids"]
        print(f"  [{mask_name}] lesions: {n}", flush=True)

        rows, drop_ids = [], []
        for lid in range(1, n + 1):
            vox = int(voxel_counts[lid])
            centroid_si = float(centroids[lid][si_axis])

            # "above the humerus, toward the head" along the SI axis
            above = (centroid_si > hum_edge) if head_dir > 0 else (centroid_si < hum_edge)

            bbox = bboxes[lid]
            local = cc[bbox] == lid
            vals, counts = np.unique(cads_arr[bbox][local], return_counts=True)
            frac_by_label = {int(v): int(c) / vox for v, c in zip(vals, counts)}
            background_fraction = frac_by_label.pop(0, 0.0)
            dom = max(frac_by_label, key=frac_by_label.get) if frac_by_label else 0

            skull_ignore_min_dist = float(skull_ignore_dist[bbox][local].min())
            near_skull_ignore = skull_ignore_min_dist <= SKULL_IGNORE_NEAR_MM

            is_subcut = dom in ALLOWED_DOMINANT or background_fraction >= 0.5
            would_flag = above and is_subcut
            flag = would_flag and not near_skull_ignore

            touches = {NAMES.get(int(v), int(v)): int(c)
                       for v, c in zip(vals, counts) if v != 0}
            if flag:
                mark = "  <-- FLAG (above-humerus subcutaneous)"
            elif would_flag and near_skull_ignore:
                mark = f"  <-- SUPPRESSED (CSF dist {skull_ignore_min_dist:.1f}mm - skull region)"
            else:
                mark = ""
            print(f"    [{tag}] lesion {lid:>2}: {vox:>6} vox | SI {centroid_si:6.1f} "
                  f"(edge {hum_edge}) | above={str(above):5} | bg {background_fraction:4.0%} "
                  f"| CSF dist {skull_ignore_min_dist:5.1f}mm | touches {touches}{mark}", flush=True)

            if not flag:
                continue

            drop_ids.append(lid)
            ranked = sorted(frac_by_label.items(), key=lambda kv: -kv[1])[:3]
            rows.append({
                "session": tag,
                "mask_name": mask_name,
                "component_id": lid,
                "voxels": vox,
                "centroid_si": round(centroid_si, 1),
                "humerus_edge_si": hum_edge,
                "background_fraction": round(background_fraction, 2),
                "organs": [{"name": NAMES.get(label, str(label)), "fraction": round(frac, 2)}
                           for label, frac in ranked],
            })

        # update _postPro in place + save the removed lesions (skipped with --no-save)
        if not self.save_mask:
            print(f"  [{mask_name}] flagged {len(drop_ids)} (--no-save; no files written)", flush=True)
            return rows

        if not drop_ids:
            print(f"  [{mask_name}] nothing to remove; {in_path.name} kept", flush=True)
            return rows

        # the arm lesions are ADDED to the shared <mask>_postProRemoved file (one file holds every
        # lesion removed so far). This is a union, so rerunning is safe: lesions already removed from
        # _postPro are not found again, and after a lacrimal rerun (fresh _postPro and a fresh removed
        # file) the arm lesions are found again and merged in again.
        removed_file = session / f"{mask_name}{REMOVED_SUFFIX}.nii.gz"
        dtype = model_img.get_data_dtype()
        removed = np.isin(cc, drop_ids)
        removed_arr = np.where(removed, data, 0)
        if removed_file.exists():
            old = np.asanyarray(nib.load(str(removed_file)).dataobj).reshape(data.shape)
            removed_arr = np.where(old > 0, old, removed_arr)
        # removed file first, _postPro second: a crash in between leaves the lesions in _postPro, so a
        # rerun finds them again and merges them again (no lesion can end up in neither file)
        save_atomic(removed_arr.astype(dtype), model_img, removed_file)
        save_atomic(np.where(removed, 0, data).astype(dtype), model_img, in_path)
        print(f"  [{mask_name}] removed {len(drop_ids)} -> updated {in_path.name} (in place), "
              f"added to {removed_file.name}", flush=True)
        return rows


def arm_lesion_removal_entrypoint():
    parser = argparse.ArgumentParser(
        description="Flag/remove above-shoulder arm lesions from the lacrimal-cleaned masks (<mask>_postPro) using CADS."
    )
    parser.add_argument("--input-dirpath", type=str, default=str(DATA_DIR),
                        help="Processed data root; searched recursively for sessions with CTcadsres + <mask>_postPro.")
    parser.add_argument("--masks", nargs="+", choices=MASK_CHOICES, default=DEFAULT_MASKS,
                        help="Which mask(s): PETseg (SUV), PETsegSUL (SUL), PETseg_revised (doctors' annotation).")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write files: by default the flagged lesions are removed from <mask>_postPro "
                             "in place and added to <mask>_postProRemoved.nii.gz (calibration/dry run).")
    parser.add_argument("--multiprocessing", action="store_true",
                        help="Process sessions in parallel with a process pool.")
    parser.add_argument("--max-workers", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N sessions found (for test runs).")
    parser.add_argument("--patient-id", type=str, default=None,
                        help="Only process sessions for this patient_id (folder name), for debugging.")
    args = parser.parse_args()

    ArmLesionRemover(
        input_dirpath=args.input_dirpath,
        multiprocessing=args.multiprocessing,
        max_workers=args.max_workers,
        limit=args.limit,
        patient_id=args.patient_id,
        save_mask=not args.no_save,
        masks=args.masks,
    ).run()


if __name__ == "__main__":
    arm_lesion_removal_entrypoint()
