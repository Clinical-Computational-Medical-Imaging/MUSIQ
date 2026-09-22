"""
Choose which mask(s) to process by passing --masks when running the code:
  --masks PETseg                            # SUV only
  --masks PETseg_revised                    # Helen's only
  --masks PETseg PETsegSUL                  # both model masks
  --masks PETseg PETsegSUL PETseg_revised   # all three
If --masks is not given, it defaults to PETseg PETsegSUL (SUV + SUL).

for runing from the terminal:LacrimalRemover("/path/to/processed", multiprocessing=True, max_workers=5).run()
for runing from SLURM: python3 /path/to/processed/lacrimalRemoval.py --multiprocessing --max-workers 5
Remove lacrimal-gland FP from masks, using CADS labels.

A PET lesion is removed if:
  (a) it overlaps a CADS lacrimal label at all (>= MIN_LAC_VOX voxels), OR
  (b) it lies within NEAR_MM of the lacrimal gland AND is dominantly
      background / subcutaneous tissue (gland uptake spilling into nearby fat).
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
import cc3d  # pip install connected-components-3d
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt, binary_dilation

from labelMap import labelmap_cads

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DATA_DIR            = Path("/data/Data2/MULTIPRO/processed/") #/home/amka26/Documents/test_data
LACRIMAL_LABELS     = [144, 145]          # CADS: Lacrimal gland L / R
SUBCUTANEOUS_LABEL  = 158                 # CADS: Subcutaneous tissue
ALLOWED_DOMINANT    = {0, SUBCUTANEOUS_LABEL}   # background(0) / subcutaneous -> removable when near gland
NEAR_MM_DEFAULT     = 15.0                 # "near lacrimal" radius for the fat/spill-out rule
MIN_LAC_VOX_DEFAULT = 1                    # a lesion touching >= this many lacrimal voxels is removed
CONNECTIVITY        = 18                   # keep consistent with your metrics/compare stage
LACRIMAL_DILATE_VOX = 3                    # dilate lacrimal labels by this many native-CT voxels before
                                           # downsampling, so a gland smaller than one PET voxel still
                                           # survives nearest-neighbor resampling onto the PET grid

MASK_CHOICES  = ["PETseg",            # SUV
                 "PETsegSUL",         # SUL
                 "PETseg_revised"]    # doctors' annotation (label mask)
MASK_EXTS     = (".nii.gz", ".nii")
DEFAULT_MASKS = ["PETseg", "PETsegSUL"]
OUT_SUFFIX    = "_postPro"            # cleaned mask saved as <mask><suffix>.nii.gz
REMOVED_SUFFIX = "_postProRemoved"   # only the removed lesions, saved as <mask><suffix>.nii.gz
CADS_FILE       = "CTcads.nii.gz"
NECESSARY_FILES = [CADS_FILE]

def find_mask_file(session, stem):
    """Path of <stem>.nii.gz / <stem>.nii in this session, or None."""
    for ext in MASK_EXTS:
        p = Path(session) / f"{stem}{ext}"
        if p.exists():
            return p
    return None
OUT_JSON        = "lacrimal_removed_components.json"

NAMES = labelmap_cads

def dilate_labels(arr, labels, iters):
    """Grow the given labels outward into background only,
    leaving other structures intact."""
    out = arr.copy()
    for label in labels:
        mask = arr == label
        if not mask.any():
            continue
        grown = binary_dilation(mask, iterations=iters)
        out[grown & (arr == 0)] = label
    return out

def resample_cads_to_grid(img, arr, ref_img):
    """Put a CADS label array onto ref_img's grid with nearest-neighbour;
    no-op if grids already match."""
    needs_resample = (img.shape != ref_img.shape or
                      not np.allclose(img.affine, ref_img.affine))
    if not needs_resample:
        return arr
    src = nib.Nifti1Image(arr, img.affine, img.header)
    res = resample_from_to(src, (ref_img.shape, ref_img.affine), order=0)
    return np.rint(res.get_fdata()).astype(np.int32)

def distance_field_from_native(img, arr, labels):
    """Distance (mm) to the nearest voxel of any of `labels`, computed on an
    already-loaded native-resolution CADS array. Returns None if none of
    `labels` are present anywhere in this CADS image."""
    mask = np.isin(arr, labels)
    if not mask.any():
        return None
    spacing = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    return distance_transform_edt(~mask, sampling=spacing).astype(np.float32)


def resample_distance_to_grid(img, dist, ref_img):
    """Put a distance field on ref_img's grid (linear); returns
    (dist, lacrimal_found)."""
    if dist is None:
        return np.full(ref_img.shape, np.inf, dtype=np.float32), False
    if img.shape == ref_img.shape and np.allclose(img.affine, ref_img.affine):
        return dist, True
    src = nib.Nifti1Image(dist, img.affine, img.header)
    res = resample_from_to(src, (ref_img.shape, ref_img.affine), order=1)
    return res.get_fdata().astype(np.float32), True


class LacrimalRemover:
    def __init__(self, input_dirpath, near_mm=NEAR_MM_DEFAULT, min_lac_vox=MIN_LAC_VOX_DEFAULT,
                 multiprocessing=False, max_workers=30, limit=None, patient_id=None,
                 masks=None, save_mask=True):
        self.masks = list(masks) if masks else list(DEFAULT_MASKS)
        self.save_mask = save_mask
        self.input_dirpath = input_dirpath
        self.near_mm = near_mm
        self.min_lac_vox = min_lac_vox
        self.multiprocessing = multiprocessing
        self.max_workers = max_workers
        self.limit = limit
        self.patient_id = patient_id

    # ---- discovery + dispatch ------------------------------------------- #
    def run(self):
        sub_dirs = sorted(
            dirpath
            for dirpath, _, filenames in os.walk(self.input_dirpath)
            if all(f in filenames for f in NECESSARY_FILES)
            and any(m + ext in filenames for m in self.masks for ext in MASK_EXTS)
        )
        print(f"Found {len(sub_dirs)} session(s) with {NECESSARY_FILES} + any of {self.masks}", flush=True)
        if self.patient_id is not None:
            sub_dirs = [d for d in sub_dirs if Path(d).parent.name == self.patient_id]
            print(f"Filtering to patient_id={self.patient_id}: {len(sub_dirs)} session(s)", flush=True)
        if self.limit is not None:
            sub_dirs = sub_dirs[:self.limit]
            print(f"Limiting to first {len(sub_dirs)} session(s)", flush=True)
        if not sub_dirs:
            return

        if self.multiprocessing:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self.process_wrapper, sub_dirs))
        else:
            results = [self.process_wrapper(d) for d in sub_dirs]

        rows = [r for res in results if res for r in res]   # flatten, skip errored (None/[])
        if rows:
            per_patient = {}
            for r in rows:
                per_patient.setdefault(r["patient_id"], []).append({
                    k: v for k, v in r.items() if k != "patient_id"
                })

            summary = {
                "total_sessions": len(sub_dirs),
                "total_patients_with_removed_lesions": len(per_patient),
                "total_removed_lesions": len(rows),
                "per_patient": [
                    {
                        "patient_id": pid,
                        "removed_lesions": sorted(lesions, key=lambda l: (l["session"], l["mask_name"], l["component_id"])),
                    }
                    for pid, lesions in sorted(per_patient.items())
                ],
            }

            out_name = OUT_JSON if self.patient_id is None else \
                OUT_JSON.replace(".json", f"_{self.patient_id}.json")
            out_json = Path(self.input_dirpath) / out_name
            tmp = out_json.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(summary, f, indent=2)
            tmp.replace(out_json)
            print(f"\nWrote {len(rows)} removed lesion(s) across {len(per_patient)} patient(s) -> {out_json}",
                  flush=True)
        else:
            print("\nNo lesions removed in any session.", flush=True)
        print("Lacrimal removal done.", flush=True)

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
        raw = nib.load(str(session / CADS_FILE))

        # native-resolution CADS work (dilation, native distance transform) is
        # expensive on this big grid -- do it ONCE per session and reuse for
        # both PETseg and PETsegSUL instead of redoing it per mask
        cads_img = nib.squeeze_image(raw)
        arr = np.rint(cads_img.get_fdata()).astype(np.int32)
        arr_dil = dilate_labels(arr, LACRIMAL_LABELS, LACRIMAL_DILATE_VOX)
        dist_nat = distance_field_from_native(cads_img, arr, LACRIMAL_LABELS)

        rows = []
        for mask_name in self.masks:
            rows.extend(self.process_mask(session, mask_name, cads_img, arr_dil, dist_nat))
        return rows

    def process_mask(self, session, mask_name, cads_img, arr_dil, dist_nat):
        mask_path = find_mask_file(session, mask_name)
        if mask_path is None:
            return []

        mask_img = nib.squeeze_image(nib.load(str(mask_path)))
        data = mask_img.get_fdata()

        # resample the already-dilated native labels onto this mask's grid
        cads = resample_cads_to_grid(cads_img, arr_dil, mask_img)
        lac = np.isin(cads, LACRIMAL_LABELS)

        # resample the already-computed native distance field onto this mask's grid
        dist, lac_found = resample_distance_to_grid(cads_img, dist_nat, mask_img)
        if not lac_found:
            print(f"  [{mask_name}] no lacrimal voxels in native CADS - nothing to remove", flush=True)

        # lesions + vectorized per-lesion stats
        cc, n = cc3d.connected_components((data > 0).astype(np.int32),
                                          connectivity=CONNECTIVITY, return_N=True)
        stats = cc3d.statistics(cc)
        sizes = stats["voxel_counts"]
        boxes = stats["bounding_boxes"]
        lac_counts = np.bincount(cc.ravel(), weights=lac.ravel(), minlength=n + 1)
        print(f"  [{mask_name}] lesions: {n}", flush=True)

        rows, drop = [], []
        for lid in range(1, n + 1):
            vox = int(sizes[lid])
            in_lac = int(lac_counts[lid])
            frac_lac = in_lac / vox if vox else 0.0

            box = boxes[lid]
            local = cc[box] == lid
            min_dist = float(dist[box][local].min())

            # CADS label composition of this lesion (fractions of its voxels)
            vals, counts = np.unique(cads[box][local], return_counts=True)
            fracs = {int(v): int(c) / vox for v, c in zip(vals, counts)}
            bg = fracs.pop(0, 0.0)     # label 0 = outside any CADS structure
            dom = max(fracs, key=fracs.get) if fracs else 0

            # --- removal decision ---
            touches_lac = in_lac >= self.min_lac_vox
            near_fat = (min_dist <= self.near_mm and
                        (dom in ALLOWED_DOMINANT or bg >= 0.5))
            remove = touches_lac or near_fat
            if touches_lac:
                reason = "lacrimal overlap"
            elif near_fat:
                reason = "near-lacrimal fat"
            else:
                reason = "kept"

            # verbose per-lesion line for EVERY lesion (calibration view)
            touches = {NAMES.get(int(v), int(v)): int(c)
                       for v, c in zip(vals, counts) if v != 0}
            tag = f"  <-- REMOVE ({reason})" if remove else ""
            print(f"    lesion {lid:>2}: {vox:>6} vox | lacrimal {in_lac:>5} "
                  f"({frac_lac:6.1%}) | dist {min_dist:5.1f}mm | bg {bg:4.0%} "
                  f"| touches {touches}{tag}", flush=True)

            if not remove:
                continue   # keep this lesion

            drop.append(lid)
            ranked = sorted(fracs.items(), key=lambda kv: -kv[1])[:3]  # top-3 organs
            rows.append({
                "patient_id": session.parent.name,
                "session": session.name,
                "mask_name": mask_name,
                "component_id": lid,
                "reason": reason,
                "voxels": vox,
                "lacrimal_voxels": in_lac,
                "min_dist_mm": round(min_dist, 1),
                "background_fraction": round(bg, 2),
                "organs": [
                    {"name": NAMES.get(label, str(label)), "fraction": round(frac, 2)}
                    for label, frac in ranked
                ],
            })

        # save cleaned mask (same geometry, original label values kept; removed lesions -> 0)
        # always rebuilt from the ORIGINAL mask (never from a previous _postPro file) and
        # written to a temp file then swapped in, so a rerun fully replaces the old output
        if self.save_mask:
            removed = np.isin(cc, drop)
            dtype = mask_img.get_data_dtype()
            outputs = {
                f"{mask_name}{OUT_SUFFIX}": np.where(removed, 0, data).astype(dtype),      # cleaned
                f"{mask_name}{REMOVED_SUFFIX}": np.where(removed, data, 0).astype(dtype),  # removed only
            }
            for stem, arr_out in outputs.items():
                out_file = session / f"{stem}.nii.gz"
                tmp_file = session / f"{stem}.tmp.nii.gz"
                nib.save(nib.Nifti1Image(arr_out, mask_img.affine, mask_img.header), str(tmp_file))
                tmp_file.replace(out_file)
                print(f"  [{mask_name}] saved {out_file.name}", flush=True)
        print(f"  [{mask_name}] {n} lesion(s), removed {len(drop)}", flush=True)
        return rows


def lacrimal_removal_entrypoint():
    parser = argparse.ArgumentParser(
        description="Remove lacrimal-gland false positives (and near-gland fat) from PETseg / PETsegSUL using CADS."
    )
    parser.add_argument("--input-dirpath", type=str, default=str(DATA_DIR),
                        help="Processed data root; searched recursively for sessions with CTcads + PETseg.")
    parser.add_argument("--near-mm", type=float, default=NEAR_MM_DEFAULT,
                        help="Proximity radius (mm) for the near-gland fat/subcutaneous rule.")
    parser.add_argument("--min-lac-vox", type=int, default=MIN_LAC_VOX_DEFAULT,
                        help="Remove a lesion overlapping >= this many lacrimal voxels.")
    parser.add_argument("--masks", nargs="+", choices=MASK_CHOICES, default=DEFAULT_MASKS,
                        help="Which mask(s) to clean: PETseg (SUV), PETsegSUL (SUL), PETseg_revised (doctors' annotation).")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not write the cleaned <mask>_postPro.nii.gz files (calibration/dry run).")
    parser.add_argument("--multiprocessing", action="store_true",
                        help="Process sessions in parallel with a process pool.")
    parser.add_argument("--max-workers", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N sessions found (for test runs).")
    parser.add_argument("--patient-id", type=str, default=None,
                        help="Only process sessions for this patient_id (folder name), for debugging.")
    args = parser.parse_args()

    LacrimalRemover(
        input_dirpath=args.input_dirpath,
        near_mm=args.near_mm,
        min_lac_vox=args.min_lac_vox,
        multiprocessing=args.multiprocessing,
        max_workers=args.max_workers,
        limit=args.limit,
        patient_id=args.patient_id,
        masks=args.masks,
        save_mask=not args.no_save,
    ).run()


if __name__ == "__main__":
    lacrimal_removal_entrypoint()