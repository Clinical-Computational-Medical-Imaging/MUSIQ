"""
Remove small lesions (<= SMALL_VOX_MAX voxels) that lie outside the prostate.
Run this AFTER lacrimalRemoval.py: it reads <mask>_postPro.nii.gz (the lacrimal-cleaned
mask) and never touches the original mask or the lacrimal outputs.

Choose which mask(s) to process by passing --masks:
  --masks PETseg                            # SUV only
  --masks PETseg_revised                    # Helen's only
  --masks PETseg PETsegSUL                  # both model masks
  --masks PETseg PETsegSUL PETseg_revised   # all three
If --masks is not given, it defaults to PETseg PETsegSUL (SUV + SUL).

Per session and mask:
  <mask>_postPro.nii.gz          updated IN PLACE: the small lesions are removed from it
  <mask>_postProRemoved.nii.gz  the small lesions are ADDED to it, so it holds every lesion removed so far
                                 (lacrimal + arm + small), labels kept, 0 elsewhere. No separate file is made.
Rerunning this script on the same _postPro is safe: nothing more is removed and the removed-lesions
file is unchanged. If lacrimalRemoval.py is rerun (which rebuilds _postPro and _postProRemoved from the
original mask), run arm_lesions.py and this script again afterwards.
Pass --no-save to only log and write nothing.

for running from SLURM: python3 smallVoxRemoval.py --input-dirpath /path/to/processed --masks PETseg_revised
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
import cc3d  # pip install connected-components-3d

# data discovery / config / helpers shared with the lacrimal step
from lacrimalRemoval import (
    DATA_DIR, CADS_FILE, MASK_CHOICES, DEFAULT_MASKS, OUT_SUFFIX, REMOVED_SUFFIX,
    CONNECTIVITY, NAMES, resample_cads_to_grid,
)

PROSTATE_LABEL  = 111   # CADS combined 1..167 map: Prostate
SMALL_VOX_MAX   = 2     # lesions of this size (in voxels) or smaller are candidates for removal


def save_atomic(arr, ref_img, out_file):
    """Write to a temp name then swap in, so a rerun fully replaces the old file."""
    tmp_file = out_file.with_name(out_file.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(nib.Nifti1Image(arr, ref_img.affine, ref_img.header), str(tmp_file))
    tmp_file.replace(out_file)


def process_session(args):
    session, masks, max_vox, save = args
    session = Path(session)
    tag = f"{session.parent.name}/{session.name}"
    try:
        return process_session_inner(session, tag, masks, max_vox, save)
    except Exception as e:
        print(f"  !! ERROR processing {tag}: {e}", flush=True)
        return []


def process_session_inner(session, tag, masks, max_vox, save):
    print(f"\n=== {tag} ===", flush=True)
    cads_img = nib.squeeze_image(nib.load(str(session / CADS_FILE)))
    cads_native = np.rint(np.asanyarray(cads_img.dataobj)).astype(np.int32)   # labels are ints; skip float64 copy
    grid_cache = {}      # masks on the same grid share one resampled CADS

    rows = []
    for mask_name in masks:
        in_path = session / f"{mask_name}{OUT_SUFFIX}.nii.gz"
        if not in_path.exists():
            print(f"  [{mask_name}] no {in_path.name} (run lacrimalRemoval.py first) - skipped", flush=True)
            continue

        mask_img = nib.squeeze_image(nib.load(str(in_path)))
        data = mask_img.get_fdata()

        key = (mask_img.shape, mask_img.affine.tobytes())
        if key not in grid_cache:
            grid_cache[key] = resample_cads_to_grid(cads_img, cads_native, mask_img)
        cads = grid_cache[key]
        prostate = cads == PROSTATE_LABEL

        cc, n = cc3d.connected_components((data > 0).astype(np.int32),
                                          connectivity=CONNECTIVITY, return_N=True)
        print(f"  [{mask_name}] input {in_path.name} | prostate vox on PET grid: {int(prostate.sum())} "
              f"| lesions: {n}", flush=True)
        if n:
            stats = cc3d.statistics(cc)
            sizes, boxes = stats["voxel_counts"], stats["bounding_boxes"]
            prostate_counts = np.bincount(cc.ravel(), weights=prostate.ravel(), minlength=n + 1)

        drop = []
        for lid in range(1, n + 1):
            vox, in_prostate = int(sizes[lid]), int(prostate_counts[lid])
            box = boxes[lid]
            local = cc[box] == lid
            vals, counts = np.unique(cads[box][local], return_counts=True)
            touches = {NAMES.get(int(v), int(v)): int(c) for v, c in zip(vals, counts) if v != 0}

            remove = vox <= max_vox and in_prostate == 0
            flag = "  <-- REMOVE (small, outside prostate)" if remove else ""
            print(f"    [{tag}] lesion {lid:>2}: {vox:>6} vox | in prostate {in_prostate:>5} "
                  f"| touches {touches}{flag}", flush=True)
            if remove:
                drop.append(lid)
                rows.append({"session": tag, "mask_name": mask_name, "component_id": lid,
                             "voxels": vox, "in_prostate": in_prostate, "touches": touches})

        if save and drop:
            # the small lesions are ADDED to the shared <mask>_postProRemoved file (one file holds every
            # lesion removed so far). This is a union, so rerunning is safe: lesions already removed from
            # _postPro are not found again, and after a lacrimal rerun (fresh _postPro and a fresh removed
            # file) the small lesions are found again and merged in again.
            removed_file = session / f"{mask_name}{REMOVED_SUFFIX}.nii.gz"
            dtype = mask_img.get_data_dtype()
            removed = np.isin(cc, drop)
            removed_arr = np.where(removed, data, 0)
            if removed_file.exists():
                old = np.asanyarray(nib.load(str(removed_file)).dataobj).reshape(data.shape)
                removed_arr = np.where(old > 0, old, removed_arr)
            # removed file first, _postPro second: a crash in between leaves the lesions in _postPro, so a
            # rerun finds them again and merges them again (no lesion can end up in neither file)
            save_atomic(removed_arr.astype(dtype), mask_img, removed_file)
            save_atomic(np.where(removed, 0, data).astype(dtype), mask_img, in_path)
            print(f"  [{mask_name}] updated {in_path.name} (in place), added to {removed_file.name}", flush=True)
        print(f"  [{mask_name}] {n} lesion(s), removed {len(drop)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Remove small lesions outside the prostate from the lacrimal-cleaned masks (<mask>_postPro)."
    )
    parser.add_argument("--input-dirpath", type=str, default=str(DATA_DIR),
                        help="Processed data root; searched recursively for sessions with CTcads + <mask>_postPro.")
    parser.add_argument("--masks", nargs="+", choices=MASK_CHOICES, default=DEFAULT_MASKS,
                        help="Which mask(s): PETseg (SUV), PETsegSUL (SUL), PETseg_revised (doctors' annotation).")
    parser.add_argument("--small-vox-max", type=int, default=SMALL_VOX_MAX,
                        help="Lesions with this many voxels or fewer (and outside prostate) are removed.")
    parser.add_argument("--no-save", action="store_true", help="Log only; do not write any mask files.")
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N sessions found.")
    parser.add_argument("--patient-id", type=str, default=None, help="Only this patient_id (folder name).")
    a = parser.parse_args()

    sessions = sorted(
        d for d, _, files in os.walk(a.input_dirpath)
        if CADS_FILE in files and any(f"{m}{OUT_SUFFIX}.nii.gz" in files for m in a.masks)
    )
    print(f"Found {len(sessions)} session(s) with {CADS_FILE} + <mask>{OUT_SUFFIX} for {a.masks}", flush=True)
    if a.patient_id is not None:
        sessions = [d for d in sessions if Path(d).parent.name == a.patient_id]
        print(f"Filtering to patient_id={a.patient_id}: {len(sessions)} session(s)", flush=True)
    if a.limit is not None:
        sessions = sessions[:a.limit]
        print(f"Limiting to first {len(sessions)} session(s)", flush=True)
    if not sessions:
        return

    jobs = [(s, a.masks, a.small_vox_max, not a.no_save) for s in sessions]
    if a.multiprocessing:
        with ProcessPoolExecutor(max_workers=a.max_workers) as ex:
            results = list(ex.map(process_session, jobs))
    else:
        results = [process_session(j) for j in jobs]

    rows = [r for res in results for r in res]
    print(f"\nRemoved {len(rows)} small lesion(s) across {len(sessions)} session(s):", flush=True)
    for r in rows:
        print(f"  {r['session']} [{r['mask_name']}] lesion {r['component_id']}: "
              f"{r['voxels']} vox, touches {r['touches']}", flush=True)
    print("\nSmall-voxel removal done.", flush=True)


if __name__ == "__main__":
    main()
