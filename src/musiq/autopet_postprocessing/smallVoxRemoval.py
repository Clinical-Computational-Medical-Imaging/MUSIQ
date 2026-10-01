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
  <mask>_postProRemoved.nii.gz   the small lesions are ADDED to it, so it holds every lesion removed so far
                                 (lacrimal + arm + small), labels kept, 0 elsewhere. No separate file is made.
Rerunning this script on the same _postPro is safe: nothing more is removed and the removed-lesions
file is unchanged. If lacrimalRemoval.py is rerun (which rebuilds _postPro and _postProRemoved from the
original mask), run arm_lesions.py and this script again afterwards.
Pass --no-save to only log and write nothing.

for running from the terminal: SmallVoxRemover("/path/to/processed", multiprocessing=True, max_workers=5).run()
for running from SLURM: python3 smallVoxRemoval.py --input-dirpath /path/to/processed --masks PETseg_revised
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cc3d  # pip install connected-components-3d
import nibabel as nib
import numpy as np

# data discovery / config / helpers shared with the lacrimal step
from musiq.autopet_postprocessing.lacrimalRemoval import (
    CADS_FILE,
    CONNECTIVITY,
    DEFAULT_MASKS,
    MASK_CHOICES,
    NAMES,
    OUT_SUFFIX,
    REMOVED_SUFFIX,
    resample_cads_to_grid,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
PROSTATE_LABEL = 111  # CADS combined 1..167 map: Prostate
SMALL_VOX_MAX_DEFAULT = 2  # lesions of this size (in voxels) or smaller are candidates for removal
OUT_JSON = "small_vox_removed_components.json"


def save_atomic(arr, ref_img, out_file):
    """Write to a temp name then swap in, so a rerun fully replaces the old file."""
    tmp_file = out_file.with_name(out_file.name.replace(".nii.gz", ".tmp.nii.gz"))
    nib.save(nib.Nifti1Image(arr, ref_img.affine, ref_img.header), str(tmp_file))
    tmp_file.replace(out_file)


class SmallVoxRemover:
    def __init__(
        self,
        input_dirpath,
        max_vox=SMALL_VOX_MAX_DEFAULT,
        multiprocessing=False,
        max_workers=5,
        limit=None,
        patient_id=None,
        masks=None,
        save_mask=True,
    ):
        self.masks = list(masks) if masks else list(DEFAULT_MASKS)
        self.save_mask = save_mask
        self.input_dirpath = input_dirpath
        self.max_vox = max_vox
        self.multiprocessing = multiprocessing
        self.max_workers = max_workers
        self.limit = limit
        self.patient_id = patient_id

    # ---- discovery + dispatch ------------------------------------------- #
    def run(self):
        sub_dirs = sorted(
            dirpath
            for dirpath, _, filenames in os.walk(self.input_dirpath)
            if CADS_FILE in filenames and any(f"{m}{OUT_SUFFIX}.nii.gz" in filenames for m in self.masks)
        )
        print(f"Found {len(sub_dirs)} session(s) with {CADS_FILE} + <mask>{OUT_SUFFIX} for {self.masks}", flush=True)
        if self.patient_id is not None:
            sub_dirs = [d for d in sub_dirs if Path(d).parent.name == self.patient_id]
            print(f"Filtering to patient_id={self.patient_id}: {len(sub_dirs)} session(s)", flush=True)
        if self.limit is not None:
            sub_dirs = sub_dirs[: self.limit]
            print(f"Limiting to first {len(sub_dirs)} session(s)", flush=True)
        if not sub_dirs:
            return

        if self.multiprocessing:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(self.process_wrapper, sub_dirs))
        else:
            results = [self.process_wrapper(d) for d in sub_dirs]

        rows = [r for res in results if res for r in res]  # flatten, skip errored (None/[])
        if rows:
            per_patient = {}
            for r in rows:
                per_patient.setdefault(r["patient_id"], []).append({k: v for k, v in r.items() if k != "patient_id"})

            summary = {
                "total_sessions": len(sub_dirs),
                "small_vox_max": self.max_vox,
                "total_patients_with_removed_lesions": len(per_patient),
                "total_removed_lesions": len(rows),
                "per_patient": [
                    {
                        "patient_id": pid,
                        "removed_lesions": sorted(
                            lesions, key=lambda les: (les["session"], les["mask_name"], les["component_id"])
                        ),
                    }
                    for pid, lesions in sorted(per_patient.items())
                ],
            }

            if self.save_mask:
                out_name = (
                    OUT_JSON if self.patient_id is None else OUT_JSON.replace(".json", f"_{self.patient_id}.json")
                )
                out_json = Path(self.input_dirpath) / out_name
                tmp = out_json.with_suffix(".json.tmp")
                with open(tmp, "w") as f:
                    json.dump(summary, f, indent=2)
                tmp.replace(out_json)
                print(
                    f"\nWrote {len(rows)} removed lesion(s) across {len(per_patient)} patient(s) -> {out_json}",
                    flush=True,
                )
            else:
                print(
                    f"\n(--no-save) {len(rows)} lesion(s) would be removed across {len(per_patient)} patient(s)",
                    flush=True,
                )

            for r in rows:
                print(
                    f"  {r['patient_id']}/{r['session']} [{r['mask_name']}] lesion {r['component_id']}: "
                    f"{r['voxels']} vox, touches {r['touches']}",
                    flush=True,
                )
        else:
            print("\nNo lesions removed in any session.", flush=True)
        print("Small-voxel removal done.", flush=True)

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

        # load CADS once per session and reuse for every mask
        cads_img = nib.squeeze_image(nib.load(str(session / CADS_FILE)))
        cads_native = np.rint(np.asanyarray(cads_img.dataobj)).astype(np.int32)  # labels are ints
        grid_cache = {}  # masks on the same grid share one resampled CADS

        rows = []
        for mask_name in self.masks:
            rows.extend(self.process_mask(session, mask_name, cads_img, cads_native, grid_cache))
        return rows

    def process_mask(self, session, mask_name, cads_img, cads_native, grid_cache):
        tag = f"{session.parent.name}/{session.name}"
        in_path = session / f"{mask_name}{OUT_SUFFIX}.nii.gz"
        if not in_path.exists():
            print(f"  [{mask_name}] no {in_path.name} (run lacrimalRemoval.py first) - skipped", flush=True)
            return []

        mask_img = nib.squeeze_image(nib.load(str(in_path)))
        data = mask_img.get_fdata()

        # resample CADS onto this mask's grid (cached per grid)
        key = (mask_img.shape, mask_img.affine.tobytes())
        if key not in grid_cache:
            grid_cache[key] = resample_cads_to_grid(cads_img, cads_native, mask_img)
        cads = grid_cache[key]
        prostate = cads == PROSTATE_LABEL

        # lesions + vectorized per-lesion stats
        cc, n = cc3d.connected_components((data > 0).astype(np.int32), connectivity=CONNECTIVITY, return_N=True)
        print(
            f"  [{mask_name}] input {in_path.name} | prostate vox on PET grid: {int(prostate.sum())} | lesions: {n}",
            flush=True,
        )
        if n:
            stats = cc3d.statistics(cc)
            sizes, boxes = stats["voxel_counts"], stats["bounding_boxes"]
            prostate_counts = np.bincount(cc.ravel(), weights=prostate.ravel(), minlength=n + 1)

        rows, drop = [], []
        for lid in range(1, n + 1):
            vox = int(sizes[lid])
            in_prostate = int(prostate_counts[lid])

            box = boxes[lid]
            local = cc[box] == lid
            vals, counts = np.unique(cads[box][local], return_counts=True)
            touches = {NAMES.get(int(v), int(v)): int(c) for v, c in zip(vals, counts, strict=False) if v != 0}

            # --- removal decision ---
            remove = vox <= self.max_vox and in_prostate == 0

            # verbose per-lesion line for EVERY lesion (calibration view)
            flag = "  <-- REMOVE (small, outside prostate)" if remove else ""
            print(
                f"    [{tag}] lesion {lid:>2}: {vox:>6} vox | in prostate {in_prostate:>5} | touches {touches}{flag}",
                flush=True,
            )

            if not remove:
                continue  # keep this lesion

            drop.append(lid)
            rows.append(
                {
                    "patient_id": session.parent.name,
                    "session": session.name,
                    "mask_name": mask_name,
                    "component_id": lid,
                    "reason": "small, outside prostate",
                    "voxels": vox,
                    "in_prostate": in_prostate,
                    "touches": touches,
                }
            )

        # the small lesions are ADDED to the shared <mask>_postProRemoved file (one file holds every
        # lesion removed so far). This is a union, so rerunning is safe: lesions already removed from
        # _postPro are not found again, and after a lacrimal rerun (fresh _postPro and a fresh removed
        # file) the small lesions are found again and merged in again.
        if self.save_mask and drop:
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


def small_vox_removal_entrypoint():
    parser = argparse.ArgumentParser(
        description="Remove small lesions outside the prostate from the lacrimal-cleaned masks (<mask>_postPro)."
    )
    parser.add_argument(
        "--input-dirpath",
        type=str,
        required=True,
        help="Processed data root; searched recursively for sessions with CTcads + <mask>_postPro.",
    )
    parser.add_argument(
        "--small-vox-max",
        type=int,
        default=SMALL_VOX_MAX_DEFAULT,
        help="Lesions with this many voxels or fewer (and outside prostate) are removed.",
    )
    parser.add_argument(
        "--masks",
        nargs="+",
        choices=MASK_CHOICES,
        default=DEFAULT_MASKS,
        help="Which mask(s): PETseg (SUV), PETsegSUL (SUL), PETseg_revised (doctors' annotation).",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Log only; do not write any mask or JSON files (calibration/dry run)."
    )
    parser.add_argument(
        "--multiprocessing", action="store_true", help="Process sessions in parallel with a process pool."
    )
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument(
        "--limit", type=int, default=None, help="Only process the first N sessions found (for test runs)."
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Only process sessions for this patient_id (folder name), for debugging.",
    )
    args = parser.parse_args()

    SmallVoxRemover(
        input_dirpath=args.input_dirpath,
        max_vox=args.small_vox_max,
        multiprocessing=args.multiprocessing,
        max_workers=args.max_workers,
        limit=args.limit,
        patient_id=args.patient_id,
        masks=args.masks,
        save_mask=not args.no_save,
    ).run()


if __name__ == "__main__":
    small_vox_removal_entrypoint()
