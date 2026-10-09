import argparse

from .arm_lesions import ArmLesionRemover
from .lacrimalRemoval import DEFAULT_MASKS as _DEFAULT_MASKS
from .lacrimalRemoval import MASK_CHOICES, MIN_LAC_VOX_DEFAULT, NEAR_MM_DEFAULT, LacrimalRemover
from .smallVoxRemoval import SMALL_VOX_MAX_DEFAULT, SmallVoxRemover


def autopet_postprocessing_entrypoint() -> None:
    """Run all three AutoPET postprocessing steps in order: lacrimal → arm → small-vox."""
    parser = argparse.ArgumentParser(
        description=(
            "Run all AutoPET postprocessing steps in sequence: "
            "lacrimal-gland FP removal, arm-lesion removal, small-voxel removal."
        )
    )
    parser.add_argument(
        "--input-dirpath",
        type=str,
        required=True,
        help="Processed data root; searched recursively for sessions.",
    )
    parser.add_argument(
        "--masks",
        nargs="+",
        choices=MASK_CHOICES,
        default=list(_DEFAULT_MASKS),
        help="Which mask(s) to clean: PETseg (SUV), PETsegSUL (SUL), PETseg_revised (doctors' annotation).",
    )
    parser.add_argument(
        "--multiprocessing",
        action="store_true",
        help="Process sessions in parallel with a process pool.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=30,
        help="Worker processes per step when --multiprocessing is set (default: 30).",
    )
    parser.add_argument(
        "--near-mm",
        type=float,
        default=NEAR_MM_DEFAULT,
        help="Proximity radius (mm) for the near-lacrimal-gland fat/subcutaneous rule.",
    )
    parser.add_argument(
        "--min-lac-vox",
        type=int,
        default=MIN_LAC_VOX_DEFAULT,
        help="Remove a lesion overlapping >= this many lacrimal voxels.",
    )
    parser.add_argument(
        "--small-vox-max",
        type=int,
        default=SMALL_VOX_MAX_DEFAULT,
        help="Lesions with this many voxels or fewer (and outside prostate) are removed.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Log only; do not write any mask files (calibration/dry run).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N sessions found (for test runs).",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Only process sessions for this patient_id folder, for debugging.",
    )
    args = parser.parse_args()

    save = not args.no_save

    LacrimalRemover(
        input_dirpath=args.input_dirpath,
        near_mm=args.near_mm,
        min_lac_vox=args.min_lac_vox,
        multiprocessing=args.multiprocessing,
        max_workers=args.max_workers,
        limit=args.limit,
        patient_id=args.patient_id,
        masks=args.masks,
        save_mask=save,
    ).run()

    ArmLesionRemover(
        input_dirpath=args.input_dirpath,
        multiprocessing=args.multiprocessing,
        max_workers=args.max_workers,
        limit=args.limit,
        patient_id=args.patient_id,
        masks=args.masks,
        save_mask=save,
    ).run()

    SmallVoxRemover(
        input_dirpath=args.input_dirpath,
        max_vox=args.small_vox_max,
        multiprocessing=args.multiprocessing,
        max_workers=args.max_workers,
        limit=args.limit,
        patient_id=args.patient_id,
        masks=args.masks,
        save_mask=save,
    ).run()


if __name__ == "__main__":
    autopet_postprocessing_entrypoint()
