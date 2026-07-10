import json
import multiprocessing as mp
import os
import pathlib as plb
import platform
import subprocess

from .utils import RESERVED_PROCESSED_DIRS, create_logger, setup_series_keywords

_REPO_ROOT = plb.Path(__file__).parent.parent.parent


def build_cohort_info(output_dirpath: str) -> dict:
    """Rebuild ``cohort_info.json`` by merging every ``patient_info.json`` (orphaned patients dropped)."""
    cohort_info = {}
    for dirpath, dirnames, filenames in os.walk(output_dirpath):
        # Don't descend into non-patient dirs (e.g. cads_staging intermediates).
        dirnames[:] = [d for d in dirnames if d not in RESERVED_PROCESSED_DIRS]
        if "patient_info.json" in filenames:
            with open(os.path.join(dirpath, "patient_info.json")) as json_file:
                patient_info = json.load(json_file)
                patient_id = patient_info.get("PatientID", "Unknown")
                cohort_info.update({patient_id: patient_info})

    with open(os.path.join(output_dirpath, "cohort_info.json"), "w") as f:
        json.dump(cohort_info, f)

    return cohort_info


class Workflow:
    def __init__(
        self,
        input_dirpath: str,
        output_dirpath: str,
        tasks: list[str] | None = None,
        cads_tasks: list[str] | None = None,
        cads_work_dir: str | None = None,
        cads_cpu: bool = False,
        pet_metric: str | list[str] | None = None,
        mask_sources: list[str] | None = None,
        label_dirpath: str | None = None,
        label_glob: str | None = None,
        radiomics_workers: int = 1,
        ct_primary_keywords: list[str] | None = None,
        ct_secondary_keywords: list[str] | None = None,
        ct_exclusion_keywords: list[str] | None = None,
        pt_primary_keywords: list[str] | None = None,
        pt_secondary_keywords: list[str] | None = None,
        pt_exclusion_keywords: list[str] | None = None,
        mr_primary_keywords: list[str] | None = None,
        mr_secondary_keywords: list[str] | None = None,
        mr_exclusion_keywords: list[str] | None = None,
        boa_weights_path: str | None = None,
        boa_image: str = "shipai/boa-cli",
        boa_fast: bool = False,
        boa_no_pdf: bool = False,
        boa_device: str = "gpu",
        boa_reuse_total: bool = True,
        boa_runtime: str = "docker",
        boa_sif: str | None = None,
    ) -> None:
        """Configure a MUSIQ workflow run.

        Runs the selected task stages over the input tree, writing results into the processed tree.
        """
        self.input_dirpath = input_dirpath
        self.output_dirpath = output_dirpath
        if pet_metric is None:
            pet_metric = ["SUV", "SUL"]
        self.pet_metric = pet_metric
        # mask_sources: 'auto' (PETseg -> TumorStats[SUL]) and/or 'revised' (physician label -> TumorStatsRevised)
        if mask_sources is None:
            mask_sources = ["auto"]
        for s in mask_sources:
            if s not in ("auto", "revised"):
                raise ValueError(f"mask_sources entries must be 'auto' or 'revised', got '{s}'")
        self.mask_sources = mask_sources
        self.label_dirpath = label_dirpath
        self.label_glob = label_glob
        self.radiomics_workers = radiomics_workers
        if tasks is None:
            tasks = [
                "series_selection",
                "radiomics",
                "autopet",
                "totalsegmentator",
                "tumor",
                "plot",
                "moose",
                "muscle_fat",
                "sul",
                "cads",
                "boa",
            ]
        else:
            tasks_error = bool(
                any(
                    t
                    not in [
                        "series_selection",
                        "radiomics",
                        "autopet",
                        "totalsegmentator",
                        "tumor",
                        "plot",
                        "moose",
                        "muscle_fat",
                        "sul",
                        "cads",
                        "boa",
                    ]
                    for t in tasks or []
                )
            )
            if tasks_error:
                raise ValueError(
                    "Invalid tasks specified. Possible values are: "
                    "'series_selection', 'radiomics', 'autopet', "
                    "'totalsegmentator', 'tumor', 'plot', 'moose', "
                    "'muscle_fat', 'sul', 'cads', 'boa'."
                )

        self.series_selection = "series_selection" in (tasks or [])
        self.autopet = "autopet" in (tasks or [])
        self.cads = "cads" in (tasks or [])
        self.totalsegmentator = "totalsegmentator" in (tasks or [])
        self.muscle_fat = "muscle_fat" in (tasks or [])
        self.sul = "sul" in (tasks or [])
        self.moose = "moose" in (tasks or [])
        self.radiomics = "radiomics" in (tasks or [])
        self.tumor = "tumor" in (tasks or [])
        self.plot = "plot" in (tasks or [])
        self.boa = "boa" in (tasks or [])

        self.boa_weights_path = boa_weights_path
        self.boa_image = boa_image
        self.boa_fast = boa_fast
        self.boa_no_pdf = boa_no_pdf
        self.boa_device = boa_device
        self.boa_reuse_total = boa_reuse_total
        self.boa_runtime = boa_runtime
        self.boa_sif = boa_sif

        cads_tasks_error = bool(
            any(
                t
                not in [
                    "551",
                    "552",
                    "553",
                    "554",
                    "555",
                    "556",
                    "557",
                    "558",
                    "559",
                    "all",
                    "",
                    None,
                ]
                for t in cads_tasks or []
            )
        )

        if cads_tasks_error:
            logger.warning(
                "Wrong input tasks for CADS model."
                "Please use one of the following: "
                "all, 551, 552, 553, 554, 555, 556, 557, 558, 559"
            )
            self.cads = False

        self.cads_tasks = cads_tasks
        self.cads_work_dir = cads_work_dir
        self.cads_cpu = cads_cpu

        self.series_keywords = setup_series_keywords(
            ct_primary_keywords,
            ct_secondary_keywords,
            ct_exclusion_keywords,
            pt_primary_keywords,
            pt_secondary_keywords,
            pt_exclusion_keywords,
            mr_primary_keywords,
            mr_secondary_keywords,
            mr_exclusion_keywords,
        )

    def _revised_label_kwargs(self, source: str) -> dict:
        """Label lookup kwargs for the extractors — only meaningful for the 'revised' mask source."""
        if source != "revised":
            return {}
        kwargs = {}
        if self.label_dirpath:
            kwargs["label_dirpath"] = self.label_dirpath
        if self.label_glob:
            kwargs["label_glob"] = self.label_glob
        return kwargs

    def run(self) -> None:
        os.makedirs(self.output_dirpath, exist_ok=True)

        if self.series_selection:
            from .series_selection import SeriesSelection

            logger.info("\n" + "#" * 50 + "\nStarting Series Selection\n" + "#" * 50)
            SeriesSelection(
                input_dirpath=self.input_dirpath,
                output_dirpath=self.output_dirpath,
                series_keywords=self.series_keywords,
            ).run()

        if self.totalsegmentator:
            from .totalsegmentator_inference import TotalSegmentatorInference

            logger.info("\n" + "#" * 50 + "\nStarting Total Segmentator Inference\n" + "#" * 50)
            TotalSegmentatorInference(
                input_dirpath_processed=self.output_dirpath,
            ).run()

        if self.muscle_fat:
            from .totalsegmentator_muscle_fat import TotalSegmentatorMuscleFat

            logger.info("\n" + "#" * 50 + "\nStarting TotalSegmentator Muscle Fat computation\n" + "#" * 50)
            TotalSegmentatorMuscleFat(
                input_dirpath_processed=self.output_dirpath,
            ).run()

        if self.sul:
            from .sul_computation import SulInference

            logger.info("\n" + "#" * 50 + "\nStarting SUL computation\n" + "#" * 50)
            SulInference(
                input_dirpath_processed=self.output_dirpath,
            ).run()

        if self.boa:
            from .boa_inference import BoaInference

            logger.info("\n" + "#" * 50 + "\nStarting BOA Body Composition Analysis\n" + "#" * 50)
            BoaInference(
                input_dirpath_processed=self.output_dirpath,
                weights_dirpath=self.boa_weights_path,
                image=self.boa_image,
                fast_bca=self.boa_fast,
                no_pdf=self.boa_no_pdf,
                device=self.boa_device,
                reuse_total=self.boa_reuse_total,
                runtime=self.boa_runtime,
                sif_path=self.boa_sif,
            ).run()

        if self.autopet:
            from .autopet_inference import AutopetInference

            logger.info("\n" + "#" * 50 + "\nStarting Autopet Inference\n" + "#" * 50)
            AutopetInference(
                input_dirpath_processed=self.output_dirpath,
                autopet_checkpoint_dirpath=_REPO_ROOT
                / "autopet-3-model/Dataset222_AutoPETIII_2024/"
                / "autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3",
                pet_metric=self.pet_metric,
            ).run()

        if self.cads:
            from .cads_inference import CadsInference

            logger.info("\n" + "#" * 50 + "\nStarting CADS Inference\n" + "#" * 50)
            CadsInference(
                input_dirpath_processed=self.output_dirpath,
                tasks=self.cads_tasks,
                work_dir=self.cads_work_dir,
                use_cpu=self.cads_cpu,
            ).run()

        if self.moose:
            logger.info("\n" + "#" * 50 + "\nStarting Moose Inference\n" + "#" * 50)
            # Run Moosez with the Python interpreter from another venv:
            if platform.system() == "Windows":
                logger.info("Using Windows paths to start the moose_venv Python interpreter.")
                moose_venv_python = os.path.join(os.getcwd(), ".venv_moose", "Scripts", "python.exe")
            else:
                logger.info("Using Linux paths to start the moose_venv Python interpreter.")
                moose_venv_python = os.path.join(os.getcwd(), ".venv_moose", "bin", "python")

            moose_script = os.path.join(os.getcwd(), "src", "musiq", "moose_inference.py")
            cmd = [
                moose_venv_python,
                moose_script,
                "--input-dirpath",
                self.output_dirpath,
                "--moose_task",
                "clin_ct_organs",
            ]
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(result.stderr)
            except subprocess.CalledProcessError as e:
                logger.error("Error during Moose inference:\n" + e.stderr)

        if self.radiomics:
            from .radiomics_extraction import RadiomicsExtractor

            logger.info("\n" + "#" * 50 + "\nStarting Radiomics Computation\n" + "#" * 50)
            for source in self.mask_sources:
                # Revised runs on SUV only (the manual label is drawn once, independent of SUV/SUL).
                metric = self.pet_metric if source == "auto" else ["SUV"]
                RadiomicsExtractor(
                    input_dirpath_processed=self.output_dirpath,
                    pet_metric=metric,
                    mask_source=source,
                    workers=self.radiomics_workers,
                    **self._revised_label_kwargs(source),
                ).run()

        if self.tumor:
            from .tumor_info_extraction import TumorInfoExtraction

            logger.info("\n" + "#" * 50 + "\nStarting Tumor Info Extraction\n" + "#" * 50)
            for source in self.mask_sources:
                metric = self.pet_metric if source == "auto" else ["SUV"]
                TumorInfoExtraction(
                    input_dirpath_processed=self.output_dirpath,
                    pet_metric=metric,
                    mask_source=source,
                    workers=self.radiomics_workers,
                    **self._revised_label_kwargs(source),
                ).run()

        logger.info("\n" + "#" * 50 + "\nStarting Cohort Info Creation\n" + "#" * 50)
        build_cohort_info(self.output_dirpath)

        if self.plot:
            from .plot_summary import PlotSummary

            logger.info("\n" + "#" * 50 + "\nStarting Plot Summary\n" + "#" * 50)
            PlotSummary(
                input_dirpath_processed=self.output_dirpath,
            ).run()

        logger.info("\n" + "#" * 50 + "\nWorkflow completed successfully\n" + "#" * 50)


def workflow_entrypoint():
    """Entrypoint to run the MUSIQ workflow."""
    # Use spawn start method
    mp.set_start_method("spawn", force=True)

    global logger
    logger = create_logger("musiq.workflow")

    import argparse

    parser = argparse.ArgumentParser(description="Arguments for the MUSIQ pipeline.")
    parser.add_argument("--input-dirpath", help="Path to PET/CT input directory.", required=True)
    parser.add_argument("--output-dirpath", help="Path to designated output directory.", required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="List of tasks to run. Possible values: series_selection, "
        "radiomics, autopet, totalsegmentator, tumor, plot, moose, muscle_fat, sul, cads, boa.",
        default=None,
    )
    parser.add_argument(
        "--cads-tasks",
        nargs="+",
        help="List of tasks to run in the CADS model. Possible tasks (different tasks can be separated by spaces): "
        "all 551 552 553 554 555 556 557 558 559",
        default=None,
    )
    parser.add_argument(
        "--cads-work-dir",
        type=str,
        default=None,
        help="Staging dir for CADS intermediates. Default: <output-dirpath>/cads_staging.",
    )
    parser.add_argument("--cads-cpu", action="store_true", help="Run CADS inference on CPU instead of GPU.")
    # nargs="*": passing a keyword flag empty selects all series (anonymized cohorts w/ empty descriptions)
    parser.add_argument(
        "--ct-primary-keywords",
        nargs="*",
        help="Keywords to look for in CT series descriptions for default selection. Pass empty to select all series.",
    )
    parser.add_argument(
        "--ct-secondary-keywords",
        nargs="*",
        help="Keywords to look for in CT series descriptions for alternative selection.",
    )
    parser.add_argument("--ct-exclusion-keywords", nargs="*", help="Keywords to exclude CT series from selection.")
    parser.add_argument(
        "--pt-primary-keywords",
        nargs="*",
        help="Keywords to look for in PT series descriptions for default selection.",
    )
    parser.add_argument(
        "--pt-secondary-keywords",
        nargs="*",
        help="Keywords to look for in PT series descriptions for alternative selection.",
    )
    parser.add_argument("--pt-exclusion-keywords", nargs="*", help="Keywords to exclude PT series from selection.")
    parser.add_argument(
        "--mr-primary-keywords",
        nargs="*",
        help="Keywords to look for in MR series descriptions for default selection.",
    )
    parser.add_argument(
        "--mr-secondary-keywords",
        nargs="*",
        help="Keywords to look for in MR series descriptions for alternative selection.",
    )
    parser.add_argument("--mr-exclusion-keywords", nargs="*", help="Keywords to exclude MR series from selection.")
    parser.add_argument(
        "--pet-metric",
        type=str,
        nargs="+",
        choices=["SUV", "SUL"],
        default=["SUV", "SUL"],
        help="PET metric(s) to use. Pass one or both: --pet-metric SUV SUL (default: SUV SUL)",
    )
    parser.add_argument(
        "--mask-source",
        dest="mask_sources",
        type=str,
        nargs="+",
        choices=["auto", "revised"],
        default=["auto"],
        help="Mask(s) for radiomics/tumor: 'auto' (PETseg -> TumorStats[SUL]) and/or 'revised' "
        "(physician label -> TumorStatsRevised, SUV only). Pass both to compute all: "
        "--mask-source auto revised (default: auto).",
    )
    parser.add_argument(
        "--label-dirpath",
        type=str,
        default=None,
        help="Physician label location for --mask-source revised. Omit to look inside each study dir "
        "(e.g. MULTIPRO PETseg_revised.nii); set to a parallel labels root to look under "
        "<label_dirpath>/<PatientID>/ (e.g. Scheurer).",
    )
    parser.add_argument(
        "--label-glob",
        type=str,
        default=None,
        help="Filename pattern of the revised label for --mask-source revised (wildcards allowed), "
        "e.g. 'PETseg_revised.nii' (default in code) or '*segmentation_Tumor.nii' (Scheurer).",
    )
    parser.add_argument(
        "--radiomics-workers",
        type=int,
        default=1,
        help="Parallel worker processes for the radiomics and tumor stages (patients run in parallel). "
        "Default: 1 (serial).",
    )
    parser.add_argument(
        "--boa-weights-path",
        type=str,
        default=None,
        help="Local BOA weights directory, mounted into the BOA container at /app/weights.",
    )
    parser.add_argument("--boa-image", type=str, default="shipai/boa-cli", help="BOA Docker image tag.")
    parser.add_argument("--boa-fast", action="store_true", help="Use the fast single-fold BCA variant.")
    parser.add_argument("--boa-no-pdf", action="store_true", help="Skip the BCA PDF report (keep JSON measurements).")
    parser.add_argument("--boa-device", type=str, default="gpu", help="BOA device: gpu, cuda or cpu.")
    parser.add_argument(
        "--boa-no-reuse-total",
        action="store_true",
        help="Let BOA compute its own total segmentation instead of reusing CTseg.nii.gz.",
    )
    parser.add_argument(
        "--boa-runtime",
        type=str,
        default="docker",
        choices=["docker", "apptainer"],
        help="Container runtime for BOA (default: docker). Use 'apptainer' on HPC clusters.",
    )
    parser.add_argument(
        "--boa-sif",
        type=str,
        default=None,
        help="Path to the BOA Apptainer/Singularity image (.sif). Required when --boa-runtime apptainer.",
    )
    args = parser.parse_args()

    if not args.input_dirpath or not args.output_dirpath:
        logger.error("Input and output directories must be specified.")
    logger.info(f"Starting the MUSIQ pipeline with arguments: {args}")

    Workflow(
        input_dirpath=args.input_dirpath,
        output_dirpath=args.output_dirpath,
        tasks=args.tasks,
        cads_tasks=args.cads_tasks,
        cads_work_dir=args.cads_work_dir,
        cads_cpu=args.cads_cpu,
        ct_primary_keywords=args.ct_primary_keywords,
        ct_secondary_keywords=args.ct_secondary_keywords,
        ct_exclusion_keywords=args.ct_exclusion_keywords,
        pt_primary_keywords=args.pt_primary_keywords,
        pt_secondary_keywords=args.pt_secondary_keywords,
        pt_exclusion_keywords=args.pt_exclusion_keywords,
        mr_primary_keywords=args.mr_primary_keywords,
        mr_secondary_keywords=args.mr_secondary_keywords,
        mr_exclusion_keywords=args.mr_exclusion_keywords,
        pet_metric=args.pet_metric,
        mask_sources=args.mask_sources,
        label_dirpath=args.label_dirpath,
        label_glob=args.label_glob,
        radiomics_workers=args.radiomics_workers,
        boa_weights_path=args.boa_weights_path,
        boa_image=args.boa_image,
        boa_fast=args.boa_fast,
        boa_no_pdf=args.boa_no_pdf,
        boa_device=args.boa_device,
        boa_reuse_total=not args.boa_no_reuse_total,
        boa_runtime=args.boa_runtime,
        boa_sif=args.boa_sif,
    ).run()


def cohort_info_entrypoint():
    """Entrypoint to rebuild cohort_info.json from the processed tree without running any stage."""
    logger = create_logger("musiq.cohort_info")

    import argparse

    parser = argparse.ArgumentParser(description="Rebuild cohort_info.json fresh from all patient_info.json files.")
    parser.add_argument(
        "--input-dirpath-processed",
        required=True,
        help="Path to the processed output directory (root containing the per-patient folders).",
    )
    args = parser.parse_args()

    logger.info(f"Rebuilding cohort_info.json under {args.input_dirpath_processed}")
    cohort_info = build_cohort_info(args.input_dirpath_processed)
    logger.info(f"Wrote cohort_info.json with {len(cohort_info)} patient(s).")


if __name__ == "__main__":
    workflow_entrypoint()
