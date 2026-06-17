import json
import multiprocessing as mp
import os
import pathlib as plb
import platform
import subprocess

from .utils import create_logger, setup_series_keywords

_REPO_ROOT = plb.Path(__file__).parent.parent.parent


class Workflow:
    def __init__(
        self,
        input_dirpath: str,
        output_dirpath: str,
        tasks: list[str] | None = None,
        cads_tasks: list[str] | None = None,
        pet_metric: str | list[str] | None = None,
        ct_primary_keywords: list[str] | None = None,
        ct_secondary_keywords: list[str] | None = None,
        ct_exclusion_keywords: list[str] | None = None,
        pt_primary_keywords: list[str] | None = None,
        pt_secondary_keywords: list[str] | None = None,
        pt_exclusion_keywords: list[str] | None = None,
        mr_primary_keywords: list[str] | None = None,
        mr_secondary_keywords: list[str] | None = None,
        mr_exclusion_keywords: list[str] | None = None,
    ) -> None:
        """
        Run the MUSIQ workflow with the specified parameters.

        Args:
            input_dir (str): Path to the input directory containing PET/CT images.
            output_dir (str): Path to the output directory for results.
            tasks (list[str] | None): List of tasks to run. If None, all tasks are run. Possible values are:
                - "series_selection": Select series based on keywords.
                - "autopet": Run autopet3 on PET images.
                - "totalsegmentator": Run TotalSegmentator on CT images.
                - "moose": Run Moose on CT images.
                - "radiomics": Extract radiomics features from selected series.
                - "tumor": Compute tumor level statistics.
                - "plot": Create visualisations.
                - "cads": Run CADS on CT images.
            ct_primary_keywords (list[str] | None): Keywords for primary selection of CT series.
            ct_secondary_keywords (list[str] | None): Keywords for secondary selection of CT series.
            ct_exclusion_keywords (list[str] | None): Keywords to exclude CT series.
            pt_primary_keywords (list[str] | None): Keywords for primary selection of PT series.
            pt_secondary_keywords (list[str] | None): Keywords for secondary selection of PT series.
            pt_exclusion_keywords (list[str] | None): Keywords to exclude PT series.
            mr_primary_keywords (list[str] | None): Keywords for primary selection of MR series.
            mr_secondary_keywords (list[str] | None): Keywords for secondary selection of MR series.
            mr_exclusion_keywords (list[str] | None): Keywords to exclude MR series.
            pet_metric (str | list[str] | None): PET metric(s) to use for radiomics and tumor computations.
            Possible values are "SUV" and "SUL". Can pass one or both. Defaults to ["SUV", "SUL"].
        """
        self.input_dirpath = input_dirpath
        self.output_dirpath = output_dirpath
        if pet_metric is None:
            pet_metric = ["SUV", "SUL"]
        self.pet_metric = pet_metric
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
                "cads",
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
                        "cads",
                    ]
                    for t in tasks or []
                )
            )
            if tasks_error:
                raise ValueError(
                    "Invalid tasks specified. Possible values are: "
                    "'series_selection', 'radiomics', 'autopet', "
                    "'totalsegmentator', 'tumor', 'plot', 'moose', "
                    "'muscle_fat', 'cads'."
                )

        self.series_selection = "series_selection" in (tasks or [])
        self.autopet = "autopet" in (tasks or [])
        self.cads = "cads" in (tasks or [])
        self.totalsegmentator = "totalsegmentator" in (tasks or [])
        self.muscle_fat = "muscle_fat" in (tasks or [])
        self.moose = "moose" in (tasks or [])
        self.radiomics = "radiomics" in (tasks or [])
        self.tumor = "tumor" in (tasks or [])
        self.plot = "plot" in (tasks or [])

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

    def run(self) -> None:
        # Ensure output directory exists
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
            from .totalsegmentator_muscle_fat_sul import TotalSegmentatorMuscleFatSUL

            logger.info("\n" + "#" * 50 + "\nStarting TotalSegmentator Muscle Fat and SUL computation\n" + "#" * 50)
            TotalSegmentatorMuscleFatSUL(
                input_dirpath_processed=self.output_dirpath,
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
            RadiomicsExtractor(
                input_dirpath_processed=self.output_dirpath,
                pet_metric=self.pet_metric,
            ).run()

        if self.tumor:
            from .tumor_info_extraction import TumorInfoExtraction

            logger.info("\n" + "#" * 50 + "\nStarting Tumor Info Extraction\n" + "#" * 50)
            TumorInfoExtraction(
                input_dirpath_processed=self.output_dirpath,
                pet_metric=self.pet_metric,
            ).run()

        logger.info("\n" + "#" * 50 + "\nStarting Cohort Info Creation\n" + "#" * 50)
        cohort_info = {}
        if os.path.exists(os.path.join(self.output_dirpath, "cohort_info.json")):
            with open(os.path.join(self.output_dirpath, "cohort_info.json")) as f:
                cohort_info = json.load(f)
        for dirpath, _, filenames in os.walk(self.output_dirpath):
            if "patient_info.json" in filenames:
                with open(os.path.join(dirpath, "patient_info.json")) as json_file:
                    patient_info = json.load(json_file)
                    patient_id = patient_info.get("PatientID", "Unknown")
                    cohort_info.update({patient_id: patient_info})

        with open(os.path.join(self.output_dirpath, "cohort_info.json"), "w") as f:
            json.dump(cohort_info, f)

        if self.plot:
            from .plot_summary import PlotSummary

            logger.info("\n" + "#" * 50 + "\nStarting Plot Summary\n" + "#" * 50)
            PlotSummary(
                input_dirpath_processed=self.output_dirpath,
            ).run()

        logger.info("\n" + "#" * 50 + "\nWorkflow completed successfully\n" + "#" * 50)


def workflow_entrypoint():
    """Entrypoint to run the MUSIQ workflow."""
    # Set the start method for multiprocessing to 'spawn' for compatibility
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
        "radiomics, autopet, totalsegmentator, tumor, plot, moose, muscle_fat, cads.",
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
        "--ct-primary-keywords", help="List of keywords to look for in CT study descriptions for default selection."
    )
    parser.add_argument(
        "--ct-secondary-keywords",
        help="List of keywords to look for in CT study descriptions for alternative selection.",
    )
    parser.add_argument("--ct-exclusion-keywords", help="List of keywords to exclude CT studies from selection.")
    parser.add_argument(
        "--pt-primary-keywords", help="List of keywords to look for in PT study descriptions for default selection."
    )
    parser.add_argument(
        "--pt-secondary-keywords",
        help="List of keywords to look for in PT study descriptions for alternative selection.",
    )
    parser.add_argument("--pt-exclusion-keywords", help="List of keywords to exclude PT studies from selection.")
    parser.add_argument(
        "--mr-primary-keywords", help="List of keywords to look for in MR study descriptions for default selection."
    )
    parser.add_argument(
        "--mr-secondary-keywords",
        help="List of keywords to look for in MR study descriptions for alternative selection.",
    )
    parser.add_argument("--mr-exclusion-keywords", help="List of keywords to exclude MR studies from selection.")
    parser.add_argument(
        "--pet-metric",
        type=str,
        nargs="+",
        choices=["SUV", "SUL"],
        default=["SUV", "SUL"],
        help="PET metric(s) to use. Pass one or both: --pet-metric SUV SUL (default: SUV SUL)",
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
    ).run()


if __name__ == "__main__":
    workflow_entrypoint()
