import json
import logging
import multiprocessing as mp
import os

from .plot_summary import PlotSummary
from .radiomics_extraction import RadiomicsExtractor
from .series_selection import SeriesSelection
from .tumor_info_extraction import TumorInfoExtraction
from .utils import setup_series_keywords

logger = logging.getLogger(__name__)


class Workflow:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        task: list[str] | None = None,
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
        Run the ccmir-radiomics workflow with the specified parameters.

        Args:
            input_dir (str): Path to the input directory containing PET/CT images.
            output_dir (str): Path to the output directory for results.
            task (list[str] | None): List of tasks to run. If None, all tasks are run. Possible values are:
                - "series_selection": Select series based on keywords.
                - "radiomics": Extract radiomics features from selected series.
                - "autopet": Run autopet3 on PET images.
                - "totalsegmentator": Run TotalSegmentator on CT images.
                - "tumor": Compute tumor level statistics.
                - "plot": Create visualisations.
            ct_primary_keywords (list[str] | None): Keywords for primary selection of CT series.
            ct_secondary_keywords (list[str] | None): Keywords for secondary selection of CT series.
            ct_exclusion_keywords (list[str] | None): Keywords to exclude CT series.
            pt_primary_keywords (list[str] | None): Keywords for primary selection of PT series.
            pt_secondary_keywords (list[str] | None): Keywords for secondary selection of PT series.
            pt_exclusion_keywords (list[str] | None): Keywords to exclude PT series.
            mr_primary_keywords (list[str] | None): Keywords for primary selection of MR series.
            mr_secondary_keywords (list[str] | None): Keywords for secondary selection of MR series.
            mr_exclusion_keywords (list[str] | None): Keywords to exclude MR series.
            plot (bool): Whether to create visualisations.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        task_error = True if any(t not in [
            "series_selection", "radiomics", "autopet", "totalsegmentator", "tumor", "plot"
        ] for t in task or []) else False
        if task_error:
            raise ValueError(
                "Invalid task specified. Possible values are: "
                "'series_selection', 'radiomics', 'autopet', "
                "'totalsegmentator', 'tumor', 'plot'."
            )
        self.series_selection = "series_selection" in (task or [])
        self.radiomics = "radiomics" in (task or [])
        self.autopet = "autopet" in (task or [])
        self.totalsegmentator = "totalsegmentator" in (task or [])
        self.tumor = "tumor" in (task or [])
        self.plot = "plot" in (task or [])
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
        os.makedirs(self.output_dir, exist_ok=True)

        if self.series_selection:
            from .series_selection import SeriesSelection

            SeriesSelection(
                input_dirpath=self.input_dir,
                output_dirpath=self.output_dir,
                series_keywords=self.series_keywords,
            ).run()

        if self.autopet:
            from .autopet_inference import AutopetInference

            AutopetInference(
                input_dirpath_processed=self.output_dir,
                autopet_checkpoint_dirpath=os.path.join(
                    "./autopet-3-model/Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3"
                ),
            ).run()

        if self.totalsegmentator:
            from .totalsegmentator_inference import TotalSegmentatorInference

            TotalSegmentatorInference(
                input_dirpath_processed=self.output_dir,
            ).run()

        if self.radiomics:
            from .radiomics_extraction import RadiomicsExtractor

            RadiomicsExtractor(
                input_dirpath_processed=self.output_dir,
            ).run()

        if self.tumor:
            TumorInfoExtraction(
                input_dirpath_processed=self.output_dir,
            ).run()

        if self.plot:
            PlotSummary(
                input_dirpath_processed=self.output_dir,
            ).run()

        # Save patient info JSON with all relevant metadata.
        cohort_info = {}
        if os.path.exists(os.path.join(self.output_dir, "cohort_info.json")):
            with open(os.path.join(self.output_dir, "cohort_info.json")) as f:
                cohort_info = json.load(f)
        for dirpath, _, filenames in os.walk(self.output_dir):
            if "patient_info.json" in filenames:
                with open(os.path.join(dirpath, "patient_info.json")) as json_file:
                    patient_info = json.load(json_file)
                    patient_id = patient_info.get("PatientID", "Unknown")
                    cohort_info.update({patient_id: patient_info})

        with open(os.path.join(self.output_dir, "cohort_info.json"), "w") as f:
            json.dump(cohort_info, f)
        logger.info("Workflow completed successfully.")


def workflow_entrypoint():
    """Entrypoint to run the ccmir-radiomics workflow."""
    # Set the start method for multiprocessing to 'spawn' for compatibility
    mp.set_start_method("spawn", force=True)

    from .utils import create_logger

    global logger
    logger = create_logger()

    import argparse

    parser = argparse.ArgumentParser(description="Arguments for ccmir-radiomics pipeline.")
    parser.add_argument("--input-dir", help="Path to PET/CT input directory.", required=True)
    parser.add_argument("--output-dir", help="Path to designated output directory.", required=True)
    parser.add_argument("--task", nargs="+", help="List of tasks to run. Possible values: series_selection, " \
    "radiomics, autopet, totalsegmentator, tumor, plot.", default=None)
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
    args = parser.parse_args()

    if not args.input_dir or not args.output_dir:
        logger.error("Input and output directories must be specified.")
    logger.info(f"Starting ccmir-radiomics pipeline with arguments: {args}")

    Workflow(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        task=args.task,
        ct_primary_keywords=args.ct_primary_keywords,
        ct_secondary_keywords=args.ct_secondary_keywords,
        ct_exclusion_keywords=args.ct_exclusion_keywords,
        pt_primary_keywords=args.pt_primary_keywords,
        pt_secondary_keywords=args.pt_secondary_keywords,
        pt_exclusion_keywords=args.pt_exclusion_keywords,
        mr_primary_keywords=args.mr_primary_keywords,
        mr_secondary_keywords=args.mr_secondary_keywords,
        mr_exclusion_keywords=args.mr_exclusion_keywords,
    ).run()


if __name__ == "__main__":
    workflow_entrypoint()
