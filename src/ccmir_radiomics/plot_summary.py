import logging
import os

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class PlotSummary:
    def __init__(self, input_dirpath_processed: str | os.PathLike) -> None:
        """
        Class to generate dual-axis plots comparing FDG and PSMA metrics over time.

        Args:
            input_dirpath_processed (str | os.PathLike): Directory containing the extracted patient radiomics.
            Can be nested.
        """
        self.input_dirpath_processed = input_dirpath_processed

    def run(self) -> None:
        # Ensure the root directory exists
        if not os.path.exists(self.input_dirpath_processed):
            raise FileNotFoundError(f"Root directory {self.input_dirpath_processed} does not exist.")

        # Iterate over all subfolders in the root directory
        for folder in sorted(os.listdir(self.input_dirpath_processed)):
            folder_path = os.path.join(self.input_dirpath_processed, folder)
            if not os.path.isdir(folder_path):
                continue
            logger.info(f"Processing folder: {folder_path}")
            try:
                self.process_folder(folder_path)
            except Exception as e:
                logger.info(f"Error processing folder {folder_path}: {e}")
                continue

    def process_folder(self, folder_path: str | os.PathLike) -> None:
        # Embed the CSV data provided by the user
        csv_data = f"{folder_path}/patient_radiomics.csv"
        output_dir = f"{folder_path}/plots"
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(csv_data):
            logger.info(f"CSV file not found in {folder_path}, skipping...")
            return

        # Read into DataFrame
        df = pd.read_csv(csv_data, parse_dates=["Date"])

        # List of metrics to plot
        metrics = [
            "Dmax",
            "LesionCount",
            "SUVmax",
            "SUVmean",
            "SUVpeak",
            "SUVstd",
            "TLG",
            "TMTV",
            "SurfaceArea",
            "MTV2.5",
            "MTV3.0",
            "MTV3.5",
            "MTV4.0",
            "MTV30",
            "MTV40",
            "MTV41",
            "MTV50",
            "SDmax",
        ]

        # Generate a separate dual-axis plot for each metric
        for metric in metrics:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()

            # Select FDG and PSMA series
            fdg = df[
                df["Radiopharmaceutical"].isin(["FDG", "Fluorodeoxyglucose", "FDG -- fluorodeoxyglucose"])
            ].set_index("Date")[metric]
            psma = df[df["Radiopharmaceutical"] == "PSMA"].set_index("Date")[metric]

            # Plot on respective axes
            ax1.plot(fdg.index, fdg.values, marker="o", label="FDG", color="blue")
            ax2.plot(psma.index, psma.values, marker="s", label="PSMA", color="orange")

            # Label axes
            ax1.set_xlabel("Date")
            ax1.set_ylabel(f"{metric} (FDG)")
            ax2.set_ylabel(f"{metric} (PSMA)")

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

            plt.title(f"{metric} over Time: FDG vs PSMA")
            plt.tight_layout()
            fname = os.path.join(output_dir, f"{metric}_dualaxis.png")
            plt.savefig(fname)
            plt.close(fig)


def plot_summary_entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    from .utils import create_logger

    global logger
    logger = create_logger()

    import argparse

    parser = argparse.ArgumentParser(description="TBD")
    parser.add_argument("--input-dirpath-processed", help="Path to root directory.", required=True)
    args = parser.parse_args()

    PlotSummary(
        input_dirpath_processed=args.input_dirpath_processed,
    ).run()


if __name__ == "__main__":
    plot_summary_entrypoint()
