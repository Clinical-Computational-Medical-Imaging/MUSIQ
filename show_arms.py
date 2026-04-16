import json
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import matplotlib.cm as cm
from pathlib import Path
from totalsegmentator.nifti_ext_header import load_multilabel_nifti


class Show:
    def __init__(self, path:str) -> None:
        self.path = Path(path)

    def run(self):
        print("start")
        pat_folder = sorted([f for f in self.path.iterdir() if f.is_dir()])        
        for p in pat_folder:
            if (p / "CT_muscle_fat.nii.gz").is_file():
                with open(p.parent / "patient_info.json") as f:
                    info = json.load(f)
                
                ct_img = nib.load(p / "CT.nii.gz")
                ct_data = ct_img.get_fdata()
            else: 
                continue

            ct_series_list = info["Studies"][p.parent.name]["Modalities"]["CT"]

            serie_dict = ct_series_list[0]              # Liste → Dict
            series_name = next(iter(serie_dict))       # Name extrahieren
            series_data = serie_dict[series_name]
            if series_data["body_composition_analysis"]["full_picture"]["total_fat_in_%"] is None:
                y = ct_data.shape[1] // 2 - 20
                ct_slice = ct_data[:, y , :]

                # Plot
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                plt.imshow(np.rot90(ct_slice), cmap="gray")
                plt.title("CT")
                plt.axis("off")
                plt.show() #savefig(f"{p.parent.parent.name}_all_labels.png", dpi=150)
                plt.close()
        print("finish")


def entrypoint() -> None:
    """Entry point to run the script without full workflow."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Go through output data an check if the arms are detected correctly"
    )
    parser.add_argument(
        "-path", type=str, help="Path to the input folder containing CT.nii.gz files", required=True
    )
    args = parser.parse_args()

    Show(
        path=args.path,
    ).run()


if __name__ == "__main__":
    entrypoint()
