import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Configuration
root_dir = './data/'  # Root directory containing all subfolders

# 2. Iterate over all subfolders in the root directory
for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    print(f"Processing folder: {folder_path}")
    try:
        # Embed the CSV data provided by the user
        csv_data = f"{folder_path}/combined_metrics.csv"
        output_dir = f"{folder_path}/plots"

        if not os.path.exists(csv_data):
            print(f"CSV file not found in {folder_path}, skipping...")
            continue

        # Read into DataFrame
        df = pd.read_csv(csv_data, parse_dates=['Date'])

        # List of metrics to plot
        metrics = ['Dmax', 'LesionCount', 'SUVmax', 'SUVmean', 'SUVpeak', 'SUVstd', 'TLG', 'TMTV', 'SurfaceArea', 'MTV2.5', 'MTV3.0', 'MTV3.5', 'MTV4.0', 'MTV30', 'MTV40', 'MTV41', 'MTV50', 'SDmax']

        # Generate a separate dual-axis plot for each metric
        for metric in metrics:
            fig, ax1 = plt.subplots(figsize=(8, 5))
            ax2 = ax1.twinx()

            # Select FDG and PSMA series
            fdg = df[df['Radiopharmaceutical'].isin(['FDG', 'Fluorodeoxyglucose', 'FDG -- fluorodeoxyglucose'])].set_index('Date')[metric]
            psma = df[df['Radiopharmaceutical'] == 'PSMA'].set_index('Date')[metric]

            # Plot on respective axes
            ax1.plot(fdg.index, fdg.values, marker='o', label='FDG', color='blue')
            ax2.plot(psma.index, psma.values, marker='s', label='PSMA', color='orange')

            # Label axes
            ax1.set_xlabel('Date')
            ax1.set_ylabel(f'{metric} (FDG)')
            ax2.set_ylabel(f'{metric} (PSMA)')

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

            plt.title(f'{metric} over Time: FDG vs PSMA')
            plt.tight_layout()
            fname = os.path.join(output_dir, f'{metric}_dualaxis.png')
            plt.savefig(fname)
            plt.close(fig)
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        continue
