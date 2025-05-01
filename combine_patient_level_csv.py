import os
import pandas as pd
import matplotlib.pyplot as plt
import json

# 1. Configuration
root_dir = '/home/lukas-foerner/Data/EANM_IMM_NIFTI'  # Root directory containing all subfolders

# 2. Iterate over all subfolders in the root directory
for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    output_csv = os.path.join(folder_path, 'combined_metrics.csv')  # Save CSV in the respective folder
    #output_dir = os.path.join(folder_path, 'plots')                # Where to save plots (if needed)
    #os.makedirs(output_dir, exist_ok=True)

    # Collect rows across all dates and radiopharma
    rows = []

    for sub in sorted(os.listdir(folder_path)):
        subpath = os.path.join(folder_path, sub)
        if not os.path.isdir(subpath) or not sub.isdigit():
            continue

        # --- read CSV ---
        csv_path = os.path.join(subpath, 'patient_level_stats.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)  # expects columns Metric,Value
        df['Date'] = pd.to_datetime(sub, format='%Y%m%d')
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce')  # Convert Value column to numeric, coercing errors to NaN

        # --- read PET.json ---
        json_path = os.path.join(subpath, 'PET.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                info = json.load(f)
            rad = info.get('Radiopharmaceutical', 'Unknown')
        else:
            rad = 'Unknown'

        df['Radiopharmaceutical'] = rad
        rows.append(df)

    if not rows:
        print(f"No data found in folder {folder_path}, skipping...")
        continue

    # Combine into one DataFrame
    combined = pd.concat(rows, ignore_index=True)
    print(combined)
    # Pivot so each row is Date+Radioname and columns are metrics
    pivoted = combined.pivot_table(
        index=['Date', 'Radiopharmaceutical'],
        columns='Metric',
        values='Value'
    ).sort_index()

    # Save the combined table
    pivoted.to_csv(output_csv)
    print(f'Combined table saved to {output_csv}')
