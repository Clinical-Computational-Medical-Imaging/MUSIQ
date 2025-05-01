import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Configuration
root_dir = '/home/lukas-foerner/Data/EANM_IMM_NIFTI'  # Root directory containing all subfolders
combined_csv_path = './combined_data_melanoma.csv'  # Output path for the combined CSV

# 2. Initialize an empty DataFrame to store combined data
combined_df = pd.DataFrame()

# 3. Iterate over all subfolders in the root directory
for folder in sorted(os.listdir(root_dir)):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    print(f"Processing folder: {folder_path}")
    try:
        # Extract patient ID from the folder name
        patient_id = os.path.basename(folder_path)

        # Embed the CSV data provided by the user
        csv_data = f"{folder_path}/combined_metrics.csv"

        if not os.path.exists(csv_data):
            print(f"CSV file not found in {folder_path}, skipping...")
            continue

        # Read into DataFrame
        df = pd.read_csv(csv_data)

        # Normalize Radiopharmaceutical column
        df['Radiopharmaceutical'] = df['Radiopharmaceutical'].replace(
            ['FDG', 'Fluorodeoxyglucose', 'FDG -- fluorodeoxyglucose'], 'FDG'
        )

        # Add Patient ID column
        df['PatientID'] = patient_id

        # Append to the combined DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    except Exception as e:
        print(f"Error processing folder {folder_path}: {e}")
        continue

# 4. Save the combined DataFrame to a CSV file
combined_df.to_csv(combined_csv_path, index=False)
print(f"Combined CSV saved to {combined_csv_path}")
