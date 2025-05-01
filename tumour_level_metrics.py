import os
import json
import numpy as np
import SimpleITK as sitk
from skimage.measure import label, regionprops
#from radiomics import featureextractor
from tqdm import tqdm
import nibabel as nib
import nilearn.image
import metrics
import cc3d
from multiprocessing import Pool, cpu_count

# Root folder containing the studies
root_folder = "/home/lukas-foerner/Projects/ALPS_PROSTATE/data/"

def get_spacing_from_niftipath(path):
    """Get voxel spacing from a NIfTI file."""
    return sitk.ReadImage(path).GetSpacing()

def load_nifti_as_array(path):
    """Load a NIfTI file as a numpy array."""
    image = sitk.ReadImage(path)
    array = sitk.GetArrayFromImage(image)
    return array, image

def compute_connected_components(mask):
    """Perform connected component analysis on a binary mask."""
    labeled_mask = label(mask, connectivity=3)
    return labeled_mask

def compute_tumor_organ_overlap(tumor_mask, organ_mask, organ_labels):
    """Determine which organs the tumor overlaps with."""
    overlap = {}
    for organ_id, organ_name in organ_labels.items():
        organ_region = (organ_mask == int(organ_id))
        if np.any(np.logical_and(tumor_mask, organ_region)):
            overlap[organ_name] = True
    return overlap

def compute_pet_metrics(tumor_mask, pet_array, spacing):
    """Compute standard PET metrics for a tumor."""
    tumor_voxels = pet_array[tumor_mask > 0]
    if tumor_voxels.size == 0:
        return {"SUVmean": None, "SUVmax": None, "SUVstd": None}
    return {
        "SUVmean": float(np.mean(tumor_voxels)),
        "SUVmax": float(np.max(tumor_voxels)),
        "SUVstd": float(np.std(tumor_voxels)),
        "SUVpeak": metrics.calculate_suvpeak_median(pet_array, tumor_mask, spacing),
        "SurfaceArea_mm2": metrics.calculate_patient_level_surface_area(tumor_mask, spacing),
    }

#def extract_radiomics_features(tumor_mask, pet_image):
#    """Extract PyRadiomics features for a tumor."""
#    extractor = featureextractor.RadiomicsFeatureExtractor()
#    features = extractor.execute(pet_image, sitk.GetImageFromArray(tumor_mask))
#    return {key: float(value) for key, value in features.items() if isinstance(value, (int, float))}
def resample_ct(nii_out_path):
    # resample CT to PET and mask resolution
    ct   = nib.load(nii_out_path+'/CTseg.nii.gz')
    pet  = nib.load(nii_out_path+'/PET.nii.gz')
    CTres = nilearn.image.resample_to_img(ct, pet, interpolation='nearest', fill_value=0)
    nib.save(CTres, nii_out_path+'/CTsegres.nii.gz')

def process_study(study_path):
    """Process a single study directory."""
    petseg_path = os.path.join(study_path, "PETseg.nii.gz")
    ctseg_path = os.path.join(study_path, "CTsegres.nii.gz")
    suv_path = os.path.join(study_path, "SUV.nii.gz")
    organ_labels_path = os.path.join(study_path, "CTseg.json")

    if not all(os.path.exists(p) for p in [petseg_path, ctseg_path, suv_path, organ_labels_path]):
        print(f"Skipping incomplete study: {study_path}")
        return None

    # Load data
    petseg_array = metrics.get_3darray_from_niftipath(petseg_path)
    ctseg_array = metrics.get_3darray_from_niftipath(ctseg_path)
    suv_array = metrics.get_3darray_from_niftipath(suv_path)
    spacing = get_spacing_from_niftipath(suv_path)
    voxel_volume_cc = np.prod(spacing)/1000
    with open(organ_labels_path, "r") as f:
        organ_labels = json.load(f)["labels"]

    # Perform connected component analysis
    #labeled_tumors = compute_connected_components(petseg_array)
    labeled_tumors, num_lesions = cc3d.connected_components(petseg_array, connectivity=26, return_N=True)
    #print(labeled_tumors)
    # Process each tumor
    results = []
    for i in range(1, num_lesions+1):
        tumor_mask = np.zeros_like(labeled_tumors)
        tumor_mask[labeled_tumors == i] = 1

        # Compute metrics
        pet_metrics = compute_pet_metrics(tumor_mask, suv_array, spacing)
        organ_overlap = compute_tumor_organ_overlap(tumor_mask, ctseg_array, organ_labels)
        #radiomics_features = extract_radiomics_features(tumor_mask, suv_image)
        num_nonzero_voxels = len(np.nonzero(tumor_mask)[0])

        results.append({
            "TumorID": i,
            "Volume_cm3": num_nonzero_voxels * voxel_volume_cc,
            "PETMetrics": pet_metrics,
            "OrganOverlap": organ_overlap,
            #"RadiomicsFeatures": radiomics_features,
        })

    return results

def process_and_save_study(dirpath):
    """Process a study and save results to a JSON file."""
    #if not "CTsegres.nii.gz" in os.listdir(dirpath):
    #    print(f"Resampling CT for: {dirpath}")
    resample_ct(dirpath)
    study_results = process_study(dirpath)
    if study_results:
        output_path = os.path.join(dirpath, "tumor_analysis_results.json")
        with open(output_path, "w") as f:
            json.dump(study_results, f, indent=4)
        print(f"Results saved to {output_path}")
        return dirpath, study_results
    return dirpath, None

def main():

    all_results = {}
    study_dirs = [
        dirpath for dirpath, _, filenames in os.walk(root_folder)
        if "PETseg.nii.gz" in filenames
    ]

    with Pool(cpu_count()) as pool:
        for dirpath, study_results in tqdm(pool.imap_unordered(process_and_save_study, study_dirs), total=len(study_dirs)):
            if study_results:
                all_results[dirpath] = study_results

    # Save aggregated results to a JSON file
    output_path = os.path.join(root_folder, "tumor_analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()