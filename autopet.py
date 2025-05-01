import subprocess
import tempfile
import shutil
import nibabel as nib
import nilearn.image
import argparse
import os
import pathlib as plb

## Download checkpoint from here first: https://github.com/mic-dkfz/autopet-3-submission

def resample_ct(nii_out_path):
    # resample CT to PET and mask resolution
    ct   = nib.load(nii_out_path+'/CT.nii.gz')
    pet  = nib.load(nii_out_path+'/PET.nii.gz')
    CTres = nilearn.image.resample_to_img(ct, pet, fill_value=-1024)
    nib.save(CTres, nii_out_path+'/CTres.nii.gz')


def process_folder(root_folder, path_to_checkpoint):
    i = 0
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "SUV.nii.gz":
                i+=1
                print(f"Processing #{i}")
                if os.path.isfile(dirpath+'/PETseg.nii.gz') or not os.path.isfile(dirpath+'/CT.nii.gz'):
                    continue
                ct_res = os.path.join(dirpath, "CTres.nii.gz")
                if not os.path.isfile(ct_res):
                    print(f"Resamling: {dirpath}")
                    resample_ct(dirpath)
                    
                with tempfile.TemporaryDirectory() as tmp:
                    shutil.copy(dirpath+'/CTres.nii.gz', tmp+'/ALPS_0000.nii.gz')
                    shutil.copy(dirpath+'/SUV.nii.gz', tmp+'/ALPS_0001.nii.gz')
                    try:
                        # Set GPU visibility
                        #os.environ["CUDA_VISIBLE_DEVICES"] = str("cuda:0")

                        with tempfile.TemporaryDirectory() as output_folder:
                            output_folder = plb.Path(str(output_folder))
                            os.environ["nnUNet_raw"] = ""
                            os.environ["nnUNet_preprocessed"] = ""
                            os.environ["nnUNet_results"] = ""

                            command = [
                                "nnUNetv2_predict_from_modelfolder",
                                "-i", tmp,
                                "-o", str(output_folder),
                                "-m", path_to_checkpoint
                            ]

                            print("Running nnUNet prediction...")
                            subprocess.run(command, check=True)
                            print(f"Inference completed. Results saved to {output_folder}.")

                            nii = next(output_folder.glob('*nii.gz'))
                            shutil.copy(nii, dirpath+'/PETseg.nii.gz')
                    
                    except subprocess.CalledProcessError as e:
                        print(f"Error during nnUNet prediction: {e}")
                    except Exception as e:
                        print(f"Unexpected error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Recursively run AutoPET on all SUV.nii.gz files in a folder. "
    )
    parser.add_argument("folder", type=str, help="Path to the input folder containing SUV.nii.gz files")
    parser.add_argument("nnunet_checkpoint", type=str, help="Path to the nnunet checkpoint folder")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a valid directory.")
        return

    process_folder(args.folder, args.nnunet_checkpoint)

if __name__ == "__main__":
    main()