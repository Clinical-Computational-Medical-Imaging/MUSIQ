# import nibabel as nib, numpy as np
# img = nib.load("/home/amka26/Documents/test_data/mp_0001/20221227/PETseg.nii.gz")
# print(img.shape)                       # (x, y, z)
# print(img.header.get_zooms())          # spacing in mm, e.g. (4.07, 4.07, 3.0)
# print(np.prod(img.header.get_zooms()[:3]), "mm³ per voxel")
# print(img.header)                      # the full header, if you want everything

from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("/home/amka26/Documents/test_data")
MASK = "PETseg.nii.gz"
N_PATIENTS = 10

seen = set()
for f in sorted(ROOT.glob(f"*/*/{MASK}")):  # <patient>/<session>/<mask>
    patient = f.parent.parent.name
    if patient in seen:  # one session per patient
        continue
    seen.add(patient)

    img = nib.load(str(f))
    sx, sy, sz = img.header.get_zooms()[:3]
    vox_mm3 = sx * sy * sz
    units = img.header.get_xyzt_units()[0]
    print(
        f"{patient} ({f.parent.name}): spacing {sx:.2f} x {sy:.2f} x {sz:.2f} {units} "
        f"| 1 vox = {vox_mm3:.2f} mm³ | 100 mm³ = {100 / vox_mm3:.1f} vox"
    )

    if len(seen) == N_PATIENTS:
        break