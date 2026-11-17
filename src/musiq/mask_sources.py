"""Mask sources for the radiomics/tumor stages and the file names they resolve to.

Kept dependency-free so the workflow can validate ``--mask-source`` without importing the heavy
extraction modules.
"""

# "autopet" = pipeline PETseg/PETsegSUL, "revised" = physician tumor label (must share the SUV/PET grid),
# "lion" = PETseg_LION, "postprocessed" / "lion_postprocessed" = the autopet_postprocessing outputs.
MASK_SOURCES = ("autopet", "revised", "lion", "postprocessed", "lion_postprocessed")

# Sources with a mask for both SUV and SUL; all other sources run on SUV only.
SUL_CAPABLE_SOURCES = ("autopet", "postprocessed")

POSTPROCESSED_SUFFIX = "_postprocessed"  # cleaned mask: <mask><suffix>.nii.gz
POSTPROCESSED_REMOVED_SUFFIX = "_postprocessedRemoved"  # only the removed lesions: <mask><suffix>.nii.gz
