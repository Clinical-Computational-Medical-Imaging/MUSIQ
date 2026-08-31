# CT/PT: volume selection and sidecar lookup do not consistently check PRIMARY/SECONDARY

## Summary

When dcm2niix emits more than one candidate NIfTI for a series, `series_selection.py` picks one
and reads its DICOM-tag sidecar. For CT, the gantry-tilt-corrected `*_Eq_1` preference in
`_select_ct_volume` is checked *before* the PRIMARY/SECONDARY `ImageType` ranking and bypasses it
entirely — an Eq_1 file belonging to a SECONDARY reconstruction (e.g. a localizer) would be
chosen over an un-tilted PRIMARY volume, with no ImageType check at all. Separately, once a
volume is chosen, both `convert_dcm2nii_CT` and `convert_dcm2nii_MR` fall back to "any json in
the tmp dir" when the exact-stem sidecar is missing — if several candidates' sidecars are
present, the fallback can pick metadata belonging to a different candidate than the one chosen.
`convert_dcm2nii_PET` is more exposed than either: it has no ranking at all (`next(tmp.glob("*
nii.gz"))`, first by glob order) and never even attempts an exact-stem sidecar match before
falling back to "any json".

## Root cause

`src/musiq/series_selection.py`, `_select_ct_volume`:

```python
eq = [f for f in nii_files if f.name.endswith("_Eq_1.nii.gz")]
if eq:
    return eq[0]          # returns immediately; _rank() (the only ImageType check) never runs
...
def _rank(f):
    ...
    primary = "PRIMARY" in image_type and "SECONDARY" not in image_type
    ...
```

`convert_dcm2nii_CT` / `convert_dcm2nii_MR`, sidecar lookup:

```python
jsn = nii.with_suffix("").with_suffix(".json")
if not jsn.is_file():
    jsn = next(tmp.glob("*json"))   # arbitrary pick among any remaining candidates
```

`convert_dcm2nii_PET`:

```python
nii = next(tmp.glob("*nii.gz"))     # no ranking at all
...
sidecar = next(tmp.glob("*json"))   # never tries the matching stem first
```

## Verified against real data

Confirmed with a real, publicly downloadable series (`ACRIN-NSCLC-FDG-PET-114`, TCIA — see
`test/integration/manifest-acrin-nsclc-fdg-pet.tcia`): a series with one irregular interslice
gap makes dcm2niix emit both a plain and an `_Eq_1` output; `_select_ct_volume` picks the Eq_1
one before any ImageType check exists to run
(`test_real_irregular_slice_spacing_is_repaired_to_the_dicom_derived_value` in
`test/integration/series_selection/test_ct_conversion.py`). In that case the Eq_1 file happens to be the correct
choice, so this specific series does not demonstrate a wrong *outcome* — but it demonstrates that
the ImageType check is skipped, not merely deprioritized, whenever an Eq_1 candidate exists. No
real series exhibiting the sidecar-mismatch or the PT no-ranking issue has been found yet; both
remain a structural risk rather than an observed wrong result.

## Impact

- CT: if a SECONDARY reconstruction (e.g. a localizer) happens to be the one dcm2niix
  gantry-corrects, `_select_ct_volume` would choose it over an untilted PRIMARY volume.
- CT/MR: a chosen volume's sidecar-lookup fallback can attach a different candidate's DICOM tags
  (`SeriesDescription`, `ImageType`, ...) to the correct image file.
- PT: both the image and its tags are picked without any correctness check; a bundled
  NAC/CTAC pair or a bundled MIP could result in either the wrong image, the wrong tags, or both.

## Suggested fix

- Scope the Eq_1 preference to within the winning `ImageType` group instead of short-circuiting
  ahead of it: rank by `(primary, has_eq1, ...)` rather than checking Eq_1 first.
- Try the exact-stem sidecar match before any fallback in `convert_dcm2nii_PET`, matching the
  CT/MR pattern.
- Log a warning when the "any json" fallback fires, in all three conversion functions, so an
  actual mismatch (should one ever occur) leaves a trace instead of failing silently.
- Give `convert_dcm2nii_PET` a ranking function analogous to CT/MR's, rather than an unconditional
  `next(...)`.
