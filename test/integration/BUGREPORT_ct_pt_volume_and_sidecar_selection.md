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

The slice-spacing test above only checks the resulting physical extent, not which file was picked
or why — it would not notice if the ImageType check were skipped. `test_real_eq1_volume_is_chosen_without_any_imagetype_check`
(same file) makes that mechanism itself the assertion against the same real series: it runs the
real `dcm2niix` directly, confirms it emits exactly one `_Eq_1` file and one plain file, confirms
dcm2niix writes a JSON sidecar (with a valid `PRIMARY` ImageType) only for the plain file — never
for `_Eq_1` — and then confirms `_select_ct_volume` still picks the `_Eq_1` file despite having
strictly *less* ImageType information available for it than for the alternative (none at all, vs.
a confirmed PRIMARY). This is real dcm2niix output, not a synthetic directory listing, and passes
today — it will only start failing once a fix makes the ImageType check actually run ahead of the
Eq_1 preference.

Re-checked directly against `run_dcm2niix` (not just this one downstream test) for all four
CT/PT series currently in `test/integration/conftest.py`'s `_DOWNLOAD_TARGETS`
(`TCGA-VP-A878`'s two CT reconstructions, `TCGA-VP-A879`'s PET WB, and the ACRIN series above):
only the ACRIN series produces more than one `.nii.gz` candidate at all; the other three each
produce exactly one PRIMARY volume, so none of the three code paths in this report has anything
to choose between for them.

For the ACRIN series specifically: dcm2niix writes exactly **one** `.json` sidecar total (for the
plain `5.4_C_A_P_4.nii.gz`), and none for `..._Eq_1.nii.gz`. This is not evidence of the
sidecar-mismatch risk (the second code path above) — dcm2niix only ever emits one sidecar per
*underlying acquisition*, regardless of how many resampled `.nii.gz` variants it derives from it,
and `_Eq_1` is a resample of that same acquisition, not a distinct one. With only one `.json` in
existence, `next(tmp.glob("*json"))` cannot land on a wrong candidate — there is no other
candidate for it to confuse this one with. Demonstrating the sidecar-mismatch risk for real needs
a series where dcm2niix emits **multiple genuinely distinct** sidecars (e.g. an actual bundled
ORIGINAL/PRIMARY + ORIGINAL/SECONDARY pair), which none of the currently downloaded series exhibit
— confirmed instead via the synthetic reproduction below.

## Verified with a synthetic reproduction

Since no currently available real series exercises the sidecar-mismatch or PT-no-ranking paths,
each was additionally reproduced by fabricating the exact post-dcm2niix directory layout this
report hypothesizes and calling the real, unmodified `_select_ct_volume`/`convert_dcm2nii_CT`/
`convert_dcm2nii_PET` against it (dcm2niix itself is bypassed here — only the selection/sidecar
logic under test runs for real):

- **Eq_1 bypass**: a `primary.nii.gz` (`ORIGINAL/PRIMARY`, 8 slices) plus a
  `secondary_Eq_1.nii.gz` (`ORIGINAL/SECONDARY`, 4 slices) in the same tmp dir →
  `_select_ct_volume` returns `secondary_Eq_1.nii.gz`, confirming the ImageType ranking is never
  reached whenever *any* Eq_1 candidate exists, even a worse one.
- **CT sidecar mismatch**: a `chosen_primary.nii.gz` (`ORIGINAL/PRIMARY`, 8 slices, the volume
  `_rank()` would legitimately pick) with **no** sidecar of its own, alongside an
  `other_secondary.nii.gz` + `other_secondary.json` (`ORIGINAL/SECONDARY`, distinct
  `SeriesDescription`) → `convert_dcm2nii_CT` copies the correct 8-slice PRIMARY image to
  `CT.nii.gz` but returns `other_secondary.json`'s tags, i.e. a different series' metadata
  attached to the right image.
- **PT no-ranking**: an `aaa_mip.nii.gz` (2 slices, no sidecar) and a `zzz_wb.nii.gz` (20 slices,
  `SeriesDescription: "WHOLE BODY PET"`) in the same tmp dir → `convert_dcm2nii_PET` copied the
  2-slice MIP to `PET.nii.gz` while returning the whole-body series' tags, on this filesystem's
  glob order (Python does not guarantee `Path.glob()` order, so which file wins is itself
  filesystem-dependent — part of the problem, not a controlled choice made by the code).

These are adversarial, hand-constructed inputs, not observed real-world DICOM — they establish
that the code *does* misbehave exactly as hypothesized once its precondition is met, without
claiming that precondition occurs in practice for any specific scanner/protocol.

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
