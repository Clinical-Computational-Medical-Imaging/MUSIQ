# Bug: `find_mr_niftis` confuses different MR series that share a ProtocolName

## Summary

`find_mr_niftis` (`src/musiq/utils.py`) is used by `convert_dcm2nii_MR` to detect whether a
series has already been converted, so re-runs can skip it. It matches candidate NIfTI files by
`ProtocolName` (falling back to `SeriesDescription`) followed by **any digit**, instead of the
current series' exact `SeriesNumber`. When multiple series in one study share the same
`ProtocolName` — which several real scanners set at the exam level, not per series — converting
a second, unrelated series into the same output directory is wrongly treated as "already
converted," and returns the **first** series' NIfTI path while reporting the **second**
series' DICOM tags. The image data and its metadata end up mismatched in `patient_info.json`.

## Reproduction

Real data, TCGA-EJ-5495 (`test/integration/manifest-1773751814915.tcia`, TCIA):

- T1 AXIAL — SeriesNumber 5, ProtocolName `"female Pelvis/"`
- T2 AXIAL — SeriesNumber 4, ProtocolName `"female Pelvis/"` (same as above)

```python
t1_path, t1_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=t1_axial_dir, output_dirpath=out_dir)
t2_path, t2_tags = collector.convert_dcm2nii_MR(MR_dcm_dirpath=t2_axial_dir, output_dirpath=out_dir)
```

Result:

```
t1_path == ".../female_Pelvis_5.nii.gz"
t2_path == ".../female_Pelvis_5.nii.gz"   # same file as t1_path
t1_tags["SeriesDescription"] == "T1 AXIAL"
t2_tags["SeriesDescription"] == "T2 AXIAL"
```

`t2_path` should point to a distinct `female_Pelvis_4.nii.gz`; instead it's identical to
`t1_path`, while `t2_tags` still carries T2 AXIAL's own (freshly re-extracted) metadata. A
`patient_info.json` entry for T2 AXIAL would end up pointing at T1 AXIAL's image data.

Automated regression test (currently `xfail(strict=True)`, so it flips to a hard failure once
fixed and needs the marker removed):
`test/integration/series_selection/test_mr_conversion.py::test_mr_series_sharing_a_protocol_name_are_not_confused_with_each_other`

## Root cause

```python
# src/musiq/utils.py, find_mr_niftis()
stem = normalize_dcm2niix_name(protocol_name) or normalize_dcm2niix_name(series_description)
...
re.match(rf"^{re.escape(stem)}_\d", normalize_dcm2niix_name(f.name.removesuffix(".nii.gz")))
```

dcm2niix names output files `%p_%s` (ProtocolName_SeriesNumber). The regex only checks that the
filename starts with `stem_` followed by *some* digit — not the current series' actual
`SeriesNumber`. If `ProtocolName` is identical across series (common — it's often set once for
the whole exam), any prior series' output satisfies the match.

## Impact

- Silent: no exception, no warning logged. `convert_dcm2nii_MR` returns normally with tags that
  look correct (`SeriesDescription` etc. are freshly re-extracted from the actual series' own
  DICOM), masking that the returned *path* belongs to a different series.
- Affects any scanner/protocol where `ProtocolName` is not set per-series — observed on GE MR
  data in this codebase's own test cohort; likely broader than this one dataset.
- Downstream: `patient_info.json` would record one series' metadata against another series'
  image content for every additional series sharing that `ProtocolName` within a study.

## Suggested fix

Require the exact `SeriesNumber` in the match, not just a trailing digit, e.g. anchor the regex
to `f"^{re.escape(stem)}_{series_number}(?!\\d)"` and thread the series' actual `SeriesNumber`
into `find_mr_niftis`/`mr_nifti_exists` (currently not a parameter — would need to be added at
both call sites in `convert_dcm2nii_MR`).
