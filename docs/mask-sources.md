# Mask sources: automated vs. revised labels

The `radiomics` and `tumor` stages compute on one or both mask sources, chosen with `--mask-source`:

| `--mask-source` | Mask used | JSON keys | Metrics |
| --- | --- | --- | --- |
| `auto` (default) | automated `PETseg.nii.gz` / `PETsegSUL.nii.gz` | `TumorStats` / `TumorStatsSUL` | per `--pet-metric` (SUV and/or SUL) |
| `revised` | physician label (see `--label-dirpath` / `--label-glob` below) | `TumorStatsRevised` | SUV only (the manual label is drawn once, independent of SUV/SUL) |

Pass both to compute everything in a single call — the sources run **sequentially** so their distinct keys never collide, and the automated `TumorStats*` are left untouched by the revised pass:

```bash
musiq --input-dirpath /data/raw --output-dirpath /data/processed \
  --tasks radiomics tumor \
  --mask-source auto revised --pet-metric SUV SUL \
  --label-dirpath /path/to/labels --label-glob '*segmentation_Tumor.nii' \
  --radiomics-workers 30
```

The revised label's **filename and location vary by cohort**, so both are configurable:
- `--label-glob` — filename pattern (wildcards allowed). Default `PETseg_revised.nii`
- `--label-dirpath` — **omit** to look for the label *inside each study dir*; **set** it to look under `<label-dirpath>/<PatientID>/`. Only used for `--mask-source revised`.

## Notes
- The label must share the SUV/PET grid; studies whose label grid differs (e.g. the physician segmented a different reconstruction) are **skipped** and listed in `tumor_seg_radiomics_skipped.csv` / `tumor_seg_tumorinfo_skipped.csv` at the processed root.
- `--radiomics-workers N` parallelises the `radiomics` and `tumor` stages across patients (`N` worker processes; default `1` = serial). Parallelism is patient-level, so each `patient_info.json` is only ever written by one worker. The standalone `musiq_extraction` (radiomics) / `musiq_tumor_info_extraction` scripts take the same `--mask-source`, `--label-dirpath` and `--workers` flags.
