# SUV / SUL computation and decay correction

MUSIQ builds `SUV.nii.gz` during the `series_selection` stage as a scalar rescale of the raw
`PET.nii.gz`, and `SUL.nii.gz` (lean-body-mass–corrected) in the `sul` stage. Both share one
decay reference by construction. This document describes how the conversion factor is derived and,
in particular, how the decay reference is chosen from the DICOM tags.

## The factor

```
SUV(voxel) = PET(voxel) × SUVFactor
SUVFactor  = 1000 × weight / decayed_dose
decayed_dose = total_dose × 0.5 ^ ((ref_time − injection_time) / half_life)
```

Units: the `× 1000` converts kg→g so SUV is dimensionless (g/mL activity concentration normalised
to injected dose per body weight). SUL replaces `weight` with the lean body mass:
`SULFactor = SUVFactor × LBM / weight`.

Inputs (all read from a deterministic, filtered DICOM file — never `os.listdir()[0]`):

| Quantity | DICOM source | Tag |
| --- | --- | --- |
| `total_dose` | `RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose` | (0018,1074) |
| `injection_time` | `…[0].RadiopharmaceuticalStartTime` | (0018,1072) |
| `half_life` | `…[0].RadionuclideHalfLife` | (0018,1075) |
| `weight` | `PatientWeight` (study-level fallback if ≤ 0) | (0010,1030) |
| `ref_time` | resolved from `DecayCorrection` — see below | — |

## Reference-consistent decay correction

A correct SUV requires the numerator (PET voxel values) and the denominator (injected activity) to
be referenced to the **same instant**. The scanner decay-corrects the pixels to a reference declared
in `DecayCorrection` (0054,1102); MUSIQ decays the injected dose to that same reference
(`utils.resolve_pet_decay_reference`):

| `DecayCorrection` | Pixels referenced to | `ref_time` returned | Effect on `decayed_dose` |
| --- | --- | --- | --- |
| `START` (Siemens default) / absent | acquisition start | `SeriesTime` (0008,0031), validated | dose decayed injection → acquisition start |
| `ADMIN` | radiopharmaceutical administration (injection) | `RadiopharmaceuticalStartTime` (0018,1072) | Δt = 0 → dose **not** decayed |
| `NONE` | not decay-corrected | earliest `AcquisitionTime` (best effort) | flagged unreliable — see below |

Why each case matters:

- **`START`** — Whole-body PET is acquired over ~10–20 min, so the per-slice `AcquisitionTime`
  (0008,0032) advances with each bed position. Using an arbitrary per-slice time as the reference
  inflated SUV by up to ~25%. MUSIQ instead uses the acquisition-start `SeriesTime`, and **validates
  it against the earliest per-slice `AcquisitionTime`**: a series cannot start after its first slice
  was acquired, so if a vendor writes a later (e.g. reconstruction) `SeriesTime`, it is discarded in
  favour of the earliest `AcquisitionTime`.
- **`ADMIN`** — the pixels are already referenced to the injection time, so the dose must be taken at
  injection time too (Δt = 0, no further decay). Decaying the dose again — as would happen if any
  acquisition time were used — would **double-correct** and bias SUV downward.
- **`NONE`** — the pixels are not decay-corrected at all. A single scalar factor cannot per-slice
  correct a whole-body scan (each slice was measured at a different time), so a quantitatively valid
  SUV is not achievable this way. MUSIQ emits a **loud warning** and falls back to the earliest
  `AcquisitionTime` as a best-effort reference rather than silently producing a bogus number.

Any unrecognised `DecayCorrection` value falls back to the earliest `AcquisitionTime` with a warning.

## Single source of truth

`series_selection` persists everything the SUV image was built with into `patient_info.json`:
`SUVFactor`, `DecayCorrectionReference` (the raw `DecayCorrection` flag), `RadiopharmaceuticalStartTime`,
`InjectedRadioactivity`, `RadionuclideHalfLife`, the resolved `AcquisitionTime` (= `ref_time`), and the
`PatientWeight` actually used. The `sul` stage **reuses `SUVFactor`** (scaling it by `LBM / weight`)
instead of re-deriving the reference — so SUV and SUL always share one decay reference, and re-runs stay
consistent with the images on disk.

## Notes / limitations

- The factor assumes the standard Siemens-style quantitative calibration. `ADMIN` and `NONE` are handled
  for correctness/safety, but cohorts are overwhelmingly `START` in practice.
- SUL additionally requires `PatientLBM` from the `muscle_fat` stage; without it the `sul` stage skips.
