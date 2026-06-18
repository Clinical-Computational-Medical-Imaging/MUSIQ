# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MUSIQ (Multimodal Unified Segmentation & Image Quantification) is an end-to-end pipeline for processing PET/CT and MRI scans pulled from a PACS. It converts DICOM → NIfTI, runs several segmentation models, and extracts radiomics / tumor quantification metrics. Python 3.12+, packaged with setuptools (`src/` layout, package = `musiq`).

## Commands

```bash
# Lint / format (config in pyproject.toml: line-length 120, target py312)
ruff check src/
ruff format src/

# Pre-commit (note: metrics.py is excluded from both ruff hooks)
pre-commit install
pre-commit run --all-files

# Run the full pipeline (installed as the `musiq` console script)
musiq --input-dirpath /data/raw --output-dirpath /data/processed \
  --tasks series_selection radiomics autopet totalsegmentator muscle_fat tumor moose cads --cads-tasks 556 558
```

Each task is also exposed as its own console script (see `[project.scripts]` in `pyproject.toml`, e.g. `musiq_cads_inference`, `musiq_autopet_inference`) so individual stages can be run in isolation.

**Tests:** `pyproject.toml` configures pytest (`testpaths = ["tests"]`, coverage on, `filterwarnings = ["error"]`), but there is currently **no `tests/` directory** — there are no tests to run yet.

## Environment setup (critical for running anything)

The pipeline depends on two external repos/checkpoints cloned into the repo root at runtime — they are **not** vendored and not in git:
- `autopet-3-submission/` — cloned from `mic-dkfz/autopet-3-submission`, installed editable via `requirements.txt`.
- `autopet-3-model/` — AutoPET3 nnU-Net checkpoint unzipped from Zenodo. `workflow.py` hardcodes the path `autopet-3-model/Dataset222_AutoPETIII_2024/autoPET3_Trainer__nnUNetResEncUNetLPlansMultiTalent__3d_fullres_bs3`.
- `CADS/` — cloned from `murong-xu/CADS`; `cads_inference.py` does `sys.path.insert` on `../../CADS` and imports `cads.utils.*`. Model weights are downloaded on first inference.

Two virtual environments are required because Moose has conflicting dependencies:
- `.venv` — main environment (`pip install -r requirements.txt`).
- `.venv_moose` — Moose only (`pip install -r requirements_moose.txt` + `moosez --no-deps`).

`workflow.py` runs Moose by shelling out to `.venv_moose`'s Python interpreter as a subprocess (see `run()`), so the Moose stage only works if `.venv_moose` exists at the repo root. The TotalSegmentator muscle/fat stage needs a TS license set via `totalseg_set_license`.

See `README.md` for the exact clone/curl/unzip sequence and the Docker path.

## Architecture

The pipeline is a sequence of independent **task stages**, orchestrated by `Workflow` in `src/musiq/workflow.py`. Each stage is selected by name via `--tasks`; valid names are
`series_selection, radiomics, autopet, totalsegmentator, muscle_fat, tumor, plot, moose, cads`.
The default order when `--tasks` is omitted is fixed in `Workflow.__init__`.

Every stage follows the same module pattern (`<stage>_inference.py` / `<stage>_extraction.py`):
- A class (e.g. `AutopetInference`, `CadsInference`, `RadiomicsExtractor`) constructed with `input_dirpath_processed` and stage-specific args, with a `run()` method.
- A `*_entrypoint()` function that builds a logger via `utils.create_logger`, parses argparse args, and instantiates the class — this is the console-script entry point.
- `run()` walks the processed output tree, finds the relevant NIfTI files, does the work, and **writes results back into `patient_info.json`** (it imports the class lazily inside `Workflow.run()` to keep heavy deps out of the import path of unused stages).

### The data model is the filesystem + `patient_info.json`

There is no database. State flows through files on disk:
- **Input tree:** `raw/<patient_id>/<study_id>/<series_id>/` of DICOMs.
- **Output tree:** `processed/<patient_id>/<study_date>/` containing the NIfTI artifacts (`CT.nii.gz`, `PET.nii.gz`, `SUV.nii.gz`, `SUL.nii.gz`, segmentation masks like `CTseg.nii.gz`, `CTcads.nii.gz`, `PETseg.nii.gz`, etc.).
- **`patient_info.json`** lives at `processed/<patient_id>/` and is the shared accumulator. Each stage reads it, adds its outputs/metrics, and writes it back. The nested structure is `patient_info["Studies"][study_date]["Modalities"][<CT|PT|MR>][series_index][series_name] = {...}` — stages locate their slot with `next(iter(...))` on that path, so they assume a single series per modality per study.
- **`cohort_info.json`** at the `processed/` root is rebuilt near the end of `Workflow.run()` by walking the tree and merging every `patient_info.json`, keyed by `PatientID`.

Because stages key off file existence, they **skip work when the output file already exists** (idempotent re-runs). Stages that need an upstream artifact (e.g. AutoPET needs `SUL.nii.gz`, which is produced by the `muscle_fat` stage) will warn/skip if it is missing — task ordering matters.

### PET metrics: SUV vs SUL

Most quantification runs for one or both of `SUV` and `SUL` (`--pet-metric`, default both). The metric name drives file names and JSON keys (e.g. `PETseg.nii.gz`/`PETsegPath` for SUV vs `PETsegSUL.nii.gz`/`PETsegSULPath` for SUL). `SUL.nii.gz` (lean-body-mass-corrected SUV) is generated by the `muscle_fat` stage (`totalsegmentator_muscle_fat_sul.py`).

### Shared helpers

- `utils.py` — DICOM→NIfTI conversion (`run_dcm2niix`, `run_dicom2nifti`), SUV factor computation, resampling (`resample_image`), connected components, PET metric computation, `make_json_safe` (DICOM/NumPy → JSON), keyword setup, and `create_logger` (logs to `./logger/` and stdout).
- `metrics.py` — radiomics math (TMTV with multiple thresholds, TLG, Dmax/SDmax, SUVpeak, surface area). **Excluded from ruff** — do not expect it to follow the lint rules and avoid reformatting it.
- `config.yaml` — default `SERIES_KEYWORDS` (PRIMARY/SECONDARY/EXCLUSION per CT/PT/MR modality) used by `series_selection` to auto-pick which DICOM series to convert when no keyword CLI args are passed.
