# Tests

Two independent suites, selected by directory:

```
test/
├── conftest.py                      # stubs heavy compiled deps (SimpleITK, gdcm) before import
├── unit/                            # fast, self-contained, always run
│   ├── conftest.py                  # dcm2niix_available guard + dicom_series_factory fixture
│   ├── _dicom_builder.py            # synthetic-DICOM-series writer shared by all unit tests
│   ├── series_selection/            # SeriesSelection: collection, conversion, volume/sidecar selection, ...
│   └── utils/                       # utils.py helpers (SUV factor, affine repair, DICOM tag extraction, ...)
└── integration/                     # slow, real dcm2niix + real DICOM data, opt-in
    ├── conftest.py                  # TCIA download/manifest fixtures (see below)
    ├── manifest-*.tcia              # NBIA Data Retriever manifests for the manual-download path
    └── series_selection/            # end-to-end DICOM -> NIfTI conversion against real series
```

## Running

```bash
pytest                 # everything; integration tests self-skip without data (see below)
pytest test/unit        # unit suite only — no real data needed
pytest test/integration --download-integration-data # integration suite only — downloads needed data to tmp
```

`pyproject.toml` (`[tool.pytest.ini_options]`) sets `testpaths = ["test"]`, coverage
(`--cov musiq --cov-report term-missing`) and `filterwarnings = ["error"]` for every run —
the last one turns *any* warning (not just assertions) into a test failure, which is how e.g. an
unclosed `os.scandir()` iterator's `ResourceWarning` gets caught as a real bug rather than a log
line nobody reads.

## `dcm2niix`

Several unit tests and all integration tests run the real `dcm2niix` binary rather than mocking
it, to catch behavior mismatches a mock would hide (multi-file output, sidecar naming, `_Eq_1`
resampling, ...). It's installed via the `dcm2niix` PyPI package (pinned in `pyproject.toml`,
already part of the main dependency list), which ships a console-script wrapper for
Windows/Linux/macOS — a plain `pip install -r requirements.txt` is enough, no separate binary
download. Tests that need it are skipped automatically (`dcm2niix_available` fixture) when it
isn't found on `PATH`.

## Integration test data (TCIA)

`test/integration` tests convert genuine DICOM series (not synthetic placeholders) pulled from
[The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) — see the main
[README's Acknowledgements section](../README.md#acknowledgements) for the required citations if
you use this data. Tests in this directory skip automatically unless one of the two options below
is configured.

**Option 1 — auto-download** just the series these tests need, straight from the public TCIA REST
API (no login, no Java NBIA Data Retriever), into a temp dir that is deleted again once the test
session ends:

```bash
pytest test/integration --download-integration-data
```

Opt-in only — it hits an external network service and fetches real (de-identified) patient
imaging data on every run, so it's never on by default.

**Option 2 — point at a cohort you already downloaded** yourself via the NBIA Data Retriever,
using the manifest files in `test/integration/`:

- `manifest-1773751814915.tcia` — TCGA-PRAD series (CT/MR/PET conversion tests)
- `manifest-acrin-nsclc-fdg-pet.tcia` — one ACRIN-NSCLC-FDG-PET CT series, used only by the
  irregular-slice-spacing affine-repair test
- `manifest-flair-tracew-dixon.tcia` — one series each from ReMIND (FLAIR), ACRIN-6698 (TRACEW
  diffusion trace) and ISPY2 (GE IDEAL water/fat, the DIXON-equivalent technique) — MR contrast
  types the TCGA-PRAD series above don't cover

Download all three into the same root directory (so it ends up containing `TCGA-PRAD/`,
`ACRIN-NSCLC-FDG-PET/`, `ReMIND/`, `ACRIN-6698/` and `ISPY2/` subfolders), then point pytest at
that root:

```bash
pytest test/integration --integration-data-dir "/path/to/that/root"
MUSIQ_INTEGRATION_DATA_DIR="/path/to/that/root" pytest test/integration
```

This never hits the network: a series missing from your manual download (e.g. one added to
`_DOWNLOAD_TARGETS` in `test/integration/conftest.py` after you downloaded the manifests) just
makes that one test skip, same as an outdated/partial manifest always has — use
`--download-integration-data` instead if you want every needed series fetched for you.
