# CT geometry helpers left `os.scandir()` iterators unclosed on early return

## Summary

`select_dominant_ct_acquisition` and `_slice_normal_and_extent` (`src/musiq/utils.py`) both scan
a DICOM series directory with a bare `for entry in os.scandir(dicom_dirpath):` loop, and both
`return` from *inside* that loop as soon as they detect a degenerate slice geometry (a zero-length
cross product of the row/column direction cosines — i.e. `ImageOrientationPatient` describing two
parallel, not perpendicular, in-plane axes). `os.scandir()` holds a native OS directory handle
(`FindFirstFile`/`FindNextFile` on Windows) that a `for` loop only closes once iteration is
*exhausted*; returning out of the loop body instead abandons the iterator with the handle still
open. The handle is then only released whenever the garbage collector happens to finalize the
`ScandirIterator` object, at which point Python emits a `ResourceWarning: unclosed scandir
iterator` from that finalizer — not from the function that leaked it, so the warning is easy to
miss unless something is actively watching for it.

This is a resource leak (an OS directory handle held open longer than necessary), not a data
leak — no data is exposed or written incorrectly; correctness of `select_dominant_ct_acquisition`/
`_slice_normal_and_extent`'s return values is unaffected.

## Root cause

`src/musiq/utils.py`, `select_dominant_ct_acquisition` (and, identically, `_slice_normal_and_extent`):

```python
normal_lps = None
files: list[tuple[str, str, float]] = []
for entry in os.scandir(dicom_dirpath):
    if not entry.is_file():
        continue
    try:
        ds = pydicom.dcmread(entry.path, stop_before_pixels=True)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    except Exception:
        continue
    if normal_lps is None:
        normal_lps = np.cross(iop[:3], iop[3:6])
        norm = np.linalg.norm(normal_lps)
        if norm == 0:
            return None   # <-- leaves the for loop (and its open scandir handle) mid-iteration
        normal_lps = normal_lps / norm
    ...
```

A `for` loop over `os.scandir()` only calls the iterator's `close()` (releasing the directory
handle) when the loop runs to completion or raises; a `return` statement executed from within the
loop body sidesteps that entirely, unlike the equivalent `with os.scandir(...) as entries:` form,
whose `__exit__` runs regardless of how the `with` block is left.

## Reproduction

Surfaced by a newly added unit test exercising exactly this branch — degenerate
`ImageOrientationPatient` on a synthetic DICOM series, no real data needed:

```
test/unit/utils/test_select_dominant_ct_acquisition.py::test_degenerate_orientation_returns_none
test/unit/utils/test_repair_slice_spacing_from_dicom.py::test_undeterminable_dicom_geometry_is_left_untouched
```

(`_slice_normal_and_extent` is a private helper shared by `repair_slice_direction_from_dicom` and
`repair_slice_spacing_from_dicom`; the second test above reaches its copy of the same early-return
pattern via `repair_slice_spacing_from_dicom`.) `pyproject.toml` sets
`filterwarnings = ["error"]`, so the `ResourceWarning` emitted when the leaked iterator was later
garbage-collected turned straight into a test failure — before this branch had a dedicated test,
nothing ever exercised the degenerate-orientation path, so the leak went unnoticed.

## Impact

- No incorrect results: both functions return the correct value (`None`) before the leak occurs.
- Each call against a series with degenerate orientation leaves one native directory handle open
  until the next garbage-collection cycle finalizes the abandoned iterator. Low real-world impact
  for a pipeline processing patients one at a time, but a genuine (if minor) resource leak that
  would compound if this code path were ever hit at higher frequency or in a long-running process.

## Fix

**Status: fixed.** Both loops now use `os.scandir()` as a context manager, so the handle is
released on every exit path, including an early `return`:

```python
with os.scandir(dicom_dirpath) as entries:
    for entry in entries:
        ...
        if norm == 0:
            return None   # the `with` block's __exit__ still closes the handle here
```

Behavior-neutral (same return values in every case) — verified by the full `test/unit` suite
(249 passed, 1 xfailed) and `ruff check`/`ruff format` passing on the changed file.
