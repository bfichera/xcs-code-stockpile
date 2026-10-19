# Getting started

## Installation

The package currently targets Python 3.9 and later. Install it from a local checkout:

```bash
python -m pip install .
```

For development and documentation work, install the optional documentation dependencies:

```bash
python -m pip install -e ".[docs]"
```

## Open a smalldata file

At LCLS, `load_smalldata` constructs the standard XCS smalldata path from the proposal and run numbers. It returns an open `h5py.File`; use it as a context manager so it is closed after analysis.

```python
import xcs_code_stockpile as xcs

h5dict = xcs.H5Dictionary()

with xcs.load_smalldata(144, 12345) as h5:
    print(h5[h5dict.stack].shape)
```

For files elsewhere, open them directly with `h5py`:

```python
import h5py
import xcs_code_stockpile as xcs

h5dict = xcs.H5Dictionary()
with h5py.File("smalldata.h5", "r") as h5:
    metadata = xcs.create_metadata(h5, run=144, h5dict=h5dict)
```

If your HDF5 paths differ from the defaults, pass them when constructing `H5Dictionary`:

```python
h5dict = xcs.H5Dictionary(
    i0="/diagnostics/i0",
    stack="/detector/roi",
    delay="/timing/delay",
)
```

## Create and filter metadata

`create_metadata` reads the one-dimensional shot fields and computes integrated detector intensity for each shot. The resulting DataFrame keeps the original HDF5 stack row in `shot_idx`.

```python
roi = (slice(100, 200), slice(300, 400))
metadata = xcs.create_metadata(h5, 144, h5dict, roi=roi)

shots = xcs.sensible_filter(
    metadata,
    i0_minimum_percentile=1,
    i0_maximum_percentile=99,
    ratio_min_percentile=1,
    ratio_max_percentile=99,
)
```

The percentile arguments are in the inclusive range normally used by `numpy.percentile`; `0` and `100` retain the full finite range.

## Select an analysis

- Use [`calc_static_correlation`](reference/correlations.md#xcs_code_stockpile.calc_static_correlation) for equilibrium XPCS data.
- Use [`differential_correlation`](reference/correlations.md#xcs_code_stockpile.differential_correlation) for pump-probe correlations.
- Use [`calc_trxrd_stats`](reference/diffraction.md#xcs_code_stockpile.calc_trxrd_stats) for delay-binned diffraction statistics.
- Use [`calc_rocking_stats`](reference/diffraction.md#xcs_code_stockpile.calc_rocking_stats) for scans grouped by angle or another scan variable.

## Build these docs locally

```bash
mkdocs serve
```

Open `http://127.0.0.1:8000` for live-reloading documentation. To perform the same strict build used in CI:

```bash
mkdocs build --strict
```
