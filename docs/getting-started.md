# Getting started

## Installation

The package currently targets Python 3.9 and later. Install it by cloning the repository and running:

```bash
python -m pip install .
```

Alternatively, to use it with `s3df`, use a custom jupyter image (conda environment) with the following in the "commands to initiate jupyter" box:

```bash
source /sdf/data/lcls/ds/xcs/xcs101927227/results/bfichera/bryans-psconda.sh
```

## Open a smalldata file

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
    i0="/.../...",
    stack="/.../...",
    delay="/.../...",
)
```

## Create and filter metadata

`create_metadata` makes a pandas dataframe with a column for everything in `h5dict`. Each row is a different shot, where the index of the frame in the stack is in a column called `shot_idx`.

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
