# xcs-code-stockpile

Correlation and diffraction analysis tools for experiments in the XCS hutch at LCLS.

The package works with LCLS smalldata HDF5 files and provides a common workflow for:

- mapping HDF5 datasets to logical experiment fields;
- constructing and filtering per-shot metadata;
- calculating equilibrium and nonequilibrium correlations;
- extracting time-resolved XRD and rocking-curve statistics; and
- inspecting beam and detector behavior with diagnostic plots.

## Typical workflow

```python
import xcs_code_stockpile as xcs

h5dict = xcs.H5Dictionary()

with xcs.load_smalldata(run_number=144, proposal_number=12345) as h5:
    metadata = xcs.create_metadata(
        h5,
        run=144,
        h5dict=h5dict,
        roi=(slice(100, 200), slice(300, 400)),
    )
    shots = xcs.sensible_filter(metadata, 1, 99, 1, 99)

    g2_minus_one = xcs.calc_static_correlation(
        h5,
        h5dict,
        metadata,
        shots & ~metadata["laser_on"],
        roi=(slice(100, 200), slice(300, 400)),
    )
```

See [Getting started](getting-started.md) for setup and a fuller walkthrough, or go directly to the [API reference](reference/index.md).

!!! warning "Detector coordinates"
    ROI coordinates refer to the frame stored in the smalldata file. They are not necessarily coordinates in the detector's full frame.
