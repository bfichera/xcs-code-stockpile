# HDF5 and metadata

## Required HDF5 data

`H5Dictionary` maps logical names used by the analysis functions to HDF5 dataset paths. The default mapping is:

| Field | Default path | Expected contents |
| --- | --- | --- |
| `event_time` | `/event_time` | One timestamp per shot |
| `laser_on` | `/lightStatus/laser` | Pump-laser status per shot |
| `xray_on` | `/lightStatus/xray` | X-ray status per shot |
| `i0` | `/ipm4/sum` | Incident intensity per shot |
| `stack` | `/zyla_ladm/ROI_area` | Detector data shaped `(shots, rows, columns)` |
| `delay` | `/enc/lasdelay` | Pump-probe delay per shot |

All one-dimensional datasets must align with the first dimension of `stack`.

## Metadata schema

`create_metadata` returns a `pandas.DataFrame` with one row per shot and these columns:

| Column | Meaning |
| --- | --- |
| `intensity_in_roi` | Sum of detector values inside the selected ROI |
| `shot_idx` | Corresponding index in the HDF5 detector stack |
| `event_time` | Value from the mapped event-time dataset |
| `laser_on` | Boolean pump-laser state |
| `xray_on` | Boolean X-ray state |
| `i0` | Incident intensity |
| `delay` | Pump-probe delay |

The run number is stored in `metadata.attrs["run"]`.

Analyses use `shot_idx`, rather than the DataFrame index, to retrieve detector frames. This remains important after filtering or otherwise subsetting metadata.

## Units

The package does not convert units. `event_time`, `delay`, bin intervals, and arbitrary scan variables retain the units of the source data. A bin interval must therefore use the same unit as the column it bins.

## NaN handling

By default, `create_metadata` raises `ValueError` when any per-shot ROI sum is NaN. Set `nan_policy="ignore"` to use `numpy.nansum`; a frame containing only NaNs then has an integrated intensity of zero.
