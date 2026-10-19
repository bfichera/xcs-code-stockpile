# Diffraction

## Time-resolved XRD

`calc_trxrd_stats` groups selected shots into left-closed delay bins. It reports:

| Column | Definition |
| --- | --- |
| `intensity_mean` | Mean integrated frame intensity divided by mean `i0` |
| `intensity_stderr` | Population standard deviation of integrated intensity divided by `sqrt(n)` and mean `i0` |
| `com_i`, `com_j` | Center of mass of the mean image, in ROI-relative pixels |
| `fwhm_i`, `fwhm_j` | Moment-derived width `2.335 sigma`, in pixels |
| `n_shots` | Number of selected shots in the bin |

```python
stats = xcs.calc_trxrd_stats(
    h5,
    h5dict,
    metadata,
    shots & metadata["laser_on"],
    bin_interval=0.1,
    roi=roi,
)
```

The widths are moment estimates, not fitted peak widths or direct half-maximum crossings.

## Rocking curves

`calc_rocking_stats` groups selected shots by an existing metadata column such as an angle or motor position:

```python
metadata["theta"] = measured_angles
rocking = xcs.calc_rocking_stats(
    h5,
    h5dict,
    "theta",
    metadata,
    shots,
    roi=roi,
)
```

The result is indexed by the scan value and contains `intensity_mean`, `intensity_stderr`, and `n_shots`. Rocking-curve intensity is not normalized by `i0`.
