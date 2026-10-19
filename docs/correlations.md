# Correlations

## Static correlation

`calc_static_correlation` constructs a two-time matrix from selected frames and averages its diagonals to produce a correlation as a function of retained-frame lag.

For the default `denominator_type="mean"`, frames \(I_a\) and \(I_b\) are correlated as

\[
C_{ab} = \frac{\langle I_a I_b \rangle_p}
{\langle I_a \rangle_p \langle I_b \rangle_p} - 1,
\]

where \(\langle\cdot\rangle_p\) denotes a pixel average. With `denominator_type="std"`, the function computes the Pearson correlation across pixels.

```python
g2_minus_one, twotime = xcs.calc_static_correlation(
    h5,
    h5dict,
    metadata,
    shots & ~metadata["laser_on"],
    roi=roi,
    denominator_type="mean",
    ret_twotime=True,
)
```

Index zero of `g2_minus_one` is `NaN`; subsequent entries average positive-offset diagonals. Lag is measured in retained frames after filtering, not physical elapsed time.

## Differential correlation

`differential_correlation` compares pump-on images with pump-off references and returns a `pandas.Series` indexed by delay-bin left edges. It supports three assumptions about reversibility:

| Mode | Pump-off reference | Pump-on averaging |
| --- | --- | --- |
| `strong` | Mean off image in each lab-time bin | Mean on image for each delay and lab-time bin |
| `weak` | Mean off image in each lab-time bin | Correlate individual on images, then average |
| `none` | Most recent preceding off image | Correlate individual on images, then average by delay |

The `strong` and `weak` modes require `labtime_bin_interval`. The `none` mode does not use it.

```python
correlation = xcs.differential_correlation(
    "weak",
    h5,
    h5dict,
    metadata,
    shots,
    delay_bin_interval=0.1,
    labtime_bin_interval=60,
    roi=roi,
)
```

Delay and lab-time intervals must use the same units as `metadata["delay"]` and `metadata["event_time"]`, respectively. Bins are left-closed and right-open.
