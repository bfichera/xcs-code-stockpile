from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def common_enough_off_type_filter(metadata: pd.DataFrame) -> pd.Series:
    metadata = metadata.copy()
    vc = metadata['laser_off_type'].value_counts(normalize=True)
    typical_vc = np.mean(vc)
    valid_filter = metadata['laser_off_type'].map(vc > 0.1 * typical_vc)
    filter = pd.Series(False, index=metadata.index)
    filter.loc[metadata.index] = valid_filter

    return filter


def i0_min_max_filter(metadata: pd.DataFrame, i0_minimum_percentile: float,
                      i0_maximum_percentile: float) -> pd.Series:
    valid_input = (metadata['xray_on'] & (metadata['i0'] > 0.0)
                   & np.isfinite(metadata['i0']))
    valid_metadata = metadata[valid_input]
    i0_minimum = np.percentile(valid_metadata['i0'], i0_minimum_percentile)
    i0_maximum = np.percentile(valid_metadata['i0'], i0_maximum_percentile)
    valid_filter = ((valid_metadata['i0'] >= i0_minimum) &
                    (valid_metadata['i0'] <= i0_maximum))

    filter = pd.Series(False, index=metadata.index)
    filter.loc[valid_metadata.index] = valid_filter

    return filter & common_enough_off_type_filter(metadata)


def sensible_filter(metadata: pd.DataFrame, i0_minimum_percentile: float,
                    i0_maximum_percentile: float, ratio_min_percentile: float,
                    ratio_max_percentile: float) -> pd.Series:
    """Select finite X-ray shots in configurable intensity ranges.

    A linear model of ROI intensity against ``i0`` is fitted to finite X-ray
    shots. Selection then applies inclusive percentile limits to ``i0`` and to
    the residual divided by ``i0``.

    Args:
        metadata: Per-shot metadata containing ``xray_on``, ``i0``, and
            ``intensity_in_roi`` columns.
        i0_minimum_percentile: Lower percentile for accepted ``i0`` values.
        i0_maximum_percentile: Upper percentile for accepted ``i0`` values.
        ratio_min_percentile: Lower percentile for scaled fit residuals.
        ratio_max_percentile: Upper percentile for scaled fit residuals.

    Returns:
        A boolean Series aligned with `metadata`. False entries include X-ray
        off shots, zero or non-finite ``i0``, non-finite ROI intensity, and
        values outside the requested percentile ranges.
    """
    valid_input = (metadata['xray_on'] & (metadata['i0'] != 0)
                   & np.isfinite(metadata['i0'])
                   & np.isfinite(metadata['intensity_in_roi']))
    valid_metadata = metadata[valid_input]
    i0_minimum = np.percentile(valid_metadata['i0'], i0_minimum_percentile)
    i0_maximum = np.percentile(valid_metadata['i0'], i0_maximum_percentile)
    X = valid_metadata['i0'].to_numpy()[:, None]
    y = valid_metadata['intensity_in_roi'].to_numpy()
    model = LinearRegression()
    model.fit(X, y)
    scaled_residual = (y - model.predict(X)) / X[:, 0]
    scaled_residual_minimum = np.percentile(scaled_residual,
                                            ratio_min_percentile)
    scaled_residual_maximum = np.percentile(scaled_residual,
                                            ratio_max_percentile)
    valid_filter = ((valid_metadata['i0'] >= i0_minimum) &
                    (valid_metadata['i0'] <= i0_maximum) &
                    (scaled_residual >= scaled_residual_minimum) &
                    (scaled_residual <= scaled_residual_maximum))
    filter = pd.Series(False, index=metadata.index)
    filter.loc[valid_metadata.index] = valid_filter

    return filter & common_enough_off_type_filter(metadata)


def no_filter(metadata: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=metadata.index)
