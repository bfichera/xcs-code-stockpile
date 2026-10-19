from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


def sensible_filter(metadata: pd.DataFrame, i0_minimum_percentile: float,
                    i0_maximum_percentile: float, ratio_min_percentile: float,
                    ratio_max_percentile: float):
    """This throws out all the shots where:

    - the x-ray was off
    - the i0 was too low or too high
    - the i-to-i0 ratio is not close enough to the linear regime
    """
    valid_input = (metadata['xray_on'] &
                   (metadata['i0'] != 0) &
                   np.isfinite(metadata['i0']) &
                   np.isfinite(metadata['intensity_in_roi']))
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

    return filter
