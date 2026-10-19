import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

from .datahandler import H5Dictionary
from ._utils import parsed_roi


def _fwhm_1d(a, center):
    x = np.arange(a.shape[0]) - center
    sigma = np.sqrt(np.sum(a * x**2) / np.sum(a))
    return sigma * 2.335


@parsed_roi
def calc_trxrd_stats(h5: h5py.File,
                     h5dict: H5Dictionary,
                     metadata: pd.DataFrame,
                     filter: np.ndarray,
                     bin_interval: float,
                     roi=None):
    # First we apply the filter
    metadata = metadata[filter].copy()
    # Then we bin the data
    delay = metadata['delay'].to_numpy()
    bins = np.arange(delay.min(), delay.max() + bin_interval, bin_interval)
    if bins[-1] <= delay.max():
        bins = np.append(bins, bins[-1] + bin_interval)
    delay_bins = pd.cut(metadata['delay'],
                        bins,
                        right=False,
                        labels=bins[:-1])
    metadata['delay_bin'] = delay_bins

    stack = h5[h5dict.stack][..., *roi]

    def func(group):
        shots = group['shot_idx'].to_numpy()
        substack = stack[shots, ...]
        n_shots = shots.shape[0]
        sums = np.sum(substack, axis=(-1, -2))
        mean_i0 = np.mean(group['i0'])
        mean_im = np.mean(substack, axis=0)
        com_i, com_j = center_of_mass(mean_im)
        return pd.Series({
            'intensity_mean': np.mean(sums) / mean_i0,
            'intensity_stderr': (np.std(sums) / np.sqrt(n_shots) /
                                 mean_i0),
            'com_i': com_i,
            'com_j': com_j,
            'fwhm_i': _fwhm_1d(np.sum(mean_im, axis=1), com_i),
            'fwhm_j': _fwhm_1d(np.sum(mean_im, axis=0), com_j),
            'n_shots': n_shots
        })

    return metadata.groupby('delay_bin').apply(func)
