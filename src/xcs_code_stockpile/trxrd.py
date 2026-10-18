import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

from .datahandler import H5Dictionary
from ._utils import parsed_roi


def _fwhm_1d(a, center):
    x = np.arange(a.shape[0]) - center
    sigma = np.sqrt(a * x**2 / np.sum(a))
    return sigma * 2.335


@parsed_roi
def calc_trxrd_stats(h5: h5py.File,
                     h5dict: H5Dictionary,
                     metadata: pd.DataFrame,
                     filter: np.ArrayLike,
                     bin_interval: float,
                     roi=None):
    # First we apply the filter
    metadata = metadata[filter]
    # Then we bin the data
    delay = metadata['delay'].to_numpy()
    bins = np.arange(delay.min(), delay.max() + bin_interval / 2, bin_interval)
    delay_bins = pd.cut(metadata[h5dict.delay],
                        bins,
                        right=False,
                        labels=bins[:-1])
    metadata['delay_bin'] = delay_bins
    num_shots = len(metadata)

    stack = h5[h5dict.stack]

    def func(group):
        shots = np.arange(num_shots)[group.index]
        substack = stack[shots, *roi]
        n_shots = shots.shape[0]
        sums = np.sum(substack, axis=(-1, -2))
        mean_im = np.mean(substack, axis=0)
        com_i, com_j = center_of_mass(mean_im)
        return pd.DataFrame({
            'intensity_mean': np.mean(sums),
            'intensity_stderr': np.std(sums) / np.sqrt(n_shots),
            'com_i': com_i,
            'com_j': com_j,
            'fwhm_i': _fwhm_1d(mean_im[com_i, :], com_j),
            'fwhm_j': _fwhm_1d(mean_im[:, com_j], com_i),
            'n_shots': n_shots
        })

    return metadata.groupby('delay_bin').apply(func)
