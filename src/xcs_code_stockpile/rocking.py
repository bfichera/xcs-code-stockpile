import h5py
import numpy as np
import pandas as pd

from .datahandler import H5Dictionary
from ._utils import parsed_roi


@parsed_roi
def calc_rocking_stats(h5: h5py.File,
                       h5dict: H5Dictionary,
                       scan_variable: str,
                       metadata: pd.DataFrame,
                       filter: np.ndarray,
                       roi=None):
    metadata = metadata[filter]
    stack = h5[h5dict.stack][..., *roi]

    def func(group):
        shots = group['shot_idx'].to_numpy()
        substack = stack[shots, ...]
        n_shots = shots.shape[0]
        sums = np.sum(substack, axis=(-1, -2))
        return pd.Series({
            'intensity_mean': np.mean(sums),
            'intensity_stderr': np.std(sums) / np.sqrt(n_shots),
            'n_shots': n_shots
        })

    return metadata.groupby(scan_variable).apply(func)
