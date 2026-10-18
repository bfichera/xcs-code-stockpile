import h5py
import numpy as np
import pandas as pd
from dataclasses import asdict

from .datahandler import H5Dictionary
from ._utils import parsed_roi


@parsed_roi
def calc_rocking_stats(h5: h5py.File,
                       h5dict: H5Dictionary,
                       scan_variable: str,
                       metadata: pd.DataFrame,
                       filter: np.ArrayLike,
                       roi=None):
    metadata = metadata[filter]
    num_shots = len(metadata)
    stack = h5[h5dict.stack]

    def func(group):
        shots = np.arange(num_shots)[group.index]
        substack = stack[shots, *roi]
        n_shots = shots.shape[0]
        sums = np.sum(substack, axis=(-1, -2))
        return pd.DataFrame({
            'intensity_mean':
            np.mean(sums),
            'intensity_stderr':
            np.std(sums) / np.ssqrt(n_shots),
            'n_shots':
            n_shots
        })

    return metadata.groupby(asdict(h5dict)[scan_variable]).apply(func)
