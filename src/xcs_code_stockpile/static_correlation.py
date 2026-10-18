import numpy as np
import pandas as pd
import h5py

from .datahandler import H5Dictionary
from ._utils import parsed_roi


def _twotime(stack: np.ndarray, denominator_type: str = 'mean'):
    if denominator_type not in ['mean', 'std']:
        raise NotImplementedError(
            f'Denominator type {denominator_type} not implemented')
    wf = stack.reshape(stack.shape[0], -1).T  # n_pixels x n_frames
    mn = np.mean(wf, axis=0)
    numerator = (wf.T @ wf) / wf.shape[0]
    if denominator_type == 'mean':
        denominator = mn[:, None] * mn
    elif denominator_type == 'std':
        std = np.std(wf, axis=0)
        denominator = std[:, None] * std
    return numerator / denominator


def _g2_minus_one(twotime: np.ndarray, denominator_type: str = 'mean'):
    return np.array([
        np.mean(np.diagonal(twotime, offset=i)) if i != 0 else np.nan
        for i in range(twotime.shape[0])
    ])


@parsed_roi
def calc_static_correlation(h5: h5py.File,
                            h5dict: H5Dictionary,
                            metadata: pd.DataFrame,
                            filter: np.ndarray,
                            roi: list[int, int, int, int] | tuple[slice, slice]
                            | None = None,
                            denominator_type: str = 'mean',
                            ret_twotime: bool = False):
    metadata = metadata[filter]
    stack = h5[h5dict.stack][..., *roi]
    stack = stack[metadata['shot_idx'].to_numpy(), ...]
    twotime = _twotime(stack, denominator_type=denominator_type)
    g2_minus_one = _g2_minus_one(twotime, denominator_type=denominator_type)

    if ret_twotime:
        return g2_minus_one, twotime

    return g2_minus_one
