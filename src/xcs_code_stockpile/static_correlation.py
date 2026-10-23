from typing import Optional, Union

import numpy as np
import pandas as pd
import h5py

from .datahandler import H5Dictionary
from ._utils import parsed_roi


def _twotime(stack: np.ndarray, denominator_type: str = 'mean'):
    if denominator_type not in ['mean', 'std', 'nanmean', 'nanstd']:
        raise NotImplementedError(
            f'Denominator type {denominator_type} not implemented')
    wf = stack.reshape(stack.shape[0], -1).T  # n_pixels x n_frames
    if denominator_type.startswith('nan'):
        mean_ = np.nanmean
        std_ = np.nanstd
    else:
        mean_ = np.mean
        std_ = np.std
    mn = mean_(wf, axis=0)
    numerator = (wf.T @ wf) / wf.shape[0]
    if denominator_type.endswith('mean'):
        denominator = mn[:, None] * mn
        offset = 1
    elif denominator_type.endswith('std'):
        numerator -= mn[:, None] * mn
        std = std_(wf, axis=0)
        denominator = std[:, None] * std
        offset = 0
    ret = numerator / denominator - offset
    np.fill_diagonal(ret, np.nan)
    return ret


def _g2_minus_one(twotime: np.ndarray):
    return np.array([
        np.nanmean(np.diagonal(twotime, offset=i)) if i != 0 else np.nan
        for i in range(twotime.shape[0])
    ])


@parsed_roi
def calc_static_correlation(
    h5: h5py.File,
    h5dict: H5Dictionary,
    metadata: pd.DataFrame,
    filter: np.ndarray,
    roi: Optional[Union[list[int], tuple[slice, slice]]] = None,
    denominator_type: str = 'mean',
    ret_twotime: bool = False
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Calculate an equilibrium correlation as a function of frame lag.

    Args:
        h5: Open HDF5 file containing the detector stack.
        h5dict: Mapping whose ``stack`` field identifies the detector dataset.
        metadata: Per-shot metadata containing a ``shot_idx`` column.
        filter: Boolean array or Series selecting metadata rows.
        roi: Detector region as ``[x0, x1, y0, y1]`` or a
            ``(row_slice, column_slice)`` tuple. `None` selects the full frame.
        denominator_type: ``"mean"`` computes normalized intensity
            correlation minus one; ``"std"`` computes Pearson correlation
            across pixels.
        ret_twotime: Return the two-time correlation matrix as well as its
            diagonal averages.

    Returns:
        A one-dimensional NumPy array indexed by retained-frame lag. Element
        zero is NaN. If `ret_twotime` is true, returns
        ``(g2_minus_one, twotime)`` instead. Lag is retained-frame number, not
        physical elapsed time.

    Raises:
        NotImplementedError: If `denominator_type` is unsupported.
    """
    metadata = metadata[filter]

    # TODO
    # Correlation lag is retained frame number, not physical elapsed time.
    stack = h5[h5dict.stack][..., *roi]
    stack = stack[metadata['shot_idx'].to_numpy(), ...]
    twotime = _twotime(stack, denominator_type=denominator_type)
    g2_minus_one = _g2_minus_one(twotime)

    if ret_twotime:
        return g2_minus_one, twotime

    return g2_minus_one
