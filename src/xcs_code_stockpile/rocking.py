from typing import Optional, Union

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
                       roi: Optional[Union[
                           list[int], tuple[slice, slice]
                       ]] = None) -> pd.DataFrame:
    """Calculate integrated-intensity statistics for a rocking scan.

    Args:
        h5: Open HDF5 file containing the detector stack.
        h5dict: Mapping whose ``stack`` field identifies the detector dataset.
        scan_variable: Metadata column by which selected shots are grouped.
        metadata: Per-shot metadata containing ``shot_idx`` and the selected
            scan variable.
        filter: Boolean array or Series selecting metadata rows.
        roi: Detector region as ``[x0, x1, y0, y1]`` or a
            ``(row_slice, column_slice)`` tuple. `None` selects the full frame.

    Returns:
        A DataFrame indexed by scan value with ``intensity_mean``,
        ``intensity_stderr``, and ``n_shots`` columns. Intensities are
        normalized by ``i0``.
    """
    metadata = metadata[filter]
    i0 = metadata['i0']
    stack = h5[h5dict.stack][..., *roi]

    def func(group):
        shots = group['shot_idx'].to_numpy()
        substack = stack[shots, ...]
        n_shots = shots.shape[0]
        sums = np.sum(substack, axis=(-1, -2))
        return pd.Series({
            'intensity_mean': np.mean(sums) / np.mean(i0),
            'intensity_stderr': np.std(sums) / np.sqrt(n_shots) / np.mean(i0),
            'n_shots': n_shots
        })

    return metadata.groupby(scan_variable).apply(func)
