from ._utils import parsed_roi
from .datahandler import H5Dictionary
from ._log import logger
from dataclasses import asdict
from typing import Optional, Union

import h5py
import numpy as np
import pandas as pd


@parsed_roi
def create_metadata(h5: h5py.File,
                    run: int,
                    h5dict: H5Dictionary,
                    roi: Optional[Union[list[int], tuple[slice,
                                                         slice]]] = None,
                    nan_policy: str = 'raise') -> pd.DataFrame:
    """Build per-shot metadata from a smalldata HDF5 file.

    Args:
        h5: Open HDF5 file containing every dataset mapped by `h5dict`.
        run: Run number stored in the returned DataFrame's ``attrs``.
        h5dict: Mapping from logical field names to HDF5 dataset paths.
        roi: Detector region as ``[x0, x1, y0, y1]`` or a
            ``(row_slice, column_slice)`` tuple. `None` selects the full stored
            frame.
        nan_policy: ``"raise"`` rejects NaNs in per-shot ROI sums;
            ``"ignore"`` computes them with `numpy.nansum`.

    Returns:
        A DataFrame with ``intensity_in_roi``, ``shot_idx``, ``event_time``,
        ``laser_on``, ``xray_on``, ``i0``, and ``delay`` columns. The run
        number is available as ``result.attrs["run"]``.

    Raises:
        ValueError: If `nan_policy` is invalid or detector data contains NaNs
            while the policy is ``"raise"``.
        KeyError: If a mapped HDF5 dataset is missing.
    """

    if nan_policy not in ['raise', 'ignore']:
        raise ValueError(f'Invalid nan_policy: {nan_policy}')

    keydict = asdict(h5dict)
    d = {}
    stack = h5[keydict.pop('stack')][..., *roi]

    if nan_policy == 'raise':
        intensity_in_roi = np.sum(stack, axis=(-1, -2))
        if np.isnan(intensity_in_roi).any():
            raise ValueError('Detector stack contains NaN values')
    else:
        intensity_in_roi = np.nansum(stack, axis=(-1, -2))

    d['intensity_in_roi'] = intensity_in_roi
    d['shot_idx'] = np.arange(stack.shape[0])

    for k, v in keydict.items():
        try:
            d.update({k: h5[v][...]})
        except KeyError:
            logger.warning(f'Didn\'t find the key {k}...')

    metadata = pd.DataFrame(d)
    metadata[['laser_on', 'xray_on']] = metadata[['laser_on',
                                                  'xray_on']].astype(bool)
    metadata.attrs['run'] = run
    return metadata
