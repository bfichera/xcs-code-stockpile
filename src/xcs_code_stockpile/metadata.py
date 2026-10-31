from ._utils import parsed_roi
from .datahandler import H5Dictionary
from ._log import logger
from typing import Any, Optional, Union, Callable

import h5py
import numpy as np
import pandas as pd


@parsed_roi
def create_metadata(h5: h5py.File,
                    run: int,
                    h5dict: H5Dictionary,
                    roi: Optional[Union[list[int], tuple[slice,
                                                         slice]]] = None,
                    nan_policy: str = 'raise',
                    calc_intensity: bool = True,
                    laser_off_type_func: Optional[Callable] = None,
                    *args: Any,
                    **kwargs: Any) -> pd.DataFrame:
    """Build per-shot metadata from a smalldata HDF5 file.

    Args:
        h5: Open HDF5 file containing every dataset mapped by `h5dict`.
        run: Run number stored in the returned DataFrame's ``attrs``.
        h5dict: Mapping from field names to HDF5 datasets.
        roi: Detector region as ``[x0, x1, y0, y1]`` or a
            ``(row_slice, column_slice)`` tuple. `None` selects the full stored
            frame.
        nan_policy: ``"raise"`` rejects NaNs in per-shot ROI sums;
            ``"ignore"`` computes them with `numpy.nansum`.
        calc_intensity: If False, skip the ROI sum and fill
            ``intensity_in_roi`` with NaN.
        laser_off_type_func: Called as ``laser_off_type_func(h5, *args,
            **kwargs)``; must return one label per shot. If `None`, shots
            are labeled ``"on"`` or ``"off"`` from ``laser_on``.
        *args: Passed to `laser_off_type_func`.
        **kwargs: Passed to `laser_off_type_func`.

    Returns:
        A DataFrame with ``intensity_in_roi``, ``shot_idx``, ``event_time``,
            ``laser_on``, ``xray_on``, ``i0``, ``delay`` and ``laser_off_type``
            columns, plus any extra `h5dict` fields. The run number is
            available as ``result.attrs["run"]``.

    Raises:
        ValueError: If `nan_policy` is invalid or detector data contains NaNs
            while the policy is ``"raise"``.
    """

    if nan_policy not in ['raise', 'ignore']:
        raise ValueError(f'Invalid nan_policy: {nan_policy}')

    keydict = h5dict.model_dump()
    d = {}
    stack = h5[keydict.pop('stack')][..., *roi]

    if calc_intensity is True:
        if nan_policy == 'raise':
            intensity_in_roi = np.sum(stack, axis=(-1, -2))
            if np.isnan(intensity_in_roi).any():
                raise ValueError('Detector stack contains NaN values')
        else:
            intensity_in_roi = np.nansum(stack, axis=(-1, -2))
    else:
        intensity_in_roi = np.full(stack.shape[0], np.nan)

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
    if laser_off_type_func is None:
        _lonarr = metadata['laser_on'].to_numpy()
        metadata['laser_off_type'] = [
            'on' if _lonarr[i] == 1 else 'off' for i in range(len(_lonarr))
        ]
    else:
        metadata['laser_off_type'] = laser_off_type_func(h5, *args, **kwargs)

    return metadata
