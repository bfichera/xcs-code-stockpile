from ._utils import parsed_roi
from .datahandler import H5Dictionary
from dataclasses import asdict

import h5py
import numpy as np
import pandas as pd


@parsed_roi
def create_metadata(h5: h5py.File,
                    run: int,
                    h5dict: H5Dictionary,
                    roi=None,
                    nan_policy='raise'):
    """Create a pandas dataframe with all of the entries in `h5dict`.

    The `stack` entry is required and is used to add the intensity in `roi` to
    the dataframe."""

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

    d.update({k: h5[v][...] for k, v in keydict.items()})

    metadata = pd.DataFrame(d)
    metadata[['laser_on', 'xray_on']] = metadata[[
        'laser_on', 'xray_on'
    ]].astype(bool)
    metadata.attrs['run'] = run
    return metadata
