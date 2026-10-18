from ._utils import parsed_roi
from datahandler import H5Dictionary
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

    If `h5dict` defines a `stack` key, also adds the intensity in `roi` to the
    dataframe."""

    keydict = asdict(h5dict)
    d = {'run': run}
    if 'stack' in keydict:
        stack = h5[keydict.pop('stack')]

        if nan_policy == 'raise':
            _sum = np.sum
        elif nan_policy == 'ignore':
            _sum = np.nansum

        intensity_in_roi = _sum(stack, axis=(-1, -2))
        d['intensity_in_roi'] = intensity_in_roi
    d['shot_idx'] = np.arange(stack.shape[0])

    d.update({h5[v][...] for k, v in keydict.items()})

    return pd.DataFrame(d)
