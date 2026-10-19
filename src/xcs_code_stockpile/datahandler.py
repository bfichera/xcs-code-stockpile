from pathlib import Path
from dataclasses import dataclass

import h5py

from ._log import logger

try:
    import hdf5plugin  # noqa
except ImportError:
    logger.warning('No hdf5plugin found')


@dataclass
class H5Dictionary:
    event_time: str = '/event_time'
    laser_on: str = '/lightStatus/laser'
    xray_on: str = '/lightStatus/xray'
    i0: str = '/ipm4/sum'
    stack: str = '/zyla_ladm/ROI_area'
    delay: str = '/enc/lasdelay'


def load_smalldata(run_number: int, proposal_number: int, **h5_kws):
    """Returns the h5py.File object associated with a certain run"""
    root = Path('/sdf/data/lcls/ds/xcs')
    path = (root / f'xcs{proposal_number}/hdf5/smalldata/'
            f'xcs{proposal_number}_Run{run_number:04d}.h5')
    return h5py.File(path, 'r', **h5_kws)
