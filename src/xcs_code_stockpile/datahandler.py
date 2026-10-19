from pathlib import Path
from dataclasses import dataclass
from typing import Any

import h5py

from ._log import logger

try:
    import hdf5plugin  # noqa
except ImportError:
    logger.warning('No hdf5plugin found')


@dataclass
class H5Dictionary:
    """Map logical analysis fields to datasets in a smalldata HDF5 file.

    Attributes:
        event_time: Path to the one-dimensional event-time dataset.
        laser_on: Path to the one-dimensional pump-laser status dataset.
        xray_on: Path to the one-dimensional X-ray status dataset.
        i0: Path to the one-dimensional incident-intensity dataset.
        stack: Path to detector frames shaped ``(shots, rows, columns)``.
        delay: Path to the one-dimensional pump-probe delay dataset.
    """

    event_time: str = '/event_time'
    laser_on: str = '/lightStatus/laser'
    xray_on: str = '/lightStatus/xray'
    i0: str = '/ipm4/sum'
    stack: str = '/zyla_ladm/ROI_area'
    delay: str = '/enc/lasdelay'


def load_smalldata(run_number: int, proposal_number: int,
                   **h5_kws: Any) -> h5py.File:
    """Open an XCS smalldata file for reading.

    The path is constructed as
    ``/sdf/data/lcls/ds/xcs/xcs{proposal_number}/hdf5/smalldata/``
    ``xcs{proposal_number}_Run{run_number:04d}.h5``.

    Args:
        run_number: LCLS run number.
        proposal_number: XCS proposal number, without the ``xcs`` prefix.
        **h5_kws: Additional keyword arguments passed to `h5py.File`.

    Returns:
        An open `h5py.File` in read-only mode. The caller must close the file;
        using the result as a context manager is recommended.
    """
    root = Path('/sdf/data/lcls/ds/xcs')
    path = (root / f'xcs{proposal_number}/hdf5/smalldata/'
            f'xcs{proposal_number}_Run{run_number:04d}.h5')
    return h5py.File(path, 'r', **h5_kws)
