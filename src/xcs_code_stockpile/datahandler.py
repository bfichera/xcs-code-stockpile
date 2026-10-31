from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import Any, Union
from collections.abc import Callable

import h5py

from ._log import logger

try:
    import hdf5plugin  # noqa
except ImportError:
    logger.warning('No hdf5plugin found')


class SmartH5(h5py.File):
    """An `h5py.File` that also accepts a function as a key.

    ``h5[func]`` returns ``func(h5)``. This is what lets `H5Dictionary`
    fields be functions.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        return super().__init__(*args, **kwargs)

    def __getitem__(self, key: Union[str, Callable]) -> Any:
        if hasattr(key, '__call__'):
            return key(self)
        return super().__getitem__(key)


class H5Dictionary(BaseModel):
    """Map analysis fields to datasets in a smalldata HDF5 file.

    Each field is a dataset path, or a function that takes the open file and
    returns an array (the file must then be a `SmartH5`). Extra fields are
    allowed and become columns in the metadata.

    Attributes:
        event_time: Path to the one-dimensional event-time dataset.
        laser_on: Path to the one-dimensional pump-laser status dataset.
        xray_on: Path to the one-dimensional X-ray status dataset.
        i0: Path to the one-dimensional incident-intensity dataset.
        stack: Path to detector frames shaped ``(shots, rows, columns)``.
        delay: Path to the one-dimensional pump-probe delay dataset.
    """

    model_config = ConfigDict(extra='allow')

    event_time: Union[str, Callable] = '/event_time'
    laser_on: Union[str, Callable] = '/lightStatus/laser'
    xray_on: Union[str, Callable] = '/lightStatus/xray'
    i0: Union[str, Callable] = '/ipm4/sum'
    stack: Union[str, Callable] = '/zyla_ladm/ROI_area'
    delay: Union[str, Callable] = '/enc/lasdelay'


def load_smalldata(run_number: int, proposal_number: int,
                   **h5_kws: Any) -> 'SmartH5':
    """Open an XCS smalldata file for reading.

    The path is constructed as
    ``/sdf/data/lcls/ds/xcs/xcs{proposal_number}/hdf5/smalldata/``
    ``xcs{proposal_number}_Run{run_number:04d}.h5``.

    Args:
        run_number: LCLS run number.
        proposal_number: XCS proposal number, without the ``xcs`` prefix.
        **h5_kws: Additional keyword arguments passed to `h5py.File`.

    Returns:
        An open SmartH5 file in read-only mode. The caller must close the file;
            using the result as a context manager is recommended.
    """
    root = Path('/sdf/data/lcls/ds/xcs')
    path = (root / f'xcs{proposal_number}/hdf5/smalldata/'
            f'xcs{proposal_number}_Run{run_number:04d}.h5')
    return SmartH5(path, 'r', **h5_kws)
