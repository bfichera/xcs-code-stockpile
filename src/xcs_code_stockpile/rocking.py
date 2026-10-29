from typing import Optional, Union
from os import PathLike
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .datahandler import H5Dictionary
from ._utils import parsed_roi, slow_mean, slow_intensities, save_array_to_movie


@parsed_roi
def calc_rocking_stats(h5: h5py.File,
                       h5dict: H5Dictionary,
                       scan_variable: str,
                       metadata: pd.DataFrame,
                       filter: Union[np.ndarray, None],
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
    if filter is not None:
        metadata = metadata[filter].copy()

    def func(group):
        shots = group['shot_idx'].to_numpy()
        mean_i0 = np.mean(group['i0'].to_numpy())
        intensities = slow_intensities(h5, h5dict, shots, roi=roi)
        n_shots = len(shots)
        return pd.Series({
            'intensity_mean': np.mean(intensities) / mean_i0,
            'intensity_stderr': np.std(intensities) / np.sqrt(n_shots) / mean_i0,
            'n_shots': n_shots
        })

    return metadata.groupby(scan_variable).apply(func)


def save_rocking_movie(h5: h5py.File,
                       h5dict: H5Dictionary,
                       scan_variable: str,
                       metadata: pd.DataFrame,
                       filter: Union[np.ndarray, None],
                       save_dir: PathLike,
                       fname: str = '',
                       roi: Optional[Union[
                           list[int], tuple[slice, slice]
                       ]] = None) -> pd.DataFrame:

    if filter is not None:
        metadata = metadata[filter].copy()

    ims = []
    for angle, group in metadata.groupby(scan_variable):
        shots = group['shot_idx'].to_numpy()
        mean_i0 = np.mean(group['i0'].to_numpy())
        mean_im = slow_mean(h5, h5dict, shots, roi=roi) / mean_i0
        ims.append(mean_im)
    im_array = np.array(ims)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    vmin = np.nanpercentile(im_array, 1)
    vmax = np.nanpercentile(im_array, 99)
    cmap_str = 'viridis'
    save_array_to_movie(im_array,
                        save_dir / fname,
                        vmin=vmin,
                        vmax=vmax,
                        cmap=cmap_str)
