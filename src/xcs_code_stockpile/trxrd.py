from typing import Optional, Union
from os import PathLike
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt

from .datahandler import H5Dictionary
from ._utils import parsed_roi


def save_array_to_movie(ims: np.ndarray,
                        path: Optional[PathLike] = None,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        cmap: str = 'viridis',
                        first_frame_repeats: int = 15):
    import imageio
    if vmin is None:
        vmin = np.nanmin(ims)
    if vmax is None:
        vmax = np.nanmax(ims)
    ims = np.clip(ims, vmin, vmax)
    regulated_ims = ((ims - np.nanmin(ims)) /
                     (np.nanmax(ims) - np.nanmin(ims)))
    cmap = plt.get_cmap(cmap)
    rgba_array = cmap(regulated_ims)
    rgb_array = rgba_array[..., :3]
    rgb_uint8 = (rgb_array * 255.0).astype(np.uint8)
    repeated_ims = []
    for i in range(rgb_uint8.shape[0]):
        if i == 0:
            for j in range(first_frame_repeats):
                repeated_ims.append(rgb_uint8[i, ...])
        else:
            repeated_ims.append(rgb_uint8[i, ...])

    if path is not None:
        imageio.mimsave(path, repeated_ims, fps=30)
    return regulated_ims


def _fwhm_1d(a, center):
    x = np.arange(a.shape[0]) - center
    sigma = np.sqrt(np.sum(a * x**2) / np.sum(a))
    return sigma * 2.335


@parsed_roi
def calc_trxrd_stats(
    h5: h5py.File,
    h5dict: H5Dictionary,
    metadata: pd.DataFrame,
    filter: np.ndarray,
    bin_interval: float,
    roi: Optional[Union[list[int], tuple[slice,
                                         slice]]] = None) -> pd.DataFrame:
    """Calculate delay-binned time-resolved diffraction statistics.

    Args:
        h5: Open HDF5 file containing the detector stack.
        h5dict: Mapping whose ``stack`` field identifies the detector dataset.
        metadata: Per-shot metadata containing ``shot_idx``, ``delay``, and
            ``i0`` columns.
        filter: Boolean array or Series selecting metadata rows.
        bin_interval: Width of left-closed delay bins, in the same units as
            the metadata ``delay`` column.
        roi: Detector region as ``[x0, x1, y0, y1]`` or a
            ``(row_slice, column_slice)`` tuple. `None` selects the full frame.

    Returns:
        A DataFrame indexed by delay-bin left edge. Columns are
        ``intensity_mean``, ``intensity_stderr``, ``com_i``, ``com_j``,
        ``fwhm_i``, ``fwhm_j``, and ``n_shots``. Intensities are normalized by
        mean ``i0``. Centers and moment-derived widths are ROI-relative pixels.
    """
    # First we apply the filter

    metadata = metadata[filter].copy()
    # Then we bin the data
    delay = metadata['delay'].to_numpy()
    bins = np.arange(delay.min(), delay.max() + bin_interval, bin_interval)
    if bins[-1] <= delay.max():
        bins = np.append(bins, bins[-1] + bin_interval)
    delay_bins = pd.cut(metadata['delay'], bins, right=False, labels=bins[:-1])
    metadata['delay_bin'] = delay_bins

    stack = h5[h5dict.stack][..., *roi]

    def func(group):
        on_shots = group[group['laser_on']]['shot_idx'].to_numpy()
        off_shots = group[~group['laser_on']]['shot_idx'].to_numpy()
        on_substack = stack[on_shots, ...]
        off_substack = stack[off_shots, ...]
        n_on_shots = on_shots.shape[0]
        n_off_shots = off_shots.shape[0]
        on_sums = np.sum(on_substack, axis=(-1, -2))
        off_sums = np.sum(off_substack, axis=(-1, -2))
        on_mean_i0 = np.mean(group[group['laser_on']]['i0'])
        off_mean_i0 = np.mean(group[~group['laser_on']]['i0'])
        mean_on_im = np.mean(on_substack, axis=0)
        mean_off_im = np.mean(off_substack, axis=0)
        on_com_i, on_com_j = center_of_mass(mean_on_im)
        off_com_i, off_com_j = center_of_mass(mean_off_im)
        return pd.Series({
            'on_intensity_mean':
            np.mean(on_sums) / on_mean_i0,
            'on_intensity_stderr':
            (np.std(on_sums) / np.sqrt(n_on_shots) / on_mean_i0),
            'on_com_i':
            on_com_i,
            'on_com_j':
            on_com_j,
            'on_fwhm_i':
            _fwhm_1d(np.sum(mean_on_im, axis=1), on_com_i),
            'on_fwhm_j':
            _fwhm_1d(np.sum(mean_on_im, axis=0), on_com_j),
            'n_on_shots':
            n_on_shots,
            'off_intensity_mean':
            np.mean(off_sums) / off_mean_i0,
            'off_intensity_stderr':
            (np.std(off_sums) / np.sqrt(n_off_shots) / off_mean_i0),
            'off_com_i':
            off_com_i,
            'off_com_j':
            off_com_j,
            'off_fwhm_i':
            _fwhm_1d(np.sum(mean_off_im, axis=1), off_com_i),
            'off_fwhm_j':
            _fwhm_1d(np.sum(mean_off_im, axis=0), off_com_j),
            'n_off_shots':
            n_off_shots
        })

    return metadata.groupby('delay_bin').apply(func)


# TODO
# Make a movie
@parsed_roi
def save_movie(
    h5: h5py.File,
    h5dict: H5Dictionary,
    metadata: pd.DataFrame,
    filter: np.ndarray,
    bin_interval: float,
    save_dir: PathLike,
    fname_prefix: str = '',
    roi: Optional[Union[list[int], tuple[slice, slice]]] = None,
) -> pd.DataFrame:
    metadata = metadata[filter].copy()
    # Then we bin the data
    delay = metadata['delay'].to_numpy()
    bins = np.arange(delay.min(), delay.max() + bin_interval, bin_interval)
    if bins[-1] <= delay.max():
        bins = np.append(bins, bins[-1] + bin_interval)
    delay_bins = pd.cut(metadata['delay'], bins, right=False, labels=bins[:-1])
    metadata['delay_bin'] = delay_bins

    stack = h5[h5dict.stack][..., *roi]
    delays = []
    on_ims = []
    off_ims = []
    diff_ims = []
    for delay_bin, group in metadata.groupby('delay_bin'):
        on_shots = group[group['laser_on']]['shot_idx'].to_numpy()
        off_shots = group[~group['laser_on']]['shot_idx'].to_numpy()
        on_substack = stack[on_shots, ...]
        off_substack = stack[off_shots, ...]
        on_mean_i0 = np.mean(group[group['laser_on']]['i0'])
        off_mean_i0 = np.mean(group[~group['laser_on']]['i0'])
        mean_on_im = np.mean(on_substack, axis=0)
        mean_off_im = np.mean(off_substack, axis=0)
        mean_normalized_on_im = mean_on_im / on_mean_i0
        mean_normalized_off_im = mean_off_im / off_mean_i0

        delays.append(delay_bin)
        on_ims.append(mean_normalized_on_im)
        off_ims.append(mean_normalized_off_im)
        diff_ims.append(mean_normalized_on_im / mean_normalized_off_im - 1.0)

    delays = np.array(delays)
    on_ims = np.array(on_ims)
    off_ims = np.array(off_ims)
    diff_ims = np.array(diff_ims)

    save_dir = Path(save_dir)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    for im_array, fname in zip([on_ims, off_ims, diff_ims], [
            f'{fname_prefix}{fname_suffix}.gif'
            for fname_suffix in ['on', 'off', 'diff']
    ]):
        vmin = np.nanpercentile(im_array, 1)
        vmax = np.nanpercentile(im_array, 99)
        if fname.endswith('diff.gif'):
            cmap_str = 'RdBu_r'
        else:
            cmap_str = 'viridis'
        save_array_to_movie(im_array, save_dir / fname, vmin=vmin, vmax=vmax, cmap=cmap_str)
