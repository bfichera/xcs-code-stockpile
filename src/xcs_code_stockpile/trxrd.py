from typing import Optional, Union
from os import PathLike
from pathlib import Path
from itertools import product
from collections.abc import Hashable

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass

from .datahandler import H5Dictionary
from ._utils import parsed_roi, save_array_to_movie, _round_bins


def _fwhm_1d(a, center):
    x = np.arange(a.shape[0]) - center
    sigma = np.sqrt(np.sum(a * x**2) / np.sum(a))
    return sigma * 2.355


@parsed_roi
def calc_trxrd_stats(h5: h5py.File,
                     h5dict: H5Dictionary,
                     metadata: pd.DataFrame,
                     filter: Union[np.ndarray, pd.Series, None],
                     bin_interval: float,
                     roi: Optional[Union[list[int], tuple[slice,
                                                          slice]]] = None,
                     compute_image_stats: bool = True) -> pd.DataFrame:
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
        compute_image_stats: If False, skip the center of mass and FWHM
            (those columns are NaN).

    Returns:
        A DataFrame indexed by delay-bin left edge. Each column is prefixed
        by the ``laser_off_type`` it describes, e.g. ``on_intensity_mean``.
        The columns are ``intensity_mean``, ``intensity_stderr``, ``n_shots``,
        ``com_i``, ``com_j``, ``fwhm_i`` and ``fwhm_j``. Intensities are
        normalized by mean ``i0``. Centers and widths are in ROI-relative
        pixels.
    """
    # First we apply the filter

    if filter is not None:
        metadata = metadata[filter].copy()
    # Then we bin the data
    delay = metadata['delay'].to_numpy()
    bins = _round_bins(np.arange(delay.min(), delay.max() + bin_interval, bin_interval))
    if bins[-1] <= delay.max():
        bins = np.append(bins, bins[-1] + bin_interval)
    delay_bins = pd.cut(metadata['delay'], bins, right=False, labels=bins[:-1])
    metadata['delay_bin'] = delay_bins

    stack = h5[h5dict.stack][..., *roi]

    def func(group):
        out = {}
        for laser_off_type in metadata['laser_off_type'].unique():
            shots = group[group['laser_off_type'] ==
                          laser_off_type]['shot_idx'].to_numpy()
            substack = stack[shots, ...]
            n_shots = shots.shape[0]
            sums = np.sum(substack, axis=(-1, -2))
            mean_i0 = np.mean(
                group[group['laser_off_type'] == laser_off_type]['i0'])
            key_prefix = str(laser_off_type) + '_'
            out[key_prefix + 'intensity_mean'] = np.mean(sums) / mean_i0
            out[key_prefix +
                'intensity_stderr'] = np.std(sums) / np.sqrt(n_shots) / mean_i0
            out[key_prefix + 'n_shots'] = n_shots
            out[key_prefix + 'com_i'] = np.nan
            out[key_prefix + 'com_j'] = np.nan
            out[key_prefix + 'fwhm_i'] = np.nan
            out[key_prefix + 'fwhm_j'] = np.nan
            if compute_image_stats:
                mean_im = np.mean(substack, axis=0)
                com_i, com_j = center_of_mass(mean_im)
                fwhm_i, fwhm_j = (_fwhm_1d(np.sum(mean_im, axis=1), com_i),
                                  _fwhm_1d(np.sum(mean_im, axis=0), com_j))
                out[key_prefix + 'com_i'] = com_i
                out[key_prefix + 'com_j'] = com_j
                out[key_prefix + 'fwhm_i'] = fwhm_i
                out[key_prefix + 'fwhm_j'] = fwhm_j
        return pd.Series(out)

    return metadata.groupby('delay_bin').apply(func)


@parsed_roi
def save_movie(h5: h5py.File,
               h5dict: H5Dictionary,
               metadata: pd.DataFrame,
               filter: Union[np.ndarray, pd.Series],
               bin_interval: float,
               save_dir: PathLike,
               fname_prefix: str = '',
               roi: Optional[Union[list[int], tuple[slice, slice]]] = None,
               on_types: list[Hashable] = ['on'],
               off_types: list[Hashable] = ['off']) -> None:
    """Save delay-binned detector movies as GIFs.

    For each pair in `on_types` x `off_types`, writes three files to
    `save_dir`: ``on_{on}.gif``, ``off_{off}.gif`` and
    ``diff_{on}_{off}.gif`` (on / off - 1). Frames are mean images,
    normalized by ``i0``, one per delay bin.

    Args:
        h5: Open HDF5 file containing the detector stack.
        h5dict: Mapping whose ``stack`` field identifies the detector dataset.
        metadata: Per-shot metadata with ``shot_idx``, ``delay``, ``i0`` and
            ``laser_off_type`` columns.
        filter: Boolean array or Series selecting metadata rows.
        bin_interval: Delay bin width, in the units of ``delay``.
        save_dir: Output directory. Created if missing.
        fname_prefix: Prepended to every file name.
        roi: Detector region as ``[x0, x1, y0, y1]`` or a
            ``(row_slice, column_slice)`` tuple. `None` selects the full frame.
        on_types: ``laser_off_type`` values to use as the "on" images.
        off_types: ``laser_off_type`` values to use as the "off" images.
    """
    metadata = metadata[filter].copy()
    # Then we bin the data
    delay = metadata['delay'].to_numpy()
    bins = _round_bins(np.arange(delay.min(), delay.max() + bin_interval, bin_interval))
    if bins[-1] <= delay.max():
        bins = np.append(bins, bins[-1] + bin_interval)
    delay_bins = pd.cut(metadata['delay'], bins, right=False, labels=bins[:-1])
    metadata['delay_bin'] = delay_bins

    for on_type, off_type in product(on_types, off_types):
        stack = h5[h5dict.stack][..., *roi]
        delays = []
        on_ims = []
        off_ims = []
        diff_ims = []
        for delay_bin, group in metadata.groupby('delay_bin'):
            on_shots = group[group['laser_off_type'] ==
                             on_type]['shot_idx'].to_numpy()
            off_shots = group[group['laser_off_type'] ==
                              off_type]['shot_idx'].to_numpy()
            on_substack = stack[on_shots, ...]
            off_substack = stack[off_shots, ...]
            on_mean_i0 = np.mean(group.loc[group['laser_off_type'] == on_type,
                                           'i0'])
            off_mean_i0 = np.mean(
                group.loc[group['laser_off_type'] == off_type, 'i0'])
            mean_on_im = np.mean(on_substack, axis=0)
            mean_off_im = np.mean(off_substack, axis=0)
            mean_normalized_on_im = mean_on_im / on_mean_i0
            mean_normalized_off_im = mean_off_im / off_mean_i0

            delays.append(delay_bin)
            on_ims.append(mean_normalized_on_im)
            off_ims.append(mean_normalized_off_im)
            diff_ims.append(mean_normalized_on_im / mean_normalized_off_im -
                            1.0)

        delays = np.array(delays)
        on_ims = np.array(on_ims)
        off_ims = np.array(off_ims)
        diff_ims = np.array(diff_ims)

        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        for im_array, fname in zip([on_ims, off_ims, diff_ims], [
                f'{fname_prefix}{fname_suffix}.gif' for fname_suffix in [
                    f'on_{on_type}',
                    f'off_{off_type}',
                    f'diff_{on_type}_{off_type}',
                ]
        ]):
            vmin = np.nanpercentile(im_array, 1)
            vmax = np.nanpercentile(im_array, 99)
            if fname.startswith(f'{fname_prefix}diff_'):
                cmap_str = 'RdBu_r'
            else:
                cmap_str = 'viridis'
            save_array_to_movie(im_array,
                                save_dir / fname,
                                vmin=-max(abs(vmin), abs(vmax)),
                                vmax=max(abs(vmin), abs(vmax)),
                                cmap=cmap_str)
