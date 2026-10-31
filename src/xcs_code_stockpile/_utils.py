from functools import wraps
from datetime import datetime
from os import PathLike
from typing import Callable, Optional, Union

import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
from numpy.typing import ArrayLike

from .roi import _ROI
from .datahandler import H5Dictionary


def _bin_edges(values: np.ndarray, interval: float) -> np.ndarray:
    start = np.min(values)
    n_edges = int(np.floor((np.max(values) - start) / interval)) + 2
    return start + np.arange(n_edges) * interval


def _bin_labels(edges: np.ndarray) -> np.ndarray:
    interval = edges[1] - edges[0]
    decimals = max(0, int(np.ceil(-np.log10(interval))) + 6)
    return np.round(edges[:-1], decimals)


def parsed_roi(f: Callable) -> Callable:
    roi_position = f.__code__.co_varnames.index('roi')

    @wraps(f)
    def g(*args, **kwargs):
        args = list(args)
        if len(args) > roi_position:
            roi = args[roi_position]
        else:
            roi = kwargs.get('roi')

        parsed = (_ROI.from_slices(slice(None), slice(None))
                  if roi is None else _ROI._guess(roi))
        if len(args) > roi_position:
            args[roi_position] = parsed
        else:
            kwargs['roi'] = parsed
        return f(*args, **kwargs)

    return g


@parsed_roi
def slow_mean(
        h5: h5py.File,
        h5dict: H5Dictionary,
        shots: np.ndarray,
        roi: Optional[Union[list[int], tuple[slice,
                                             slice]]] = None) -> np.ndarray:
    stack = h5[h5dict.stack]
    shape = stack[0, *roi].shape
    if len(shots) == 0:
        return np.full(shape, np.nan)
    out = np.zeros(shape)
    n = len(shots)
    for i in shots:
        out += stack[i, *roi] / n
    return out


@parsed_roi
def slow_sum(
        h5: h5py.File,
        h5dict: H5Dictionary,
        shots: np.ndarray,
        roi: Optional[Union[list[int], tuple[slice,
                                             slice]]] = None) -> np.ndarray:
    stack = h5[h5dict.stack]
    shape = h5[h5dict.stack][0, *roi].shape
    if len(shots) == 0:
        return np.full(shape, np.nan)
    out = np.zeros(shape)
    for i in shots:
        out += stack[i, *roi]
    return out


@parsed_roi
def slow_intensities(
        h5: h5py.File,
        h5dict: H5Dictionary,
        shots: np.ndarray,
        roi: Optional[Union[list[int], tuple[slice,
                                             slice]]] = None) -> np.ndarray:
    stack = h5[h5dict.stack]
    out = np.zeros(len(shots))
    for i, shot_idx in enumerate(shots):
        out[i] = np.sum(stack[shot_idx, *roi])
    return out


@np.vectorize
def ts_to_s(ts: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    sec = ts >> 32 & 0xffffffff
    nsec = ts & 0xffffffff
    return datetime.fromtimestamp(sec + nsec * 1e-9).timestamp()


def _estimate_period(laser_on, max_period=500, min_score=0.5):
    n_shots = len(laser_on)
    max_period = min(max_period, n_shots // 2)
    on_fraction = np.mean(laser_on) if n_shots else 0.
    if max_period < 1 or on_fraction in (0., 1.):
        return None
    periods = np.arange(1, max_period + 1)
    on_shots = np.flatnonzero(laser_on)
    score = np.array([
        np.mean(laser_on[on_shots[on_shots + p < n_shots] + p])
        for p in periods
    ])
    score = (score - on_fraction) / (1 - on_fraction)
    best = score.max()
    if best < min_score:
        return None
    return int(periods[np.flatnonzero(score >= 0.6 * best)[0]])


def basic_off_type_encoder(laser_on: ArrayLike,
                           unknown_label: str = 'unknown_off',
                           max_period: int = 500) -> list[str]:
    """Label each shot by its position in the laser on/off cycle.

    Pump shots are labeled ``"000_on"``. Later shots are numbered by the
    shots since the last pump shot, e.g. ``"002_off"``. Off shots before the
    first pump shot are numbered by assuming the cycle repeats with its
    detected period. If no period is found, they get `unknown_label`.

    Args:
        laser_on: Pump status of each shot.
        unknown_label: Label for shots that can't be numbered.
        max_period: Longest cycle length to look for, in shots.

    Returns:
        One label per shot, for the ``laser_off_type`` column.
    """
    laser_on = np.asarray(laser_on).astype(bool)
    n_shots = len(laser_on)
    off_type = [None] * n_shots
    for i in range(n_shots):
        if laser_on[i]:
            off_type[i] = 0
        elif i > 0 and off_type[i - 1] is not None:
            off_type[i] = off_type[i - 1] + 1

    period = _estimate_period(laser_on, max_period=max_period)
    on_shots = np.flatnonzero(laser_on)
    first_on = on_shots[0] if on_shots.size else n_shots
    if period is not None:
        for i in range(first_on - 1, -1, -1):
            j = i + period
            if j < n_shots and not laser_on[j] and off_type[j] is not None:
                off_type[i] = off_type[j]

    return [
        unknown_label
        if ot is None else f'{ot:03d}_on' if ot == 0 else f'{ot:03d}_off'
        for ot in off_type
    ]


def save_array_to_movie(ims: np.ndarray,
                        path: Optional[PathLike] = None,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        cmap: str = 'viridis',
                        first_frame_repeats: int = 15) -> np.ndarray:
    if vmin is None:
        vmin = np.nanmin(ims)
    if vmax is None:
        vmax = np.nanmax(ims)
    ims = np.clip(ims, vmin, vmax)
    regulated_ims = (ims - vmin) / (vmax - vmin)
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
        imageio.mimsave(path, repeated_ims, duration=1000 / 30)
    return regulated_ims
