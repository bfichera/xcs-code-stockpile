from typing import Any, Optional, Union

import pandas as pd

import h5py
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
import numpy as np

from .datahandler import H5Dictionary
from ._utils import slow_mean, parsed_roi


def fit_linear_i_vs_i0(
    metadata: pd.DataFrame,
    filter: Optional[Union[np.ndarray, pd.Series]] = None,
) -> tuple[float, float, float]:
    """Fit ROI intensity as a linear function of ``i0``.

    Args:
        metadata: Metadata containing ``i0`` and ``intensity_in_roi``.
        filter: Optional boolean array or Series selecting metadata rows.
            `None` uses all rows.

    Returns:
        A tuple ``(beta, intercept, r2)`` of the slope, intercept and R^2
            of the fit.
    """
    if filter is not None:
        metadata = metadata[filter]
    i0 = metadata['i0'].to_numpy()
    i = metadata['intensity_in_roi'].to_numpy()

    model = LinearRegression()
    X = i0[:, None]
    y = i
    model.fit(X, y)

    return model.coef_[0], model.intercept_, model.score(X, y)


def plot_intensity_vs_i0(
    metadata: pd.DataFrame,
    filter: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Figure:
    """Plot ROI intensity against ``i0``, with a linear fit.

    The fit uses only shots passing `filter`. All shots are plotted, and
    those that fail the filter are outlined in red.

    Args:
        metadata: Metadata containing ``i0`` and ``intensity_in_roi``.
        filter: Optional boolean array or Series selecting shots for the fit.
            `None` uses all rows.

    Returns:
        A new Matplotlib Figure.
    """
    beta, intercept, r2 = fit_linear_i_vs_i0(metadata, filter)
    metadata = metadata.copy()
    if filter is not None:
        metadata['valid'] = filter
    else:
        metadata['valid'] = [True for i in range(metadata.shape[0])]
    i = metadata['intensity_in_roi'].to_numpy()
    i0 = metadata['i0'].to_numpy()

    fig, ax = plt.subplots()
    edgecolors = [
        'black' if metadata['valid'].to_numpy()[i] else 'red'
        for i in range(len(i0))
    ]
    ax.scatter(i0, i, edgecolors=edgecolors, marker='o', facecolors='none')
    model_x = np.linspace(i0.min(), i0.max(), 100)
    ax.plot(model_x,
            beta * model_x + intercept,
            linestyle=':',
            color='red',
            label=fr'$R^2 = {r2:.2f}$')
    ax.legend()
    ax.set_xlabel('i0')
    ax.set_ylabel('intensity')

    return fig


def plot_i0_vs_time(
    metadata: pd.DataFrame,
    filter: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Figure:
    """Plot incident intensity against event time.

    Args:
        metadata: Metadata containing ``i0`` and ``event_time``, with a run
            number in ``metadata.attrs["run"]``.
        filter: Optional boolean array or Series selecting metadata rows.
            `None` uses all rows.

    Returns:
        A new Matplotlib Figure.
    """
    if filter is not None:
        metadata = metadata[filter]
    i0 = metadata['i0'].to_numpy()
    time = metadata['event_time'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(time, i0, c='black')
    ax.set_title(f'Run {metadata.attrs["run"]:04d}')

    ax.set_xlabel('event_time')
    ax.set_ylabel('i0')

    return fig


def plot_intensity_vs_time(
    metadata: pd.DataFrame,
    filter: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Figure:
    """Plot integrated ROI intensity against event time.

    Args:
        metadata: Metadata containing ``intensity_in_roi`` and ``event_time``,
            with a run number in ``metadata.attrs["run"]``.
        filter: Optional boolean array or Series selecting metadata rows.
            `None` uses all rows.

    Returns:
        A new Matplotlib Figure.
    """
    if filter is not None:
        metadata = metadata[filter]
    i = metadata['intensity_in_roi'].to_numpy()
    time = metadata['event_time'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(time, i, c='black')
    ax.set_title(f'Run {metadata.attrs["run"]:04d}')
    ax.set_xlabel('event_time')
    ax.set_ylabel('intensity')

    return fig


def plot_detector_image(h5: h5py.File, h5dict: H5Dictionary, samplerate: int,
                        **kws: Any) -> Figure:
    """Plot the mean full detector image.

    Args:
        h5: Open HDF5 file containing the detector stack.
        h5dict: Mapping whose ``stack`` field identifies the detector dataset.
        samplerate: Average every Nth shot.
        **kws: Passed to `matplotlib.pyplot.imshow`. ``vmin`` and ``vmax``
            default to the 1st and 99th percentiles.

    Returns:
        A new Matplotlib Figure.
    """
    n_shots = h5[h5dict.stack].shape[0]
    im = slow_mean(h5, h5dict, np.arange(n_shots)[::samplerate], roi=None)
    if 'vmin' not in kws:
        kws['vmin'] = np.percentile(im, 1)
    if 'vmax' not in kws:
        kws['vmax'] = np.percentile(im, 99)
    fig, ax = plt.subplots()
    ax.imshow(im, **kws)
    return fig


@parsed_roi
def plot_roi_image(
    h5: h5py.File,
    h5dict: H5Dictionary,
    metadata: pd.DataFrame,
    filter: Optional[Union[np.ndarray, pd.Series]] = None,
    roi: Optional[Union[list[int], tuple[slice, slice]]] = None,
    **kws: Any,
) -> Figure:
    """Plot the mean image of an ROI over the selected shots.

    Args:
        h5: Open HDF5 file containing the detector stack.
        h5dict: Mapping whose ``stack`` field identifies the detector dataset.
        metadata: Per-shot metadata containing ``shot_idx``.
        filter: Optional boolean array or Series selecting metadata rows.
            `None` uses all rows.
        roi: Detector region as ``[x0, x1, y0, y1]`` or a
            ``(row_slice, column_slice)`` tuple. `None` selects the full frame.
        **kws: Passed to `matplotlib.pyplot.imshow`. ``vmin`` and ``vmax``
            default to the 1st and 99th percentiles.

    Returns:
        A new Matplotlib Figure.
    """
    if filter is not None:
        metadata = metadata[filter].copy()
    valid_shots = metadata['shot_idx'].to_numpy()
    im = slow_mean(h5, h5dict, valid_shots, roi=roi)
    if 'vmin' not in kws:
        kws['vmin'] = np.percentile(im, 1)
    if 'vmax' not in kws:
        kws['vmax'] = np.percentile(im, 99)
    fig, ax = plt.subplots()
    ax.imshow(im, **kws)
    return fig
