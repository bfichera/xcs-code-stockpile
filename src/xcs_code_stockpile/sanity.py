from typing import Optional, Union

import pandas as pd

import h5py
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
import numpy as np

from .datahandler import H5Dictionary


def fit_linear_i_vs_i0(
    metadata: pd.DataFrame,
    filter: Optional[Union[np.ndarray, pd.Series]] = None,
) -> pd.DataFrame:
    """Fit pump-off ROI intensity as a linear function of incident intensity.

    Args:
        metadata: Metadata containing ``i0``, and
            ``intensity_in_roi``.
        filter: Optional boolean array or Series selecting metadata rows before
            pump-off shots are selected. `None` uses all rows.

    Returns:
        beta, intercept, R^2
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
    """Plot pump-off ROI intensity and its linear fit against ``i0``.

    Args:
        metadata: Metadata accepted by `intensity_vs_i0`, with a run number in
            ``metadata.attrs["run"]``.
        filter: Optional boolean array or Series selecting metadata rows before
            pump-off shots are selected. `None` uses all rows.

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
    edgecolors = ['black' if metadata['valid'].to_numpy()[i] else 'red' for i in range(len(i0))]
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


def plot_detector_image(
    h5: h5py.File,
    h5dict: H5Dictionary,
    samplerate: int
) -> Figure:
    """Plot the full detector.

    Args:
        h5: h5py.File
        h5dict: H5Dictionary
        samplerate: int

    Returns:
        A new Matplotlib Figure.
    """
    im = np.mean(h5[h5dict.stack][::samplerate, ...], axis=0)
    vmin = np.percentile(im, 1)
    vmax = np.percentile(im, 99)
    fig, ax = plt.subplots()
    ax.imshow(im, vmin=vmin, vmax=vmax)
    return fig
