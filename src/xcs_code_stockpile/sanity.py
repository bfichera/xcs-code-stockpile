import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def intensity_vs_i0(metadata: pd.DataFrame):
    """Plot the intensity in an roi vs i0.

    Does no filtering - drop rows from `metadata` first if you want"""
    i0 = metadata['i0'].to_numpy()
    i = metadata['intensity_in_roi'].to_numpy()

    model = LinearRegression()
    X = i0[:, None]
    y = i
    model.fit(X, y)

    out = pd.DataFrame({'i0': i0, 'intensity': i})
    out.attrs['beta'] = model.coef_[0]
    out.attrs['intercept'] = model.intercept_
    out.attrs['R^2'] = model.score(X, y)

    return out


def plot_intensity_vs_i0(metadata: pd.DataFrame):
    """Plot the intensity in an roi vs i0.

    Does no filtering - drop rows from `metadata` first if you want"""
    d = intensity_vs_i0(metadata)
    i = d['intensity']
    i0 = d['i0']
    beta = d.attrs['beta']
    intercept = d.attrs['intercept']
    r2 = d.attrs['R^2']

    fig, ax = plt.subplots()
    ax.scatter(i0, i, edgecolors='black', marker='o', facecolors='none')
    model_x = np.linspace(i0.min(), i0.max(), 100)
    ax.plot(model_x,
            beta * model_x + intercept,
            linestyle=':',
            color='red',
            label=fr'$R^2 = {r2:.2f}$')
    ax.set_title(f'Run {metadata.attrs["run"]:04d}')
    ax.legend()

    return fig


def plot_i0_vs_time(metadata: pd.DataFrame):
    """Plot i0 vs time.

    Does no filtering - drop rows from `metadata` first if you want"""
    i0 = metadata['i0'].to_numpy()
    time = metadata['event_time'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(time, i0, c='black')
    ax.set_title(f'Run {metadata.attrs["run"]:04d}')

    return fig


def plot_intensity_vs_time(metadata: pd.DataFrame):
    """Plot intensity vs time.

    Does no filtering - drop rows from `metadata` first if you want"""
    i = metadata['intensity_in_roi'].to_numpy()
    time = metadata['event_time'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(time, i, c='black')
    ax.set_title(f'Run {metadata.attrs["run"]:04d}')

    return fig
