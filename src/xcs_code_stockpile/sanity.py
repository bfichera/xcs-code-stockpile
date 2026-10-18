import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


def plot_intensity_vs_i0(metadata: pd.DataFrame):
    """Plot the intensity in an roi vs i0.

    Does no filtering - drop rows from `metadata` first if you want"""
    i0 = metadata['i0'].to_numpy()
    i = metadata['intensity_in_roi'].to_numpy()

    model = LinearRegression()
    X = i0[:, None]
    y = i
    model.fit(X, y)

    fig, ax = plt.subplots()
    ax.scatter(i0, i, c='black', marker='o', fc='none')
    model_x = np.linspace(i0.min(), i0.max(), 100)
    ax.plot(model_x,
            model.coef_[0] * model_x + model.intercept_,
            linestyle=':',
            color='red',
            label=fr'$R^2 = {model.score(X, y):.2f}$')
    ax.set_title(f'Run {metadata["run"].unique()[0]:04d}')
    ax.legend()

    return fig


def plot_i0_vs_time(metadata: pd.DataFrame):
    """Plot i0 vs time.

    Does no filtering - drop rows from `metadata` first if you want"""
    i0 = metadata['i0'].to_numpy()
    time = metadata['event_time'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(time, i0, c='black')
    ax.set_title(f'Run {metadata["run"].unique()[0]:04d}')

    return fig


def plot_intensity_vs_time(metadata: pd.DataFrame):
    """Plot intensity vs time.

    Does no filtering - drop rows from `metadata` first if you want"""
    i = metadata['intensity_in_roi'].to_numpy()
    time = metadata['event_time'].to_numpy()

    fig, ax = plt.subplots()
    ax.plot(time, i, c='black')
    ax.set_title(f'Run {metadata["run"].unique()[0]:04d}')

    return fig
