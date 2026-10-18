import h5py
import numpy as np
import pandas as pd

from ._utils import parsed_roi
from .datahandler import H5Dictionary


def _correlation_of_two_images(im1: np.ndarray,
                               im2: np.ndarray,
                               denominator_type: str = 'mean'):
    if denominator_type == 'mean':
        return np.mean(im1 * im2) / np.mean(im1) / np.mean(im2)
    elif denominator_type == 'std':
        return np.mean(im1 * im2) / np.std(im1) / np.std(im2)
    raise NotImplementedError(
        f'Denominator type {denominator_type} not implemented')


@parsed_roi
def _strong_reversibility_diffcorr(h5: h5py.File,
                                   h5dict: H5Dictionary,
                                   metadata: pd.DataFrame,
                                   filter: np.ndarray,
                                   delay_bin_interval: float,
                                   labtime_bin_interval: float,
                                   roi: list[int, int, int, int]
                                   | tuple[slice, slice] | None = None,
                                   denominator_type: str = 'mean'):
    delay = metadata['delay'].to_numpy()
    _bins = np.arange(delay.min(),
                      delay.max() + delay_bin_interval / 2, delay_bin_interval)
    delay_bins = pd.cut(metadata['delay'],
                        _bins,
                        right=False,
                        labels=_bins[:-1])
    metadata['delay_bin'] = delay_bins

    labtime = metadata['event_time'].to_numpy()
    _bins = np.arange(labtime.min(),
                      labtime.max() + labtime_bin_interval / 2,
                      labtime_bin_interval)
    labtime_bins = pd.cut(metadata['event_time'],
                          _bins,
                          right=False,
                          labels=_bins[:-1])
    metadata['labtime_bin'] = labtime_bins

    stack = h5[h5dict.stack][..., *roi]

    # For each labtime bin, we form a diffcorr
    # then we combine the diffcorrs by averaging

    def diffcorr_of_labtime_bin(group):
        off_shots_idx = group.loc[~group['laser_on'], 'shot_idx'].to_numpy()
        off_shots = stack[off_shots_idx, ...]
        mean_off_shot = np.mean(off_shots, axis=0)

        def diffcorr_of_delay_bin(dgroup):
            on_shots_idx = dgroup.loc[dgroup['laser_on'], 'shot_idx'].to_numpy()
            on_shots = stack[on_shots_idx, ...]
            mean_on_shot = np.mean(on_shots, axis=0)
            return _correlation_of_two_images(
                mean_on_shot, mean_off_shot, denominator_type=denominator_type)

        return group.groupby('delay_bin').apply(diffcorr_of_delay_bin)

    diffcorrs = metadata.groupby('labtime_bin').apply(
        diffcorr_of_labtime_bin).reset_index()
    mean_diffcorr = diffcorrs.groupby('delay_bin')[0].agg('mean')

    return mean_diffcorr


@parsed_roi
def _weak_reversibility_diffcorr(h5: h5py.File,
                                 h5dict: H5Dictionary,
                                 metadata: pd.DataFrame,
                                 filter: np.ndarray,
                                 delay_bin_interval: float,
                                 labtime_bin_interval: float,
                                 roi: list[int, int, int, int]
                                 | tuple[slice, slice] | None = None,
                                 denominator_type: str = 'mean'):
    delay = metadata['delay'].to_numpy()
    _bins = np.arange(delay.min(),
                      delay.max() + delay_bin_interval / 2, delay_bin_interval)
    delay_bins = pd.cut(metadata['delay'],
                        _bins,
                        right=False,
                        labels=_bins[:-1])
    metadata['delay_bin'] = delay_bins

    labtime = metadata['event_time'].to_numpy()
    _bins = np.arange(labtime.min(),
                      labtime.max() + labtime_bin_interval / 2,
                      labtime_bin_interval)
    labtime_bins = pd.cut(metadata['event_time'],
                          _bins,
                          right=False,
                          labels=_bins[:-1])
    metadata['labtime_bin'] = labtime_bins

    stack = h5[h5dict.stack][..., *roi]

    # For each labtime bin, we form a diffcorr
    # then we combine the diffcorrs by averaging

    def diffcorr_of_labtime_bin(group):
        off_shots_idx = group[~group['laser_on']].index
        off_shots = stack[off_shots_idx, ...]
        mean_off_shot = np.mean(off_shots, axis=0)

        def diffcorr_of_shot(row):
            im = stack[row['shot_idx'], ...]
            return _correlation_of_two_images(
                im, mean_off_shot, denominator_type=denominator_type)

        return group[group['laser_on']].apply(
            diffcorr_of_shot, axis=1).groupby(group['delay_bin']).agg('mean')

    diffcorrs = metadata.groupby('labtime_bin').apply(
        diffcorr_of_labtime_bin).reset_index()
    mean_diffcorr = diffcorrs.groupby('delay_bin')[0].agg('mean')

    return mean_diffcorr


def _none_reversibility_diffcorr(h5: h5py.File,
                                 h5dict: H5Dictionary,
                                 metadata: pd.DataFrame,
                                 filter: np.ndarray,
                                 delay_bin_interval: float,
                                 roi: list[int, int, int, int]
                                 | tuple[slice, slice] | None = None,
                                 denominator_type: str = 'mean'):
    delay = metadata['delay'].to_numpy()
    _bins = np.arange(delay.min(),
                      delay.max() + delay_bin_interval / 2, delay_bin_interval)
    delay_bins = pd.cut(metadata['delay'],
                        _bins,
                        right=False,
                        labels=_bins[:-1])
    metadata['delay_bin'] = delay_bins

    stack = h5[h5dict.stack][..., *roi]

    # For each on shot, we correlate it with the off shot that came before.
    # Then we bin and average.

    on_shot_idx = metadata[metadata['laser_on']]
    off_shot_idx = metadata[~metadata['laser_on']]
    delays = []
    corrs = []
    for shot_idx, delay_bin in zip(on_shot_idx,
                                   metadata['delay_bin'].iloc[on_shot_idx]):
        if shot_idx == 0:
            continue
        on_shot = stack[shot_idx]
        laser_off_shot_idx = max(off_shot_idx[off_shot_idx < shot_idx])
        off_shot = stack[laser_off_shot_idx]
        delays.append(delay)
        corrs.append(
            _correlation_of_two_images(on_shot,
                                       off_shot,
                                       denominator_type=denominator_type))

    diffcorrs = pd.DataFrame({'delay_bin': delays, 0: corrs})
    mean_diffcorr = diffcorrs.groupby('delay_bin')[0].agg('mean')

    return mean_diffcorr


def _correlation_dispatcher(reversibility_type: str):
    if reversibility_type == 'strong':
        return _strong_reversibility_diffcorr
    elif reversibility_type == 'weak':
        return _weak_reversibility_diffcorr
    elif reversibility_type == 'none':
        return _none_reversibility_diffcorr
    raise NotImplementedError(
        f'Reversibility type {reversibility_type} not implemented')


def differential_correlation(reversibility_type: str, *args, **kwargs):
    _diffcorr_func = _correlation_dispatcher(reversibility_type)
    return _diffcorr_func(*args, **kwargs)
