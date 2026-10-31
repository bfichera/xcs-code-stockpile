from typing import Any, Optional, Union

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from ._utils import parsed_roi, _bin_edges, _bin_labels
from .datahandler import H5Dictionary


def _slow_mean_of_ims(stack, idx):
    out = np.zeros(stack.shape[1:])
    n = 0
    for i in idx:
        n += 1
        out += stack[i, ...]
    return out / n


def _validate_metadata(metadata):
    if metadata['laser_off_type'].nunique() > 2:
        raise NotImplementedError(
            'Differential correlation is only implemented for two off types'
            ' (laser on, and laser off)')


def _correlation_of_two_images(im1: np.ndarray,
                               im2: np.ndarray,
                               denominator_type: str = 'mean'):
    # TODO
    # This also overflows sometimes?
    if denominator_type.startswith('nan'):
        mean_ = np.nanmean
        std_ = np.nanstd
    else:
        mean_ = np.mean
        std_ = np.std
    if denominator_type.endswith('mean'):
        return mean_(im1 * im2) / mean_(im1) / mean_(im2) - 1
    elif denominator_type.endswith('std'):
        covariance = mean_(im1 * im2) - mean_(im1) * mean_(im2)
        return covariance / std_(im1) / std_(im2)
    raise NotImplementedError(
        f'Denominator type {denominator_type} not implemented')


@parsed_roi
def _strong_reversibility_diffcorr(h5: h5py.File,
                                   h5dict: H5Dictionary,
                                   metadata: pd.DataFrame,
                                   filter: np.ndarray,
                                   delay_bin_interval: float,
                                   labtime_bin_interval: float,
                                   roi: Optional[Union[list[int],
                                                       tuple[slice,
                                                             slice]]] = None,
                                   denominator_type: str = 'mean'):

    metadata = metadata[filter].copy()

    _validate_metadata(metadata)

    delay = metadata['delay'].to_numpy()
    _bins = _bin_edges(delay, delay_bin_interval)
    delay_bins = pd.cut(metadata['delay'],
                        _bins,
                        right=False,
                        labels=_bin_labels(_bins))
    metadata['delay_bin'] = delay_bins

    labtime = metadata['event_time'].to_numpy()
    _bins = _bin_edges(labtime, labtime_bin_interval)
    labtime_bins = pd.cut(metadata['event_time'],
                          _bins,
                          right=False,
                          labels=_bin_labels(_bins))
    metadata['labtime_bin'] = labtime_bins
    metadata = metadata.groupby('labtime_bin').filter(
        lambda group: group['laser_on'].nunique() == 2)

    stack = h5[h5dict.stack][..., *roi]

    # For each labtime bin, we form a diffcorr
    # then we combine the diffcorrs by averaging

    def diffcorr_of_labtime_bin(group):
        off_shots_idx = group.loc[~group['laser_on'], 'shot_idx'].to_numpy()
        off_shots = stack[off_shots_idx, ...]
        mean_off_shot = np.mean(off_shots, axis=0)

        def diffcorr_of_delay_bin(dgroup):
            on_shots_idx = dgroup.loc[dgroup['laser_on'],
                                      'shot_idx'].to_numpy()
            on_shots = stack[on_shots_idx, ...]
            mean_on_shot = np.mean(on_shots, axis=0)
            return _correlation_of_two_images(
                mean_on_shot, mean_off_shot, denominator_type=denominator_type)

        return group[group['laser_on']].groupby('delay_bin').apply(
            diffcorr_of_delay_bin)

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
                                 roi: Optional[Union[list[int],
                                                     tuple[slice,
                                                           slice]]] = None,
                                 denominator_type: str = 'mean'):

    metadata = metadata[filter].copy()

    _validate_metadata(metadata)

    delay = metadata['delay'].to_numpy()
    _bins = _bin_edges(delay, delay_bin_interval)
    delay_bins = pd.cut(metadata['delay'],
                        _bins,
                        right=False,
                        labels=_bin_labels(_bins))
    metadata['delay_bin'] = delay_bins

    labtime = metadata['event_time'].to_numpy()
    _bins = _bin_edges(labtime, labtime_bin_interval)
    labtime_bins = pd.cut(metadata['event_time'],
                          _bins,
                          right=False,
                          labels=_bin_labels(_bins))
    metadata['labtime_bin'] = labtime_bins
    metadata = metadata.groupby('labtime_bin').filter(
        lambda group: group['laser_on'].nunique() == 2)

    stack = h5[h5dict.stack][..., *roi]

    # For each labtime bin, we form a diffcorr
    # then we combine the diffcorrs by averaging

    def diffcorr_of_labtime_bin(group):
        off_shots_idx = group.loc[~group['laser_on'], 'shot_idx'].to_numpy()
        off_shots = stack[off_shots_idx, ...]
        mean_off_shot = np.mean(off_shots, axis=0)

        def diffcorr_of_shot(row):
            im = stack[row['shot_idx'], ...]
            return _correlation_of_two_images(
                im, mean_off_shot, denominator_type=denominator_type)

        on_shots = group[group['laser_on']]
        return on_shots.apply(diffcorr_of_shot, axis=1).groupby(
            on_shots['delay_bin']).agg('mean')

    diffcorrs = metadata.groupby('labtime_bin').apply(
        diffcorr_of_labtime_bin).reset_index()
    mean_diffcorr = diffcorrs.groupby('delay_bin')[0].agg('mean')

    return mean_diffcorr


@parsed_roi
def _none_reversibility_diffcorr(h5: h5py.File,
                                 h5dict: H5Dictionary,
                                 metadata: pd.DataFrame,
                                 filter: np.ndarray,
                                 delay_bin_interval: float,
                                 roi: Optional[Union[list[int],
                                                     tuple[slice,
                                                           slice]]] = None,
                                 denominator_type: str = 'mean'):

    metadata = metadata[filter].copy()

    _validate_metadata(metadata)

    delay = metadata['delay'].to_numpy()
    _bins = _bin_edges(delay, delay_bin_interval)
    delay_bins = pd.cut(metadata['delay'],
                        _bins,
                        right=False,
                        labels=_bin_labels(_bins))
    metadata['delay_bin'] = delay_bins

    stack = h5[h5dict.stack][..., *roi]

    # For each on shot, we correlate it with the off shot that came before.
    # Then we bin and average.

    on_shots = metadata.loc[metadata['laser_on'], ['shot_idx', 'delay_bin']]
    off_shot_idx = metadata.loc[~metadata['laser_on'], 'shot_idx'].to_numpy()
    delays = []
    corrs = []
    for shot_idx, delay_bin in on_shots.itertuples(index=False, name=None):
        preceding_off_shots = off_shot_idx[off_shot_idx < shot_idx]
        if preceding_off_shots.size == 0:
            continue
        on_shot = stack[shot_idx]
        off_shot = stack[preceding_off_shots.max()]
        delays.append(delay_bin)
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


def _twotime_dispatcher(reversibility_type: str):
    if reversibility_type == 'strong':
        return _strong_reversibility_difftwotime
    elif reversibility_type == 'weak':
        pass
    elif reversibility_type == 'none':
        pass
    raise NotImplementedError(
        f'Reversibility type {reversibility_type} not implemented')


def differential_correlation(reversibility_type: str, *args: Any,
                             **kwargs: Any) -> pd.Series:
    """Calculate pump-on/pump-off correlation grouped by delay.

    Args:
        reversibility_type: Correlation strategy: ``"strong"``, ``"weak"``,
            or ``"none"``.
        *args: Positional arguments forwarded to the selected strategy. All
            strategies accept ``h5, h5dict, metadata, filter,
            delay_bin_interval``. Strong and weak strategies additionally
            require ``labtime_bin_interval``.
        **kwargs: Forwarded analysis options, including ``roi`` and
            ``denominator_type``. ``roi`` accepts ``[x0, x1, y0, y1]`` or a
            ``(row_slice, column_slice)`` tuple. ``denominator_type`` accepts
            ``"mean"``, ``"std"``, ``"nanmean"`` or ``"nanstd"``.

    Returns:
        A Series containing mean correlations, indexed by the left edge of
            each delay bin. Strong and weak strategies compare within lab-time
            bins; the no-reversibility strategy compares each pump-on shot with
            its most recent preceding pump-off shot.

    Raises:
        NotImplementedError: If a reversibility or denominator type is not
            supported, or if the filtered metadata has more than two
            ``laser_off_type`` values.
    """
    _diffcorr_func = _correlation_dispatcher(reversibility_type)
    return _diffcorr_func(*args, **kwargs)


@parsed_roi
def _strong_reversibility_difftwotime(
        h5: h5py.File,
        h5dict: H5Dictionary,
        metadata: pd.DataFrame,
        filter: np.ndarray,
        delay_bin_interval: float,
        labtime_bin_interval: float,
        roi: Optional[Union[list[int], tuple[slice, slice]]] = None,
        denominator_type: str = 'mean'):

    metadata = metadata[filter].copy()

    delay = metadata['delay'].to_numpy()
    _bins = _bin_edges(delay, delay_bin_interval)
    delay_bins = pd.cut(metadata['delay'],
                        _bins,
                        right=False,
                        labels=_bin_labels(_bins))
    metadata['delay_bin'] = delay_bins

    labtime = metadata['event_time'].to_numpy()
    _bins = _bin_edges(labtime, labtime_bin_interval)
    labtime_bins = pd.cut(metadata['event_time'],
                          _bins,
                          right=False,
                          labels=_bin_labels(_bins))
    metadata['labtime_bin'] = labtime_bins
    metadata = metadata.groupby('labtime_bin').filter(
        lambda group: group['laser_on'].nunique() == 2)

    stack = h5[h5dict.stack][..., *roi]

    # For each labtime bin, we form a twotime
    # Then we average the twotimes

    # TODO
    # A better estimate should be used which takes into account the number of
    # frames we mean over but I don't have time to do it right now
    n_delays = metadata['delay_bin'].nunique()
    idx = np.arange(n_delays, dtype=int)
    map_ = {
        ud: i
        for ud, i in zip(sorted(metadata['delay_bin'].unique()), idx)
    }
    with tqdm(total=metadata['labtime_bin'].nunique()) as bar:
        twotimes = []
        for lab_idx, (labtime_bin, labtime_group) in enumerate(
                metadata.groupby('labtime_bin')):
            labtime_twotime = np.full((n_delays, n_delays), np.nan)
            for i, (delay1,
                    group1) in enumerate(labtime_group.groupby('delay_bin')):
                for j, (delay2, group2) in enumerate(
                        labtime_group.groupby('delay_bin')):
                    if delay1 <= delay2:  # delay1 is always > delay2
                        continue
                    i_towrite = map_[delay1]
                    j_towrite = map_[delay2]
                    on_shots_idx1 = group1.loc[group1['laser_on'],
                                               'shot_idx'].to_numpy()
                    on_shots_idx2 = group2.loc[group2['laser_on'],
                                               'shot_idx'].to_numpy()
                    if on_shots_idx1.size * on_shots_idx2.size == 0:
                        continue
                    mean_on_shot1 = _slow_mean_of_ims(stack, on_shots_idx1)
                    mean_on_shot2 = _slow_mean_of_ims(stack, on_shots_idx2)
                    labtime_twotime[i_towrite,
                                    j_towrite] = _correlation_of_two_images(
                                        mean_on_shot1,
                                        mean_on_shot2,
                                        denominator_type=denominator_type)
            bar.update()
            twotimes.append(labtime_twotime)
        twotime = np.nanmean(twotimes, axis=0)
    return np.where(np.isnan(twotime), twotime.T, twotime)


def differential_twotime(reversibility_type: str, *args: Any,
                         **kwargs: Any) -> np.ndarray:
    """Calculate the two-time version of `differential_correlation`.

    Args:
        reversibility_type: Only ``"strong"`` is implemented.
        *args: Passed to the strategy, as in `differential_correlation`.
        **kwargs: Passed to the strategy, as in `differential_correlation`.

    Returns:
        A symmetric array with one row and column per delay bin. Entries are
            NaN on the diagonal and where two delay bins never share a lab-time
            bin.

    Raises:
        NotImplementedError: If `reversibility_type` is not supported.
    """
    _difftwotime_func = _twotime_dispatcher(reversibility_type)
    return _difftwotime_func(*args, **kwargs)
