from functools import wraps
from datetime import datetime

import numpy as np

from .roi import _ROI


def parsed_roi(f):
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
def slow_mean(h5, h5dict, shots, roi=None):
    stack = h5[h5dict.stack]
    shape = h5[h5dict.stack][0, *roi].shape
    out = np.zeros(shape)
    n = len(shots)
    for i in range(h5[h5dict.stack].shape[0]):
        if i in shots:
            out += stack[i, *roi] / n
    return out


@np.vectorize
def ts_to_s(ts):
    sec = ts >> 32 & 0xffffffff
    nsec = ts & 0xffffffff
    return datetime.fromtimestamp(sec + nsec * 1e-9).timestamp()


def basic_off_type_encoder(laser_on):
    laser_on = laser_on.astype(int)
    n_shots = len(laser_on)
    found_first_laser_on_shot = False
    off_type = [np.nan for i in range(n_shots)]
    for i in range(n_shots):
        if laser_on[i]:
            found_first_laser_on_shot = True
            off_type[i] = 0
        else:
            if found_first_laser_on_shot:
                off_type[i] = off_type[i - 1] + 1
            else:
                off_type[i] = np.nan
    for i in range(n_shots):
        if np.isnan(off_type[i]):
            next_laser_on_position = 0
            next_next_laser_on_position = 0
            for j in range(n_shots):
                next_laser_on_position += 1
                next_next_laser_on_position += 1
                if off_type[i + j] == 0:
                    for k in range(n_shots):
                        if k == 0:
                            continue
                        next_next_laser_on_position += 1
                        if off_type[i + j + k] == 0:
                            break
                    break
            off_type[i] = off_type[i + (next_next_laser_on_position -
                                        next_laser_on_position)]
    off_type_str = []
    for ot in off_type:
        if ot == 0:
            off_type_str.append(f'{ot:02d}_on')
        else:
            off_type_str.append(f'{ot:02d}_off')
    return off_type_str
