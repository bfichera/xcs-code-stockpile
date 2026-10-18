from functools import wraps

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
