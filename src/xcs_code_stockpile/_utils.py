from .roi import _ROI


def parsed_roi(f):

    def g(*args, **kwargs):
        if 'roi' not in kwargs:
            raise TypeError('kwargs does not include \'roi\'')
        if kwargs['roi'] is None:
            kwargs['roi'] = _ROI.from_slices(slice(None), slice(None))
        kwargs['roi'] = _ROI._guess(kwargs['roi'])
        return f(*args, **kwargs)

    return g
