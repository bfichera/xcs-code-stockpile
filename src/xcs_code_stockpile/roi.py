from typing import Optional
from numpy.typing import ArrayLike


class _ROI:

    def __init__(
        self,
        i0: Optional[int],
        i1: Optional[int],
        istep: Optional[int],
        j0: Optional[int],
        j1: Optional[int],
        jstep: Optional[int],
    ):
        self._i0 = i0
        self._i1 = i1
        self._istep = istep
        self._j0 = j0
        self._j1 = j1
        self._jstep = jstep

    def __repr__(self):
        return (f"row minimum (y_min): {self._i0}\n"
                f"row maximum (y_max): {self._i1}\n"
                f"row step   (y_step): {self._istep}\n"
                f"col minimum (x_min): {self._j0}\n"
                f"col maximum (x_max): {self._j1}\n"
                f"col step   (x_step): {self._jstep}\n")

    @property
    def _as_slices(self):
        return (
            slice(self._i0, self._i1, self._istep),
            slice(self._j0, self._j1, self._jstep),
        )

    @property
    def _as_list(self):
        return [self._j0, self._j1, self._i0, self._i1]

    def __iter__(self):
        return iter(self._as_slices)

    @classmethod
    def from_array(cls, a: ArrayLike) -> '_ROI':
        """Returns an ROI object from an array

        Args:
            a: ArrayLike of length 4 (x0, x1, y0, y1)

        Returns:
            [ROI object](`.ROI`)
        """

        if len(a) != 4:
            raise ValueError('Array ROIs should be of length 4')

        x0, x1, y0, y1 = a

        return cls(y0, y1, None, x0, x1, None)

    @classmethod
    def from_slices(cls, i_sl: slice, j_sl: slice) -> '_ROI':
        """Returns an ROI object from two slice objects

        Args:
            i_sl: Row slice
            j_sl: Col slice

        Returns:
            [ROI object](`.ROI`)
        """
        return cls(i_sl.start, i_sl.stop, i_sl.step, j_sl.start, j_sl.stop,
                   j_sl.step)

    @classmethod
    def _guess(cls, arg) -> '_ROI':
        if isinstance(arg, cls):
            return arg
        if len(arg) == 4:
            return cls.from_array(arg)
        if len(arg) == 2:
            return cls.from_slices(*arg)
        raise TypeError('Couldn\'t determine the ROI type?')
