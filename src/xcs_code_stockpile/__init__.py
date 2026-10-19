"""Correlation and diffraction analysis tools for XCS smalldata."""

from .datahandler import H5Dictionary, load_smalldata
from .metadata import create_metadata
from .filter import sensible_filter
from .differential_correlation import differential_correlation
from .trxrd import calc_trxrd_stats
from .static_correlation import calc_static_correlation
from .rocking import calc_rocking_stats
from .sanity import (intensity_vs_i0, plot_i0_vs_time,
                     plot_intensity_vs_i0, plot_intensity_vs_time)

__all__ = [
    'H5Dictionary',
    'load_smalldata',
    'create_metadata',
    'sensible_filter',
    'differential_correlation',
    'calc_trxrd_stats',
    'calc_static_correlation',
    'calc_rocking_stats',
    'intensity_vs_i0',
    'plot_intensity_vs_i0',
    'plot_i0_vs_time',
    'plot_intensity_vs_time',
]
