"""Correlation and diffraction analysis tools for XCS smalldata."""

from .datahandler import H5Dictionary, load_smalldata
from .metadata import create_metadata
from .filter import sensible_filter, no_filter, i0_min_max_filter
from .differential_correlation import (differential_correlation,
                                       differential_twotime)
from .trxrd import calc_trxrd_stats, save_movie
from .static_correlation import calc_static_correlation
from .rocking import calc_rocking_stats
from .sanity import (plot_i0_vs_time, plot_intensity_vs_i0,
                     plot_intensity_vs_time, plot_detector_image,
                     plot_roi_image)
from ._utils import basic_off_type_encoder

__all__ = [
    'H5Dictionary', 'load_smalldata', 'create_metadata', 'sensible_filter',
    'no_filter', 'i0_min_max_filter', 'differential_correlation',
    'differential_twotime', 'calc_trxrd_stats', 'save_movie',
    'calc_static_correlation', 'calc_rocking_stats', 'plot_intensity_vs_i0',
    'plot_i0_vs_time', 'plot_intensity_vs_time', 'plot_detector_image',
    'basic_off_type_encoder', 'plot_roi_image'
]
