"""xcs-code-stockpile."""
from .datahandler import H5Dictionary, load_smalldata
from .metadata import create_metadata
from .filter import sensible_filter
from .differential_correlation import differential_correlation
from .trxrd import calc_trxrd_status
from .static_correlation import calc_static_correlation
from .rocking import calc_rocking_stats
