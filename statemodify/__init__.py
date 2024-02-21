"""This package provides functionalities to modify StateMod files.

It includes modules for batch processing, demand data modification, evaporation data modification,
and more, allowing for the customization and generation of StateMod data files based on user-defined
parameters and scenarios.
"""

from .batch import *
from .ddm import *
from .ddr import *
from .eva import *
from .hmm import *
from .modify import *
from .res import *
from .sampler import *
from .utils import *
from .xbm_iwr import *
from .xdd import *
from .xre import *

__version__ = "0.1.4"
