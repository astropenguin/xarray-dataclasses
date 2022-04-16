# type: ignore
__version__ = "1.1.0"


# submodules
from . import dataarray
from . import dataset
from . import datamodel
from . import dataoptions
from . import typing


# aliases
from .dataarray import *
from .dataset import *
from .datamodel import *
from .dataoptions import *
from .typing import *


# for Sphinx docs
__all__ = dir()
