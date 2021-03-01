# flake8: noqa
__author__ = "Akio Taniguchi"
__version__ = "0.2.0"


# submodules
from . import dataarray
from . import field
from . import typing
from . import utils


# aliases
from .dataarray import *
from .typing import *


# for sphinx docs
__all__ = dir()
