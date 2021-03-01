# flake8: noqa
__author__ = "Akio Taniguchi"
__version__ = "0.1.2"


# submodules
from . import core
from . import field
from . import methods
from . import typing
from . import utils


# aliases
from .core import *
from .typing import *


# for sphinx docs
__all__ = dir()
