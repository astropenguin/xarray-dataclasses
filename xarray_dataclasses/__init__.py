# flake8: noqa
__author__ = "Akio Taniguchi"
__version__ = "0.1.2"


# sub-modules
from . import utils
from . import typing
from . import methods
from . import core


# aliases
from .core import *
from .typing import *


# for sphinx docs
__all__ = dir()
