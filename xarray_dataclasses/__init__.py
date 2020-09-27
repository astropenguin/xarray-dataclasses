# flake8: noqa
__author__ = "Akio Taniguchi"
__version__ = "0.1.2"


# submodules
from . import bases
from . import typing


# aliases
from .bases import *
from .typing import *


# for sphinx docs
__all__ = dir()
