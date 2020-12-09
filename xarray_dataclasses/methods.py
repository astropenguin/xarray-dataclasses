__all__ = [
    "empty",
    "zeros",
    "ones",
    "full",
]


# standard library
from typing import Any, Optional, Sequence, Union


# dependencies
import numpy as np
from numpy import ndarray
from typing_extensions import Literal


# constants
C: str = "C"
F: str = "F"


# type aliases
Shape = Sequence[int]
Dtype = Optional[Union[type, str]]
Order = Literal[C, F]


# main features
def empty(
    shape: Shape,
    dtype: Dtype = None,
    order: Order = C,
) -> ndarray:
    return np.empty(shape, dtype, order)


def zeros(
    shape: Shape,
    dtype: Dtype = None,
    order: Order = C,
) -> ndarray:
    return np.zeros(shape, dtype, order)


def ones(
    shape: Shape,
    dtype: Dtype = None,
    order: Order = C,
) -> ndarray:
    return np.ones(shape, dtype, order)


def full(
    shape: Shape,
    fill_value: Any,
    dtype: Dtype = None,
    order: Order = C,
) -> ndarray:
    return np.full(shape, fill_value, dtype, order)
