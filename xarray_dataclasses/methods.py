__all__ = ["new", "empty", "zeros", "ones", "full"]


# standard library
from typing import Any, Optional, Sequence, Union


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import Literal
from .typing import Name, Attrs


# constants
C: str = "C"
F: str = "F"


# type aliases
Shape = Sequence[int]
Dtype = Optional[Union[type, str]]
Order = Literal[C, F]


# main features
def new(
    data: Any,
    *,
    name: Name = None,
    attrs: Attrs = None,
) -> xr.DataArray:
    return xr.DataArray(data, name=name, attrs=attrs)


def empty(
    shape: Shape,
    *,
    dtype: Dtype = None,
    order: Order = C,
    name: Name = None,
    attrs: Attrs = None,
) -> xr.DataArray:
    return new(np.empty(shape, dtype, order), name, attrs)


def zeros(
    shape: Shape,
    *,
    dtype: Dtype = None,
    order: Order = C,
    name: Name = None,
    attrs: Attrs = None,
) -> xr.DataArray:
    return new(np.zeros(shape, dtype, order), name, attrs)


def ones(
    shape: Shape,
    *,
    dtype: Dtype = None,
    order: Order = C,
    name: Name = None,
    attrs: Attrs = None,
) -> xr.DataArray:
    return np.ones(shape, dtype, order)
    return new(np.ones(shape, dtype, order), name, attrs)


def full(
    shape: Shape,
    fill_value: Any,
    *,
    dtype: Dtype = None,
    order: Order = C,
    name: Name = None,
    attrs: Attrs = None,
) -> xr.DataArray:
    return new(np.full(shape, fill_value, dtype, order), name, attrs)
