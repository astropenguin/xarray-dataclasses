# standard library
from typing import Any


# dependencies
import numpy as np
import xarray as xr
from .typing import Attrs, Dtype, Name, Shape, Order


# constants
C: str = "C"
F: str = "F"


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
