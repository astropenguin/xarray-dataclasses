# standard library
from typing import Any


# third-party packages
import numpy as np


# sub-modules/packages
from .typing import Attrs, DataArray, Dtype, Name, Shape, Order


# constants
C_ORDER: Order = "C"


# main features
def new(
    data: Any,
    *,
    name: Name = None,
    attrs: Attrs = None,
) -> DataArray:
    return DataArray(data, name=name, attrs=attrs)


def empty(
    shape: Shape,
    *,
    dtype: Dtype = None,
    order: Order = C_ORDER,
    name: Name = None,
    attrs: Attrs = None,
) -> DataArray:
    data = np.empty(shape, dtype, order)
    return new(data, name=name, attrs=attrs)


def zeros(
    shape: Shape,
    *,
    dtype: Dtype = None,
    order: Order = C_ORDER,
    name: Name = None,
    attrs: Attrs = None,
) -> DataArray:
    data = np.zeros(shape, dtype, order)
    return new(data, name=name, attrs=attrs)


def ones(
    shape: Shape,
    *,
    dtype: Dtype = None,
    order: Order = C_ORDER,
    name: Name = None,
    attrs: Attrs = None,
) -> DataArray:
    data = np.ones(shape, dtype, order)
    return new(data, name=name, attrs=attrs)


def full(
    shape: Shape,
    fill_value: Any,
    *,
    dtype: Dtype = None,
    order: Order = C_ORDER,
    name: Name = None,
    attrs: Attrs = None,
) -> DataArray:
    data = np.full(shape, fill_value, dtype, order)
    return new(data, name=name, attrs=attrs)
