__all__ = ["asdataarray", "dataarrayclass"]


# standard library
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Literal


# submodules
from .common import (
    DataClass,
    get_attrs,
    get_coords,
    get_data,
    get_name,
    get_data_name,
)
from .utils import copy_wraps


# type hints (internal)
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]


# runtime functions (public)
def asdataarray(inst: DataClass) -> xr.DataArray:
    """Create a DataArray instance from a DataArray class instance."""
    dataarray = get_data(inst)
    coords = get_coords(inst, dataarray)

    dataarray.coords.update(coords)
    dataarray.attrs = get_attrs(inst)
    dataarray.name = get_name(inst)

    return dataarray


def dataarrayclass(
    cls: Optional[type] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> Union[DataClass, Callable[[type], DataClass]]:
    """Class decorator to create a DataArray class."""

    set_options = dataclass(
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
    )

    def to_dataclass(cls: type) -> DataClass:
        set_options(cls)
        set_shorthands(cls)
        return cls  # type: ignore

    if cls is not None:
        return to_dataclass(cls)
    else:
        return to_dataclass


# runtime functions (internal)
def set_shorthands(cls: type) -> None:
    """Set shorthand methods to a DataArray class."""

    @copy_wraps(cls.__init__)  # type: ignore
    def new(cls, *args, **kwargs):
        return asdataarray(cls(*args, **kwargs))

    new.__annotations__["return"] = xr.DataArray
    new.__doc__ = (
        "Create a DataArray instance. This is a shorthand for "
        f"``asdataarray({cls.__name__}(*args, **kwargs))``."
    )

    cls.new = classmethod(new)  # type: ignore
    cls.empty = classmethod(empty)  # type: ignore
    cls.zeros = classmethod(zeros)  # type: ignore
    cls.ones = classmethod(ones)  # type: ignore
    cls.full = classmethod(full)  # type: ignore


# helper functions (internal)
def empty(cls, shape: Shape, order: Order = "C", **kwargs) -> xr.DataArray:
    """Create a DataArray instance without initializing data.

    Args:
        cls: DataArray class.
        shape: Shape of the new DataArray instance.
        order: Whether to store data in row-major (C-style)
            or column-major (Fortran-style) order in memory.
        kwargs: Args of the DataArray class except for data.

    Returns:
        A DataArray instance filled without initializing data.

    """
    name = get_data_name(cls)
    data = np.empty(shape, order=order)
    return asdataarray(cls(**{name: data}, **kwargs))


def zeros(cls, shape: Shape, order: Order = "C", **kwargs) -> xr.DataArray:
    """Create a DataArray instance filled with zeros.

    Args:
        cls: DataArray class.
        shape: Shape of the new DataArray instance.
        order: Whether to store data in row-major (C-style)
            or column-major (Fortran-style) order in memory.
        kwargs: Args of the DataArray class except for data.

    Returns:
        A DataArray instance filled with zeros.

    """
    name = get_data_name(cls)
    data = np.zeros(shape, order=order)
    return asdataarray(cls(**{name: data}, **kwargs))


def ones(cls, shape: Shape, order: Order = "C", **kwargs) -> xr.DataArray:
    """Create a DataArray instance filled with ones.

    Args:
        cls: DataArray class.
        shape: Shape of the new DataArray instance.
        order: Whether to store data in row-major (C-style)
            or column-major (Fortran-style) order in memory.
        kwargs: Args of the DataArray class except for data.

    Returns:
        A DataArray instance filled with ones.

    """
    name = get_data_name(cls)
    data = np.ones(shape, order=order)
    return asdataarray(cls(**{name: data}, **kwargs))


def full(
    cls, shape: Shape, fill_value: Any, order: Order = "C", **kwargs
) -> xr.DataArray:
    """Create a DataArray instance filled with given value.

    Args:
        cls: DataArray class.
        shape: Shape of the new DataArray instance.
        fill_value: Value for the new DataArray instance.
        order: Whether to store data in row-major (C-style)
            or column-major (Fortran-style) order in memory.
        kwargs: Args of the DataArray class except for data.

    Returns:
        A DataArray instance filled with given value.

    """
    name = get_data_name(cls)
    data = np.full(shape, fill_value, order=order)
    return asdataarray(cls(**{name: data}, **kwargs))
