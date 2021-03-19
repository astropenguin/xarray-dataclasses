from __future__ import annotations


__all__ = ["asdataarray", "dataarrayclass"]


# standard library
from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable, cast, Optional, Sequence, Type, Union


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Literal


# submodules
from .common import get_attrs, get_coords, get_data, get_data_name, get_name
from .typing import DataClass
from .utils import copy_wraps, extend_class


# type hints (internal)
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]


# runtime functions (public)
def asdataarray(inst: DataClass) -> xr.DataArray:
    """Convert a DataArray-class instance to DataArray one."""
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
    shorthands: bool = True,
) -> Union[Type[DataClass], Callable[[type], Type[DataClass]]]:
    """Class decorator to create a DataArray class."""

    def to_dataclass(cls: type) -> Type[DataClass]:
        if shorthands:
            cls = extend_class(cls, DataArrayMixin)

        return dataclass(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )(cls)

    if cls is None:
        return to_dataclass
    else:
        return to_dataclass(cls)


# mix-in class (internal)
class DataArrayMixin:
    """Mix-in class that provides shorthand methods."""

    @classmethod
    def new(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> xr.DataArray:
        """Create a DataArray instance."""
        cls = cast(Type[DataClass], cls)
        return asdataarray(cls(*args, **kwargs))

    @classmethod
    def empty(
        cls,
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.DataArray:
        """Create a DataArray instance without initializing data.

        Args:
            shape: Shape of the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled without initializing data.

        """
        cls = cast(Type[DataClass], cls)
        name = get_data_name(cls)
        data = np.empty(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def zeros(
        cls,
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.DataArray:
        """Create a DataArray instance filled with zeros.

        Args:
            shape: Shape of the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled with zeros.

        """
        cls = cast(Type[DataClass], cls)
        name = get_data_name(cls)
        data = np.zeros(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def ones(
        cls,
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.DataArray:
        """Create a DataArray instance filled with ones.

        Args:
            shape: Shape of the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled with ones.

        """
        cls = cast(Type[DataClass], cls)
        name = get_data_name(cls)
        data = np.ones(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def full(
        cls,
        shape: Shape,
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.DataArray:
        """Create a DataArray instance filled with given value.

        Args:
            shape: Shape of the new DataArray instance.
            fill_value: Value for the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled with given value.

        """
        cls = cast(Type[DataClass], cls)
        name = get_data_name(cls)
        data = np.full(shape, fill_value, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Update new() based on the dataclass definition."""
        super().__init_subclass__(**kwargs)

        dataclass(
            init=True,
            repr=False,
            eq=False,
            order=False,
            unsafe_hash=False,
            frozen=False,
        )(cls)

        init = cast(FunctionType, cls.__init__)
        new = copy_wraps(init)(cls.new.__func__)  # type: ignore
        new.__annotations__["return"] = xr.DataArray
        new.__doc__ = "Create a DataArray instance."
        cls.new = classmethod(new)  # type: ignore
