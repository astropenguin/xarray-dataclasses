__all__ = ["asdataarray", "dataarrayclass"]


# standard library
from dataclasses import dataclass
from functools import wraps
from types import FunctionType
from typing import Any, Callable, cast, Optional, Sequence, Type, Union


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Literal, Protocol


# submodules
from .common import get_attrs, get_coords, get_data, get_data_name, get_name
from .typing import DataClass
from .utils import copy_class, extend_class


# constants
TEMP_CLASS_PREFIX: str = "__Copied"


# type hints (internal)
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]
DA = TypeVar("DA", covariant=True, bound=xr.DataArray)


class HasFactory(Protocol[DA]):
    __dataarray_factory__: Callable[..., DA]


class DataArrayClass(DataClass, HasFactory[DA], Protocol):
    pass


# runtime functions (public)
def asdataarray(inst: DataArrayClass[DA]) -> DA:
    """Convert a DataArray-class instance to DataArray one."""
    dataarray = get_data(inst)
    coords = get_coords(inst, dataarray)

    dataarray.coords.update(coords)
    dataarray.attrs = get_attrs(inst)
    dataarray.name = get_name(inst)

    return inst.__dataarray_factory__(dataarray)


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

    __dataarray_factory__ = xr.DataArray

    @classmethod
    def new(
        cls: Type[DataArrayClass[DA]],
        *args: Any,
        **kwargs: Any,
    ) -> DA:
        """Create a DataArray instance."""
        raise NotImplementedError

    @classmethod
    def empty(
        cls: Type[DataArrayClass[DA]],
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> DA:
        """Create a DataArray instance without initializing data.

        Args:
            shape: Shape of the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled without initializing data.

        """
        name = get_data_name(cls)
        data = np.empty(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def zeros(
        cls: Type[DataArrayClass[DA]],
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> DA:
        """Create a DataArray instance filled with zeros.

        Args:
            shape: Shape of the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled with zeros.

        """
        name = get_data_name(cls)
        data = np.zeros(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def ones(
        cls: Type[DataArrayClass[DA]],
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> DA:
        """Create a DataArray instance filled with ones.

        Args:
            shape: Shape of the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled with ones.

        """
        name = get_data_name(cls)
        data = np.ones(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def full(
        cls: Type[DataArrayClass[DA]],
        shape: Shape,
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> DA:
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
        name = get_data_name(cls)
        data = np.full(shape, fill_value, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Update new() based on the dataclass definition."""
        super().__init_subclass__(**kwargs)

        # temporary class only for getting dataclass __init__
        try:
            Temp = dataclass(copy_class(cls, TEMP_CLASS_PREFIX))
        except ValueError:
            return

        init: FunctionType = Temp.__init__  # type: ignore
        init.__annotations__["return"] = DA

        # create a concrete new method and bind
        @classmethod
        @wraps(init)
        def new(
            cls: Type[DataArrayClass[DA]],
            *args: Any,
            **kwargs: Any,
        ) -> DA:
            """Create a DataArray instance."""
            return asdataarray(cls(*args, **kwargs))

        cls.new = new  # type: ignore
