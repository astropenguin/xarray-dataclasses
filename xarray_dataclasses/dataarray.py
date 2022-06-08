"""Submodule for DataArray creation."""
__all__ = ["AsDataArray", "asdataarray"]


# standard library
from functools import partial, wraps
from types import MethodType
from typing import Any, Callable, Optional, Type, TypeVar, Union, overload


# dependencies
import numpy as np
import xarray as xr
from morecopy import copy
from typing_extensions import ParamSpec, Protocol


# submodules
from .datamodel import DataModel
from .dataoptions import DataOptions
from .typing import AnyArray, AnyXarray, DataClass, Order, Shape, Sizes


# type hints
PInit = ParamSpec("PInit")
TDataArray = TypeVar("TDataArray", bound=xr.DataArray)


class OptionedClass(DataClass[PInit], Protocol[PInit, TDataArray]):
    """Type hint for dataclass objects with options."""

    __dataoptions__: DataOptions[TDataArray]


# runtime functions
@overload
def asdataarray(
    dataclass: OptionedClass[PInit, TDataArray],
    reference: Optional[AnyXarray] = None,
    dataoptions: None = None,
) -> TDataArray:
    ...


@overload
def asdataarray(
    dataclass: DataClass[PInit],
    reference: Optional[AnyXarray] = None,
    dataoptions: None = None,
) -> xr.DataArray:
    ...


@overload
def asdataarray(
    dataclass: Any,
    reference: Optional[AnyXarray] = None,
    dataoptions: DataOptions[TDataArray] = DataOptions(xr.DataArray),
) -> TDataArray:
    ...


def asdataarray(
    dataclass: Any,
    reference: Optional[AnyXarray] = None,
    dataoptions: Any = None,
) -> Any:
    """Create a DataArray object from a dataclass object.

    Args:
        dataclass: Dataclass object that defines typed DataArray.
        reference: DataArray or Dataset object as a reference of shape.
        dataoptions: Options for DataArray creation.

    Returns:
        DataArray object created from the dataclass object.

    """
    if dataoptions is None:
        try:
            dataoptions = dataclass.__dataoptions__
        except AttributeError:
            dataoptions = DataOptions(xr.DataArray)

    model = DataModel.from_dataclass(dataclass)
    dataarray = dataoptions.factory(model.data_vars[0](reference))

    for entry in model.coords:
        if entry.name in dataarray.dims:
            dataarray.coords[entry.name] = entry(dataarray)

    for entry in model.coords:
        if entry.name not in dataarray.dims:
            dataarray.coords[entry.name] = entry(dataarray)

    for entry in model.attrs:
        dataarray.attrs[entry.name] = entry()

    if model.names:
        dataarray.name = model.names[0]()

    return dataarray


# runtime classes
class classproperty:
    """Class property only for AsDataArray.new().

    As a classmethod and a property can be chained together since Python 3.9,
    this class will be removed when the support for Python 3.7 and 3.8 ends.

    """

    def __init__(self, func: Any) -> None:
        self.__func__ = func

    @overload
    def __get__(
        self,
        obj: Any,
        cls: Type[OptionedClass[PInit, TDataArray]],
    ) -> Callable[PInit, TDataArray]:
        ...

    @overload
    def __get__(
        self,
        obj: Any,
        cls: Type[DataClass[PInit]],
    ) -> Callable[PInit, xr.DataArray]:
        ...

    def __get__(self, obj: Any, cls: Any) -> Any:
        return self.__func__(cls)


class AsDataArray:
    """Mix-in class that provides shorthand methods."""

    @classproperty
    def new(cls: Any) -> Any:
        """Create a DataArray object from dataclass parameters."""

        init = copy(cls.__init__)
        init.__doc__ = cls.__init__.__doc__
        init.__annotations__["return"] = TDataArray

        @wraps(init)
        def new(cls: Any, *args: Any, **kwargs: Any) -> Any:
            return asdataarray(cls(*args, **kwargs))

        return MethodType(new, cls)

    @overload
    @classmethod
    def shaped(
        cls: Type[OptionedClass[PInit, TDataArray]],
        func: Callable[[Shape], AnyArray],
        shape: Union[Shape, Sizes],
        **kwargs: Any,
    ) -> TDataArray:
        ...

    @overload
    @classmethod
    def shaped(
        cls: Type[DataClass[PInit]],
        func: Callable[[Shape], AnyArray],
        shape: Union[Shape, Sizes],
        **kwargs: Any,
    ) -> xr.DataArray:
        ...

    @classmethod
    def shaped(
        cls: Any,
        func: Callable[[Shape], AnyArray],
        shape: Union[Shape, Sizes],
        **kwargs: Any,
    ) -> Any:
        """Create a DataArray object from a shaped function.

        Args:
            func: Function to create an array with given shape.
            shape: Shape or sizes of the new DataArray object.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray object created from the shaped function.

        """
        model = DataModel.from_dataclass(cls)
        key, entry = model.data_vars_items[0]

        if isinstance(shape, dict):
            shape = tuple(shape[dim] for dim in entry.dims)

        return asdataarray(cls(**{key: func(shape)}, **kwargs))

    @overload
    @classmethod
    def empty(
        cls: Type[OptionedClass[PInit, TDataArray]],
        shape: Union[Shape, Sizes],
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        ...

    @overload
    @classmethod
    def empty(
        cls: Type[DataClass[PInit]],
        shape: Union[Shape, Sizes],
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.DataArray:
        ...

    @classmethod
    def empty(
        cls: Any,
        shape: Union[Shape, Sizes],
        order: Order = "C",
        **kwargs: Any,
    ) -> Any:
        """Create a DataArray object without initializing data.

        Args:
            shape: Shape or sizes of the new DataArray object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray object without initializing data.

        """
        func = partial(np.empty, order=order)
        return cls.shaped(func, shape, **kwargs)

    @overload
    @classmethod
    def zeros(
        cls: Type[OptionedClass[PInit, TDataArray]],
        shape: Union[Shape, Sizes],
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        ...

    @overload
    @classmethod
    def zeros(
        cls: Type[DataClass[PInit]],
        shape: Union[Shape, Sizes],
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.DataArray:
        ...

    @classmethod
    def zeros(
        cls: Any,
        shape: Union[Shape, Sizes],
        order: Order = "C",
        **kwargs: Any,
    ) -> Any:
        """Create a DataArray object filled with zeros.

        Args:
            shape: Shape or sizes of the new DataArray object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray object filled with zeros.

        """
        func = partial(np.zeros, order=order)
        return cls.shaped(func, shape, **kwargs)

    @overload
    @classmethod
    def ones(
        cls: Type[OptionedClass[PInit, TDataArray]],
        shape: Union[Shape, Sizes],
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        ...

    @overload
    @classmethod
    def ones(
        cls: Type[DataClass[PInit]],
        shape: Union[Shape, Sizes],
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.DataArray:
        ...

    @classmethod
    def ones(
        cls: Any,
        shape: Union[Shape, Sizes],
        order: Order = "C",
        **kwargs: Any,
    ) -> Any:
        """Create a DataArray object filled with ones.

        Args:
            shape: Shape or sizes of the new DataArray object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray object filled with ones.

        """
        func = partial(np.ones, order=order)
        return cls.shaped(func, shape, **kwargs)

    @overload
    @classmethod
    def full(
        cls: Type[OptionedClass[PInit, TDataArray]],
        shape: Union[Shape, Sizes],
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        ...

    @overload
    @classmethod
    def full(
        cls: Type[DataClass[PInit]],
        shape: Union[Shape, Sizes],
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.DataArray:
        ...

    @classmethod
    def full(
        cls: Any,
        shape: Union[Shape, Sizes],
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> Any:
        """Create a DataArray object filled with given value.

        Args:
            shape: Shape or sizes of the new DataArray object.
            fill_value: Value for the new DataArray object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray object filled with given value.

        """
        func = partial(np.full, fill_value=fill_value, order=order)
        return cls.shaped(func, shape, **kwargs)
