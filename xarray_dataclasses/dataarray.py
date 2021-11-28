__all__ = ["AsDataArray", "asdataarray"]


# standard library
from dataclasses import Field
from functools import wraps
from types import MethodType
from typing import Any, Callable, Dict, Sequence, Type, TypeVar, Union, overload


# dependencies
import numpy as np
import xarray as xr
from morecopy import copy
from typing_extensions import Literal, ParamSpec, Protocol


# submodules
from .datamodel import DataModel, Reference


# type hints
P = ParamSpec("P")
TDataArray = TypeVar("TDataArray", bound=xr.DataArray)
TDataArray_ = TypeVar("TDataArray_", bound=xr.DataArray, contravariant=True)
Order = Literal["C", "F"]
Shape = Union[Dict[str, int], Sequence[int], int]


class DataClass(Protocol[P]):
    """Type hint for a dataclass object."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]


class DataArrayClass(Protocol[P, TDataArray_]):
    """Type hint for a dataclass object with a DataArray factory."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]
    __dataarray_factory__: Callable[..., TDataArray_]


# custom classproperty
class classproperty:
    """Class property only for AsDataArray.new().

    As a classmethod and a property can be chained together since Python 3.9,
    this class will be removed when the support for Python 3.7 and 3.8 ends.

    """

    def __init__(self, func: Callable[..., Callable[P, TDataArray]]) -> None:
        self.__func__ = func

    def __get__(
        self,
        obj: Any,
        cls: Type[DataArrayClass[P, TDataArray]],
    ) -> Callable[P, TDataArray]:
        return self.__func__(cls)


# runtime functions and classes
@overload
def asdataarray(
    dataclass: DataArrayClass[Any, TDataArray],
    reference: Reference = None,
    dataarray_factory: Any = xr.DataArray,
) -> TDataArray:
    ...


@overload
def asdataarray(
    dataclass: DataClass[Any],
    reference: Reference = None,
    dataarray_factory: Callable[..., TDataArray] = xr.DataArray,
) -> TDataArray:
    ...


def asdataarray(
    dataclass: Any,
    reference: Any = None,
    dataarray_factory: Any = xr.DataArray,
) -> Any:
    """Create a DataArray object from a dataclass object.

    Args:
        dataclass: Dataclass object that defines typed DataArray.
        reference: DataArray or Dataset object as a reference of shape.
        dataset_factory: Factory function of DataArray.

    Returns:
        DataArray object created from the dataclass object.

    """
    try:
        dataarray_factory = dataclass.__dataarray_factory__
    except AttributeError:
        pass

    model = DataModel.from_dataclass(dataclass)
    dataarray = dataarray_factory(model.data[0](reference))

    for coord in model.coord:
        dataarray.coords.update({coord.name: coord(dataarray)})

    for attr in model.attr:
        dataarray.attrs.update({attr.name: attr()})

    for name in model.name:
        dataarray.name = name()

    return dataarray


class AsDataArray:
    """Mix-in class that provides shorthand methods."""

    def __dataarray_factory__(self, data: Any = None) -> xr.DataArray:
        """Default DataArray factory (xarray.DataArray)."""
        return xr.DataArray(data)

    @classproperty
    def new(cls: Type[DataArrayClass[P, TDataArray]]) -> Callable[P, TDataArray]:
        """Create a DataArray object from dataclass parameters."""

        init = copy(cls.__init__)
        init.__annotations__["return"] = TDataArray
        init.__doc__ = cls.__init__.__doc__

        @wraps(init)
        def new(
            cls: Type[DataArrayClass[P, TDataArray]],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> TDataArray:
            return asdataarray(cls(*args, **kwargs))

        return MethodType(new, cls)

    @classmethod
    def empty(
        cls: Type[DataArrayClass[P, TDataArray]],
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        """Create a DataArray object without initializing data.

        Args:
            shape: Shape of the new DataArray object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray object filled without initializing data.

        """
        model = DataModel.from_dataclass(cls)
        name = model.data[0].name
        dims = model.data[0].type["dims"]

        if isinstance(shape, dict):
            shape = tuple(shape[dim] for dim in dims)

        data = np.empty(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def zeros(
        cls: Type[DataArrayClass[P, TDataArray]],
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        """Create a DataArray object filled with zeros.

        Args:
            shape: Shape of the new DataArray object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray object filled with zeros.

        """
        model = DataModel.from_dataclass(cls)
        name = model.data[0].name
        dims = model.data[0].type["dims"]

        if isinstance(shape, dict):
            shape = tuple(shape[dim] for dim in dims)

        data = np.zeros(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def ones(
        cls: Type[DataArrayClass[P, TDataArray]],
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        """Create a DataArray object filled with ones.

        Args:
            shape: Shape of the new DataArray object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray object filled with ones.

        """
        model = DataModel.from_dataclass(cls)
        name = model.data[0].name
        dims = model.data[0].type["dims"]

        if isinstance(shape, dict):
            shape = tuple(shape[dim] for dim in dims)

        data = np.ones(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def full(
        cls: Type[DataArrayClass[P, TDataArray]],
        shape: Shape,
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        """Create a DataArray object filled with given value.

        Args:
            shape: Shape of the new DataArray object.
            fill_value: Value for the new DataArray object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray object filled with given value.

        """
        model = DataModel.from_dataclass(cls)
        name = model.data[0].name
        dims = model.data[0].type["dims"]

        if isinstance(shape, dict):
            shape = tuple(shape[dim] for dim in dims)

        data = np.full(shape, fill_value, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))
