"""Submodule for Dataset creation."""
__all__ = ["AsDataset", "asdataset"]


# standard library
from functools import partial, wraps
from types import MethodType
from typing import Any, Callable, Dict, Optional, Type, TypeVar, overload


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
TDataset = TypeVar("TDataset", bound=xr.Dataset)


class OptionedClass(DataClass[PInit], Protocol[PInit, TDataset]):
    """Type hint for dataclass objects with options."""

    __dataoptions__: DataOptions[TDataset]


# runtime functions and classes
@overload
def asdataset(
    dataclass: OptionedClass[PInit, TDataset],
    reference: Optional[AnyXarray] = None,
    dataoptions: None = None,
) -> TDataset:
    ...


@overload
def asdataset(
    dataclass: DataClass[PInit],
    reference: Optional[AnyXarray] = None,
    dataoptions: None = None,
) -> xr.Dataset:
    ...


@overload
def asdataset(
    dataclass: Any,
    reference: Optional[AnyXarray] = None,
    dataoptions: DataOptions[TDataset] = DataOptions(xr.Dataset),
) -> TDataset:
    ...


def asdataset(
    dataclass: Any,
    reference: Optional[AnyXarray] = None,
    dataoptions: Any = None,
) -> Any:
    """Create a Dataset object from a dataclass object.

    Args:
        dataclass: Dataclass object that defines typed Dataset.
        reference: DataArray or Dataset object as a reference of shape.
        dataoptions: Options for Dataset creation.

    Returns:
        Dataset object created from the dataclass object.

    """
    if dataoptions is None:
        try:
            dataoptions = dataclass.__dataoptions__
        except AttributeError:
            dataoptions = DataOptions(xr.Dataset)

    model = DataModel.from_dataclass(dataclass)
    dataset = dataoptions.factory()

    for entry in model.data_vars:
        dataset[entry.name] = entry(reference)

    for entry in model.coords:
        if entry.name in dataset.dims:
            dataset.coords[entry.name] = entry(dataset)

    for entry in model.coords:
        if entry.name not in dataset.dims:
            dataset.coords[entry.name] = entry(dataset)

    for entry in model.attrs:
        dataset.attrs[entry.name] = entry()

    return dataset


# runtime classes
class classproperty:
    """Class property only for AsDataset.new().

    As a classmethod and a property can be chained together since Python 3.9,
    this class will be removed when the support for Python 3.7 and 3.8 ends.

    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self.__func__ = func

    @overload
    def __get__(
        self,
        obj: Any,
        cls: Type[OptionedClass[PInit, TDataset]],
    ) -> Callable[PInit, TDataset]:
        ...

    @overload
    def __get__(
        self,
        obj: Any,
        cls: Type[DataClass[PInit]],
    ) -> Callable[PInit, xr.Dataset]:
        ...

    def __get__(self, obj: Any, cls: Any) -> Any:
        return self.__func__(cls)


class AsDataset:
    """Mix-in class that provides shorthand methods."""

    @classproperty
    def new(cls: Any) -> Any:
        """Create a Dataset object from dataclass parameters."""

        init = copy(cls.__init__)
        init.__doc__ = cls.__init__.__doc__
        init.__annotations__["return"] = TDataset

        @wraps(init)
        def new(cls: Any, *args: Any, **kwargs: Any) -> Any:
            return asdataset(cls(*args, **kwargs))

        return MethodType(new, cls)

    @overload
    @classmethod
    def shaped(
        cls: Type[OptionedClass[PInit, TDataset]],
        func: Callable[[Shape], AnyArray],
        sizes: Sizes,
        **kwargs: Any,
    ) -> TDataset:
        ...

    @overload
    @classmethod
    def shaped(
        cls: Type[DataClass[PInit]],
        func: Callable[[Shape], AnyArray],
        sizes: Sizes,
        **kwargs: Any,
    ) -> xr.Dataset:
        ...

    @classmethod
    def shaped(
        cls: Any,
        func: Callable[[Shape], AnyArray],
        sizes: Sizes,
        **kwargs: Any,
    ) -> Any:
        """Create a Dataset object from a shaped function.

        Args:
            func: Function to create an array with given shape.
            sizes: Sizes of the new Dataset object.
            kwargs: Args of the Dataset class except for data vars.

        Returns:
            Dataset object created from the shaped function.

        """
        model = DataModel.from_dataclass(cls)
        data_vars: Dict[str, AnyArray] = {}

        for key, entry in model.data_vars_items:
            shape = tuple(sizes[dim] for dim in entry.dims)
            data_vars[key] = func(shape)

        return asdataset(cls(**data_vars, **kwargs))

    @overload
    @classmethod
    def empty(
        cls: Type[OptionedClass[PInit, TDataset]],
        sizes: Sizes,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataset:
        ...

    @overload
    @classmethod
    def empty(
        cls: Type[DataClass[PInit]],
        sizes: Sizes,
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.Dataset:
        ...

    @classmethod
    def empty(
        cls: Any,
        sizes: Sizes,
        order: Order = "C",
        **kwargs: Any,
    ) -> Any:
        """Create a Dataset object without initializing data vars.

        Args:
            sizes: Sizes of the new Dataset object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the Dataset class except for data vars.

        Returns:
            Dataset object without initializing data vars.

        """
        func = partial(np.empty, order=order)
        return cls.shaped(func, sizes, **kwargs)

    @overload
    @classmethod
    def zeros(
        cls: Type[OptionedClass[PInit, TDataset]],
        sizes: Sizes,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataset:
        ...

    @overload
    @classmethod
    def zeros(
        cls: Type[DataClass[PInit]],
        sizes: Sizes,
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.Dataset:
        ...

    @classmethod
    def zeros(
        cls: Any,
        sizes: Sizes,
        order: Order = "C",
        **kwargs: Any,
    ) -> Any:
        """Create a Dataset object whose data vars are filled with zeros.

        Args:
            sizes: Sizes of the new Dataset object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the Dataset class except for data vars.

        Returns:
            Dataset object whose data vars are filled with zeros.

        """
        func = partial(np.zeros, order=order)
        return cls.shaped(func, sizes, **kwargs)

    @overload
    @classmethod
    def ones(
        cls: Type[OptionedClass[PInit, TDataset]],
        sizes: Sizes,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataset:
        ...

    @overload
    @classmethod
    def ones(
        cls: Type[DataClass[PInit]],
        sizes: Sizes,
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.Dataset:
        ...

    @classmethod
    def ones(
        cls: Any,
        sizes: Sizes,
        order: Order = "C",
        **kwargs: Any,
    ) -> Any:
        """Create a Dataset object whose data vars are filled with ones.

        Args:
            sizes: Sizes of the new Dataset object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the Dataset class except for data vars.

        Returns:
            Dataset object whose data vars are filled with ones.

        """
        func = partial(np.ones, order=order)
        return cls.shaped(func, sizes, **kwargs)

    @overload
    @classmethod
    def full(
        cls: Type[OptionedClass[PInit, TDataset]],
        sizes: Sizes,
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataset:
        ...

    @overload
    @classmethod
    def full(
        cls: Type[DataClass[PInit]],
        sizes: Sizes,
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> xr.Dataset:
        ...

    @classmethod
    def full(
        cls: Any,
        sizes: Sizes,
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> Any:
        """Create a Dataset object whose data vars are filled with given value.

        Args:
            sizes: Sizes of the new Dataset object.
            fill_value: Value for data vars of the new Dataset object.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the Dataset class except for data vars.

        Returns:
            Dataset object whose data vars are filled with given value.

        """
        func = partial(np.full, fill_value=fill_value, order=order)
        return cls.shaped(func, sizes, **kwargs)
