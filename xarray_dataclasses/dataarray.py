__all__ = ["asdataarray", "AsDataArray", "dataarrayclass"]


# standard library
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, overload, Sequence, Type, Union
from warnings import warn


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Literal, Protocol


# submodules
from .parser import parse
from .typing import DataClass, TDataArray
from .utils import copy_class, extend_class


# type hints
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]


class DataClassWithFactory(DataClass, Protocol[TDataArray]):
    __dataarray_factory__: Callable[..., TDataArray]


# runtime functions
@overload
def asdataarray(
    inst: DataClassWithFactory[TDataArray],
    dataarray_factory: Type[Any] = xr.DataArray,
) -> TDataArray:
    ...


@overload
def asdataarray(
    inst: DataClass,
    dataarray_factory: Type[TDataArray] = xr.DataArray,
) -> TDataArray:
    ...


def asdataarray(inst: Any, dataarray_factory: Any = xr.DataArray) -> Any:
    """Convert a DataArray-class instance to DataArray one."""
    try:
        dataarray_factory = inst.__dataarray_factory__
    except AttributeError:
        pass

    return parse(inst).to_dataarray(dataarray_factory=dataarray_factory)


def dataarrayclass(
    cls: Optional[Type[Any]] = None,
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

    warn(
        DeprecationWarning(
            "This decorator will be removed in v1.0.0. ",
            "Please consider to use the Python's dataclass ",
            "and the mix-in class (AsDataArray) instead.",
        )
    )

    def to_dataclass(cls: Type[Any]) -> Type[DataClass]:
        if shorthands:
            cls = extend_class(cls, AsDataArray)

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


# mix-in class
class AsDataArray:
    """Mix-in class that provides shorthand methods."""

    __dataarray_factory__ = xr.DataArray

    @classmethod
    def new(
        cls: Type[DataClassWithFactory[TDataArray]],
        *args: Any,
        **kwargs: Any,
    ) -> TDataArray:
        """Create a DataArray instance."""
        raise NotImplementedError

    @classmethod
    def empty(
        cls: Type[DataClassWithFactory[TDataArray]],
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        """Create a DataArray instance without initializing data.

        Args:
            shape: Shape of the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled without initializing data.

        """
        name = parse(cls).data[0].name
        data = np.empty(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def zeros(
        cls: Type[DataClassWithFactory[TDataArray]],
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        """Create a DataArray instance filled with zeros.

        Args:
            shape: Shape of the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled with zeros.

        """
        name = parse(cls).data[0].name
        data = np.zeros(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def ones(
        cls: Type[DataClassWithFactory[TDataArray]],
        shape: Shape,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
        """Create a DataArray instance filled with ones.

        Args:
            shape: Shape of the new DataArray instance.
            order: Whether to store data in row-major (C-style)
                or column-major (Fortran-style) order in memory.
            kwargs: Args of the DataArray class except for data.

        Returns:
            DataArray instance filled with ones.

        """
        name = parse(cls).data[0].name
        data = np.ones(shape, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    @classmethod
    def full(
        cls: Type[DataClassWithFactory[TDataArray]],
        shape: Shape,
        fill_value: Any,
        order: Order = "C",
        **kwargs: Any,
    ) -> TDataArray:
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
        name = parse(cls).data[0].name
        data = np.full(shape, fill_value, order=order)
        return asdataarray(cls(**{name: data}, **kwargs))

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Update new() based on the dataclass definition."""
        super().__init_subclass__(**kwargs)

        # temporary class only for getting dataclass __init__
        try:
            Temp = dataclass(copy_class(cls))
        except RuntimeError:
            return

        init = Temp.__init__
        init.__annotations__["return"] = TDataArray

        # create a concrete new method and bind
        @classmethod
        @wraps(init)
        def new(
            cls: Type[DataClassWithFactory[TDataArray]],
            *args: Any,
            **kwargs: Any,
        ) -> TDataArray:
            return asdataarray(cls(*args, **kwargs))

        cls.new = new  # type: ignore
