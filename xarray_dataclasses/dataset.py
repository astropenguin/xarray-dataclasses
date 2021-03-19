from __future__ import annotations


__all__ = ["asdataset", "datasetclass"]


# standard library
from dataclasses import dataclass
from types import FunctionType
from typing import Any, Callable, cast, Optional, Type, Union


# third-party packages
import xarray as xr


# submodules
from .common import (
    DataClass,
    get_attrs,
    get_coords,
    get_data_vars,
)
from .utils import copy_wraps, extend_class


# runtime functions (public)
def asdataset(inst: DataClass) -> xr.Dataset:
    """Convert a Dataset-class instance to Dataset one."""
    dataset = xr.Dataset(get_data_vars(inst))
    coords = get_coords(inst, dataset)

    dataset.coords.update(coords)
    dataset.attrs = get_attrs(inst)

    return dataset


def datasetclass(
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
    """Class decorator to create a Dataset class."""

    def to_dataclass(cls: type) -> Type[DataClass]:
        if shorthands:
            cls = extend_class(cls, DatasetMixin)

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
class DatasetMixin:
    """Mix-in class that provides shorthand methods."""

    @classmethod
    def new(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Create a Dataset instance."""
        cls = cast(Type[DataClass], cls)
        return asdataset(cls(*args, **kwargs))

    def __init_subclass__(cls) -> None:
        """Update new() based on the dataclass definition."""
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
        new.__annotations__["return"] = xr.Dataset
        new.__doc__ = "Create a Dataset instance."
        cls.new = classmethod(new)  # type: ignore
