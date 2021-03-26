from __future__ import annotations


__all__ = ["asdataset", "datasetclass"]


# standard library
from dataclasses import dataclass
from functools import wraps
from types import FunctionType
from typing import Any, Callable, cast, Optional, Type, Union


# third-party packages
import xarray as xr


# submodules
from .common import get_attrs, get_coords, get_data_vars
from .typing import DataClass
from .utils import copy_class, extend_class, make_marked_subclass


# constants
TEMP_CLASS_PREFIX: str = "__Copied"


# runtime functions (public)
def asdataset(
    inst: DataClass, as_class: Type[xr.Dataset] = xr.Dataset
) -> xr.Dataset:
    """Convert a Dataset-class instance to Dataset one."""
    dataset = as_class(get_data_vars(inst))
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
    xarray_base: Optional[Type[object]] = None,
) -> Union[Type[DataClass], Callable[[type], Type[DataClass]]]:
    """Class decorator to create a Dataset class."""

    def to_dataclass(cls: type) -> Type[DataClass]:
        if not shorthands and xarray_base is not None:
            raise TypeError(
                "No shorthands not compatible with xarray_base"
            )
        mixin = None
        if xarray_base is not None:
            mixin = XArrayBaseDatasetMixin
        elif shorthands:
            mixin = DatasetMixin
        if mixin is not None:
            cls = extend_class(cls, mixin)
        if xarray_base is not None:
            cast(
                XArrayBaseDatasetMixin, cls
            ).xarray_base = make_marked_subclass(
                xr.Dataset, xarray_base, dict(__slots__=tuple())
            )

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

    #: whether new() returns a subclass of xarray inheriting from cls
    extend_xarray = False

    #: subclass of Dataset
    xarray_base = xr.Dataset

    @classmethod
    def new(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Create a Dataset instance."""
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Update new() based on the dataclass definition."""
        super().__init_subclass__(**kwargs)

        # temporary class only for getting dataclass __init__
        try:
            Temp = dataclass(copy_class(cls, TEMP_CLASS_PREFIX))
        except ValueError:
            return

        init: FunctionType = Temp.__init__  # type: ignore
        init.__annotations__["return"] = xr.Dataset

        # create a concrete new method and bind
        @classmethod
        @wraps(init)
        def new(
            cls: "DatasetMixin",
            *args: Any,
            **kwargs: Any,
        ) -> xr.Dataset:
            """Create a Dataset instance."""
            cls = cast(Type[DataClass], cls)
            return asdataset(
                cls(*args, **kwargs), as_class=cls.xarray_base
            )

        cls.new = new  # type: ignore


class XArrayBaseDatasetMixin(DatasetMixin):
    """
    Mixin marks class "new" returns instance derived from dataclass.
    """

    extend_xarray = True
