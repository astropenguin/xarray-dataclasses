from __future__ import annotations


__all__ = ["asdataset", "datasetclass"]


# standard library
from dataclasses import dataclass
from functools import wraps
from types import FunctionType
from typing import (
    Any,
    Callable,
    Literal,
    TypeVar,
    cast,
    Optional,
    Type,
    Union,
    overload,
)


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


T = TypeVar("T")


@overload
def datasetclass(cls: Type[object]) -> Type[DataClass]:
    ...


@overload
def datasetclass(
    *,
    shorthands: Literal[False],
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> Callable[[Type[object]], Type[DataClass]]:
    ...


@overload
def datasetclass(
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> Callable[[Type[object]], Type[DatasetMixin]]:
    ...


@overload
def datasetclass(
    *,
    xarray_base: Type[object],
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    shorthands: bool = True,
) -> Callable[[Type[object]], Type[DatasetMixin]]:
    ...


def datasetclass(
    cls: Optional[Type[object]] = None,
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
        if xarray_base is not None:
            cls = extend_class(cls, DatasetMixin)
            cast(DatasetMixin, cls).xarray_base = make_marked_subclass(
                xr.Dataset, xarray_base, dict(__slots__=tuple())
            )
        elif shorthands:
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


class DatasetMixin(DataClass):
    """DataClass that provides shorthand methods to create datasets."""

    #: whether new() returns a subclass of xarray inheriting from cls
    extend_xarray = False

    #: subclass of Dataset to return
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
            cls: Type[DatasetMixin],
            *args: Any,
            **kwargs: Any,
        ) -> xr.Dataset:
            """Create a Dataset instance."""
            cls = cast(Type[DatasetMixin], cls)
            return asdataset(
                cls(*args, **kwargs), as_class=cls.xarray_base
            )

        cls.new = new  # type: ignore
