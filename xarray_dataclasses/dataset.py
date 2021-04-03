from __future__ import annotations


__all__ = ["asdataset", "datasetclass"]


# standard library
from dataclasses import dataclass
from functools import wraps
from types import FunctionType
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
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
from .typing import DS, DataClass, DataClassX, WithClass
from .utils import copy_class, extend_class


# constants
TEMP_CLASS_PREFIX: str = "__Copied"


@overload
def asdataset(inst: DataClassX[DS]) -> DS:
    ...


@overload
def asdataset(inst: DataClass) -> xr.Dataset:
    ...


# runtime functions (public)
def asdataset(
    inst: Union[DataClass, DataClassX[DS]]
) -> Union[xr.Dataset, DS]:
    """Convert a Dataset-class instance to Dataset one."""
    if hasattr(inst.__class__, "__dataset_class__"):
        dataset: Union[xr.Dataset, DS] = cast(
            WithClass[DS], inst
        ).__dataset_class__(get_data_vars(inst))
    else:
        dataset = xr.Dataset(get_data_vars(inst))
    coords = get_coords(inst, dataset)

    dataset.coords.update(coords)
    dataset.attrs = get_attrs(inst)

    return dataset


class WithNewX(Generic[DS], WithClass[DS]):
    """Mix-in class that provides shorthand methods."""

    @classmethod
    def new(
        cls,
        *args: Any,
        **kwargs: Any,
    ) -> DS:
        """Create a Dataset instance."""
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Update new() based on the dataclass definition."""
        super().__init_subclass__()  # type: ignore
        # temporary class only for getting dataclass __init__
        try:
            Temp: type = dataclass(copy_class(cls, TEMP_CLASS_PREFIX))
        except ValueError:
            return

        init: FunctionType = Temp.__init__  # type: ignore
        dataset_class = cast(Type[DS], _find_dataset_class_for(cls))
        init.__annotations__["return"] = dataset_class

        # create a concrete new method and bind
        @classmethod  # type: ignore
        @wraps(init)
        def new(
            cls: "Type[WithNewX[DS]]",  # type: ignore
            *args: Any,
            **kwargs: Any,
        ) -> DS:
            """Create a Dataset instance."""
            return asdataset(cls(*args, **kwargs))

        cls.new = new  # type: ignore


WithNew = WithNewX[xr.Dataset]


# possible types of class decorated
DecInU = Type[object]
DecInB = Type[DataClass]
DecInX = Type[DataClassX[DS]]
DecInW = Type[WithNewX[DS]]
DecIn = Union[DecInU, DecInB, DecInX[DS], DecInW[DS]]

# possible result types of decorator
DecOutU = DataClass
DecOutB = DataClassX[xr.Dataset]
DecOutX = DataClassX[DS]
DecOut = Union[DecOutU, DecOutB, DecOutX[DS], DecInW[DS]]

# Possible "bare" decorator calls:

# untyped in -> implicit xr.Dataset
DCDecoratorU = Callable[[DecInU], DecOutU]

# base implicit xr.Dataset
DCDecoratorB = Callable[[DecInB], DecOutB]

# explicit derived xarray
DCDecoratorX = Callable[[DecInX[DS]], DecOutX[DS]]

# explicit "WithNew"
DCDecoratorW = Callable[[DecInW[DS]], DecInW[DS]]

# untyped in -> WithNew specialized for xr.Dataset
DCDecoratorWU = Callable[[DecInU], DecInW[xr.Dataset]]

# all decorators
DCDecorator = Union[
    DCDecoratorU,
    DCDecoratorB,
    DCDecoratorX[DS],
    DCDecoratorW[DS],
    DCDecoratorWU,
]

# all without shorthand
DCDecoratorNS = Union[DCDecoratorU, DCDecoratorB, DCDecoratorX[DS]]

# all with shorthand
DCDecoratorS = Union[DCDecoratorW[DS], DCDecoratorWU]


@overload
def datasetclass(cls: DecInX[DS]) -> WithNewX[DS]:
    ...


@overload
def datasetclass(cls: DecInU) -> "WithNewX[xr.Dataset]":
    ...


# @overload
# def datasetclass(
#     *,
#     shorthands: Literal[False],
#     init: bool = True,
#     repr: bool = True,
#     eq: bool = True,
#     order: bool = False,
#     unsafe_hash: bool = False,
#     frozen: bool = False,
# ) -> DCDecoratorNS[DS]:
#     ...


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
) -> Callable[[Union[DecInU, DecInB]], Union[DecOutU, DecOutB]]:
    ...


@overload
def datasetclass(
    *,
    shorthands: Literal[True],
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> DCDecoratorS[DS]:
    ...


def datasetclass(
    cls: Optional[DecIn[DS]] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    shorthands: bool = True,
) -> Union[DecOut[DS], DCDecorator[DS]]:
    """Class decorator to create a Dataset class."""

    def to_dataclass(cls: DecIn[DS]) -> DecOut[DS]:
        if issubclass(cls, WithClass):
            dataset_class = cast(Type[DS], _find_dataset_class_for(cls))
            cls.__dataset_class__ = dataset_class  # type: ignore
        if shorthands and not issubclass(cls, WithNewX):
            cls = extend_class(cls, WithNewX)

        dc = dataclass(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )(cls)
        return cast(DecOut[DS], dc)

    if cls is None:
        return cast(DCDecorator[DS], to_dataclass)
    else:
        return to_dataclass(cls)


def _find_dataset_class_for(cls: Union[WithClass[DS], Any]) -> Type[DS]:
    """
    Get dataset class from base annotation if available.

    If `cls` is declared as `Spec(WithClass[DerivedDataset])`
    for some `xr.Dataset` derivative `DerivedDataset`, this annotation
    is preserved in `cls` metadata, from which we extract it.

    WARNING: will look for classes derived from `WithClass`
    in bases, and takes first instantiation of a generic that
    derives from `xr.Dataset`. The user could confound this mechanism
    by (e.g.) adding generic parameters for other purposes.
    """
    if not hasattr(cls, "__orig_bases__"):
        return cast(Type[DS], xr.Dataset)
    for g_base in cast(Any, cls).__orig_bases__:
        if not issubclass(g_base.__origin__, WithClass):
            continue
        for g_arg in g_base.__args__:
            if isinstance(g_arg, type) and issubclass(
                g_arg, xr.Dataset
            ):
                return cast(Type[DS], g_arg)
    return cast(Type[DS], xr.Dataset)
