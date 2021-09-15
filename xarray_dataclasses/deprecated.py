"""Submodule for functions that will be deprecated in v1.0.0."""
__all__ = ["dataarrayclass", "datasetclass"]


# standard library
from dataclasses import dataclass, Field
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
from warnings import warn


# dependencies
from typing_extensions import Protocol


# submodules
from .dataarray import AsDataArray
from .dataset import AsDataset


# type hints
T = TypeVar("T")


class DataClass(Protocol):
    """Type hint for a dataclass object."""

    __init__: Callable[..., None]
    __dataclass_fields__: Dict[str, Field[Any]]


# functions to be deprecated
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


def datasetclass(
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
    """Class decorator to create a Dataset class."""

    warn(
        DeprecationWarning(
            "This decorator will be removed in v1.0.0. ",
            "Please consider to use the Python's dataclass ",
            "and the mix-in class (AsDataset) instead.",
        )
    )

    def to_dataclass(cls: Type[Any]) -> Type[DataClass]:
        if shorthands:
            cls = extend_class(cls, AsDataset)

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


def extend_class(cls: Type[T], mixin: Type[Any]) -> Type[T]:
    """Extend a class with a mix-in class."""
    if cls.__bases__ == (object,):
        bases = (mixin,)
    else:
        bases = (*cls.__bases__, mixin)

    return type(cls.__name__, bases, cls.__dict__.copy())
