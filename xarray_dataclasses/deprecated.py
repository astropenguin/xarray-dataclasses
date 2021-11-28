"""Submodule for functions that will be deprecated in v1.0.0."""
__all__ = ["dataarrayclass", "datasetclass"]


# standard library
import re
from dataclasses import dataclass, Field
from typing import Any, Callable, Dict, ForwardRef, Optional, Type, TypeVar, Union
from typing import _eval_type  # type: ignore
from warnings import warn


# dependencies
from typing_extensions import Literal, Protocol
from typing_extensions import get_type_hints as _get_type_hints


# constants
GENERIC_NAME = re.compile(r"\[.+\]")


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
    """Class decorator to create a DataArray class (deprecated in v1.0.0)."""
    from .dataarray import AsDataArray

    warn(
        "This decorator will be removed in v1.0.0. "
        "Please consider to use the Python's dataclass "
        "and the mix-in class (AsDataArray) instead.",
        category=FutureWarning,
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
    """Class decorator to create a Dataset class (deprecated in v1.0.0)."""
    from .dataset import AsDataset

    warn(
        "This decorator will be removed in v1.0.0. "
        "Please consider to use the Python's dataclass "
        "and the mix-in class (AsDataset) instead.",
        category=FutureWarning,
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


def eval_type(type: Any, *args: Any, **kwargs: Any) -> Any:
    if not isinstance(type, ForwardRef):
        return _eval_type(type, *args, **kwargs)

    name = type.__forward_arg__

    if GENERIC_NAME.search(name):
        return _eval_type(type, *args, **kwargs)

    warn(
        f"For backward compatibility, forward reference {name!r} "
        f"was not evaluated but converted to Literal[{name!r}]. "
        "From v1.0.0, it will be evaluated by default, which may "
        f"raise NameError or unexpectedly assign an object to {name}. "
        f"Please consider to replace {name!r} with Literal[{name!r}].",
        category=FutureWarning,
    )

    return Literal[name]  # type: ignore


def extend_class(cls: Type[T], mixin: Type[Any]) -> Type[T]:
    """Extend a class with a mix-in class."""
    if cls.__bases__ == (object,):
        bases = (mixin,)
    else:
        bases = (*cls.__bases__, mixin)

    return type(cls.__name__, bases, cls.__dict__.copy())


def get_type_hints(
    obj: Callable[..., Any],
    globalns: Optional[Dict[str, Any]] = None,
    localns: Optional[Dict[str, Any]] = None,
    include_extras: bool = False,
) -> Dict[str, Any]:
    """Return type hints for an object.

    Unlike the original (``typing.get_type_hints``), it does NOT handle
    forward references encoded as string but converts them to literal types.
    Other behavior is the same as the original.

    """
    import typing

    try:
        typing._eval_type = eval_type  # type: ignore
        return _get_type_hints(obj, globalns, localns, include_extras)
    finally:
        typing._eval_type = _eval_type  # type: ignore
