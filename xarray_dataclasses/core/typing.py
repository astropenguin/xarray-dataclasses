__all__ = [
    "DataClass",
    "DataClassOf",
    "PAny",
    "TAny",
    "TDataArray",
    "TDataset",
    "TXarray",
    "Xarray",
    "is_union",
]


# standard library
import types
from dataclasses import Field
from typing import Any, Callable, ClassVar, Protocol, TypeVar, Union


# dependencies
from xarray import DataArray, Dataset
from typing_extensions import ParamSpec, get_origin


Xarray = Union[DataArray, Dataset]
"""Type hint for any xarray object."""

PAny = ParamSpec("PAny")
"""Parameter specification variable for any function."""

TAny = TypeVar("TAny")
"""Type variable for any class."""

TDataArray = TypeVar("TDataArray", bound=DataArray)
"""Type variable for xarray DataArray."""

TDataset = TypeVar("TDataset", bound=Dataset)
"""Type variable for xarray Dataset."""

TXarray = TypeVar("TXarray", bound=Xarray)
"""Type variable for any class of xarray object."""


class DataClass(Protocol[PAny]):
    """Protocol for any dataclass object."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __init__(self, *args: PAny.args, **kwargs: PAny.kwargs) -> None:
        ...


class DataClassOf(Protocol[TXarray, PAny]):
    """Protocol for any dataclass object with a factory."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]
    __xarray_factory__: Callable[..., TXarray]

    def __init__(self, *args: PAny.args, **kwargs: PAny.kwargs) -> None:
        ...


def is_union(tp: Any) -> bool:
    """Check if a type hint is a union of types."""
    if UnionType := getattr(types, "UnionType", None):
        return get_origin(tp) is Union or isinstance(tp, UnionType)
    else:
        return get_origin(tp) is Union
