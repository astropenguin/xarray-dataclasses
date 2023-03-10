__all__ = [
    "DataClass",
    "DataClassOf",
    "HashDict",
    "Pandas",
    "PAny",
    "TAny",
    "TFrame",
    "TPandas",
    "TSeries",
    "is_union",
]


# standard library
import types
from dataclasses import Field
from typing import Any, Callable, ClassVar, Dict, Hashable, Protocol, TypeVar, Union


# dependencies
from pandas import DataFrame, Series
from typing_extensions import ParamSpec, get_origin


HashDict = Dict[Hashable, Hashable]
"""Type hint for dictionary of hashable keys and values."""

Pandas = Union[DataFrame, "Series[Any]"]
"""Type hint for any pandas object."""

PAny = ParamSpec("PAny")
"""Parameter specification variable for any function."""

TAny = TypeVar("TAny")
"""Type variable for any class."""

TFrame = TypeVar("TFrame", bound=DataFrame)
"""Type variable for pandas DataFrame."""

TPandas = TypeVar("TPandas", bound=Pandas)
"""Type variable for any class of pandas object."""

TSeries = TypeVar("TSeries", bound="Series[Any]")
"""Type variable for pandas Series (of any dtype)."""


class DataClass(Protocol[PAny]):
    """Protocol for any dataclass object."""

    __dataclass_fields__: ClassVar[Dict[str, "Field[Any]"]]

    def __init__(self, *args: PAny.args, **kwargs: PAny.kwargs) -> None:
        ...


class DataClassOf(Protocol[TPandas, PAny]):
    """Protocol for any dataclass object with a factory."""

    __dataclass_fields__: ClassVar[Dict[str, "Field[Any]"]]
    __pandas_factory__: Callable[..., TPandas]

    def __init__(self, *args: PAny.args, **kwargs: PAny.kwargs) -> None:
        ...


def is_union(tp: Any) -> bool:
    """Check if a type hint is a union of types."""
    if UnionType := getattr(types, "UnionType", None):
        return get_origin(tp) is Union or isinstance(tp, UnionType)
    else:
        return get_origin(tp) is Union
