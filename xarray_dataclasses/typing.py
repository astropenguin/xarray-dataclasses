__all__ = ["Attr", "Coord", "Data", "Multiple", "Tag"]


# standard library
from collections.abc import Collection
from dataclasses import Field
from enum import auto
from typing import Annotated as Ann, Any, Callable, ClassVar, Protocol, TypeVar, Union


# dependencies
from dataspecs import TagBase
from typing_extensions import ParamSpec
from xarray import DataArray, Dataset


# type hints
P = ParamSpec("P")
TAny = TypeVar("TAny")
TArray = TypeVar("TArray", bound=DataArray)
TDims = TypeVar("TDims", covariant=True)
TDtype = TypeVar("TDtype", covariant=True)
TSet = TypeVar("TSet", bound=Dataset)
TXarray = TypeVar("TXarray", bound=Union[DataArray, Dataset])


class DataClass(Protocol[P]):
    """Protocol for dataclass (object)."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...


class DataClassOf(Protocol[TXarray, P]):
    """Protocol for dataclass (object) with an xarray factory."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]
    __xarray_factory__: Callable[..., TXarray]

    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None: ...


class Labeled(Collection[TDtype], Protocol[TDims, TDtype]):
    """Same as Collection but accepts additional type variable for dims."""

    pass


# constants
class Tag(TagBase):
    """Collection of xarray-related tags for annotating type hints."""

    ATTR = auto()
    """Tag for specifying an attribute of DataArray/set."""

    COORD = auto()
    """Tag for specifying a coordinate of DataArray/set."""

    DATA = auto()
    """Tag for specifying a data object of DataArray/set."""

    DIMS = auto()
    """Tag for specifying a dims object of DataArray/set."""

    DTYPE = auto()
    """Tag for specifying a dtype object of DataArray/set."""

    MULTIPLE = auto()
    """Tag for specifying multiple items of DataArray/set."""


# type aliases
Attr = Ann[TAny, Tag.ATTR]
Coord = Ann[Labeled[Ann[TDims, Tag.DIMS], Ann[TDtype, Tag.DTYPE]], Tag.COORD]
Data = Ann[Labeled[Ann[TDims, Tag.DIMS], Ann[TDtype, Tag.DTYPE]], Tag.DATA]
Multiple = dict[str, Ann[TAny, Tag.MULTIPLE]]
