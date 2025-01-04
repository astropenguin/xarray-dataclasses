__all__ = [
    "Attr",
    "Attrs",
    "Coord",
    "Coords",
    "Data",
    "DataVars",
    "Factory",
    "Name",
    "Tag",
]


# standard library
from collections.abc import Collection as Collection_, Hashable
from dataclasses import Field
from enum import auto
from typing import (
    Annotated,
    Any,
    Callable,
    ClassVar,
    ParamSpec,
    Protocol,
    TypeVar,
    Union,
)


# dependencies
from dataspecs import TagBase
from xarray import DataArray, Dataset


# type hints
PAny = ParamSpec("PAny")
TAny = TypeVar("TAny")
TDataArray = TypeVar("TDataArray", bound=DataArray)
TDataset = TypeVar("TDataset", bound=Dataset)
TDims = TypeVar("TDims", covariant=True)
TDtype = TypeVar("TDtype", covariant=True)
THashable = TypeVar("THashable", bound=Hashable)
TXarray = TypeVar("TXarray", covariant=True, bound="Xarray")
Xarray = Union[DataArray, Dataset]


class Collection(Collection_[TDtype], Protocol[TDims, TDtype]):
    """Same as Collection[T] but accepts additional type variable for dims."""

    pass


class DataClass(Protocol[PAny]):
    """Protocol for a dataclass object."""

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __init__(self, *args: PAny.args, **kwargs: PAny.kwargs) -> None: ...


class DataClassOf(Protocol[PAny, TXarray]):
    """Protocol for a dataclass object with an xarray factory."""

    _xarray_factory: Callable[..., TXarray]
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def __init__(self, *args: PAny.args, **kwargs: PAny.kwargs) -> None: ...


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

    FACTORY = auto()
    """Tag for specifying a factory of DataArray/set."""

    MULTIPLE = auto()
    """Tag for specifying multiple items (attrs, coords, data vars)."""

    NAME = auto()
    """Tag for specifying an item name (attr, coord, data). """


# type aliases
Arrayable = Collection[Annotated[TDims, Tag.DIMS], Annotated[TDtype, Tag.DTYPE]]
"""Type alias for Collection[TDims, TDtype] annotated by tags."""

Attr = Annotated[TAny, Tag.ATTR]
"""Type alias for an attribute of DataArray/set."""

Attrs = Annotated[dict[str, TAny], Tag.ATTR, Tag.MULTIPLE]
"""Type alias for attributes of DataArray/set."""

Coord = Annotated[Arrayable[TDims, TDtype], Tag.COORD]
"""Type alias for a coordinate of DataArray/set."""

Coords = Annotated[dict[str, Arrayable[TDims, TDtype]], Tag.COORD, Tag.MULTIPLE]
"""Type alias for coordinates of DataArray/set."""

Data = Annotated[Arrayable[TDims, TDtype], Tag.DATA]
"""Type alias for a data object of DataArray/set."""

DataVars = Annotated[dict[str, Arrayable[TDims, TDtype]], Tag.DATA, Tag.MULTIPLE]
"""Type alias for data objects of DataArray/set."""

Factory = Annotated[Callable[..., TXarray], Tag.FACTORY]
"""Type alias for a factory of DataArray/set."""

Name = Annotated[THashable, Tag.NAME]
"""Type alias for an item name (attr, coord, data)."""
