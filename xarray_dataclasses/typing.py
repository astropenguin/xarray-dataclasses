__all__ = ["Attr", "Coord", "Data", "Name"]


# standard library
from dataclasses import Field
from enum import auto, Enum
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)


# third-party packages
import xarray as xr
from typing_extensions import (
    Annotated,
    get_args,
    get_origin,
    Literal,
    Protocol,
)


# constants
class FieldType(Enum):
    """Annotation for xarray-related fields."""

    ATTR = auto()
    """Attribute field of DataArray or Dataset."""
    COORD = auto()
    """Coordinate field of DataArray or Dataset."""
    DATA = auto()
    """Data (variable) field of DataArray or Dataset."""
    NAME = auto()
    """Name field of DataArray."""

    def annotates(self, type_: Any) -> bool:
        """Check if type is annotated by the identifier."""
        args = get_args(type_)
        return len(args) > 1 and self in args[1:]


# type hints
Dims = Tuple[str, ...]
Dtype = Optional[str]
NoneType = type(None)

T = TypeVar("T")
TDataArray = TypeVar("TDataArray", bound=xr.DataArray)
TDataset = TypeVar("TDataset", bound=xr.Dataset)
TDims = TypeVar("TDims", covariant=True)
TDtype = TypeVar("TDtype", covariant=True)


class DataClass(Protocol):
    """Type hint for dataclass objects."""

    __init__: Callable[..., None]
    __dataclass_fields__: Dict[str, Field[Any]]


class ArrayLike(Protocol[TDims, TDtype]):
    """Type hint for array-like objects."""

    astype: Callable[..., Any]
    ndim: Any


DataArrayLike = Union[ArrayLike[TDims, TDtype], Sequence[TDtype], TDtype]
"""Type hint for DataArray-like objects."""

DataClassLike = Union[Type[DataClass], DataClass]
"""Type hint for DataClass-like objects."""

Attr = Annotated[T, FieldType.ATTR]
"""Type hint for an attribute member of DataArray or Dataset.

Examples:
    ::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import Data, Attr


        X = Literal["x"]
        Y = Literal["y"]


        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]
            dpi: Attr[int] = 300

"""

Coord = Annotated[DataArrayLike[TDims, TDtype], FieldType.COORD]
"""Type hint for a coordinate member of DataArray or Dataset.

Examples:
    ::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import Data, Coord


        X = Literal["x"]
        Y = Literal["y"]


        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]
            weight: Coord[tuple[X, Y], float] = 1.0
            x: Coord[X, int] = 0
            y: Coord[Y, int] = 0

"""

Data = Annotated[DataArrayLike[TDims, TDtype], FieldType.DATA]
"""Type hint for data of DataArray or variable of Dataset.

Examples:
    ::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import Data


        X = Literal["x"]
        Y = Literal["y"]


        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]

    ::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import Data


        X = Literal["x"]
        Y = Literal["y"]


        @dataclass
        class Images:
            red: Data[tuple[X, Y], float]
            green: Data[tuple[X, Y], float]
            blue: Data[tuple[X, Y], float]

"""

Name = Annotated[T, FieldType.NAME]
"""Type hint for a name of DataArray.

Examples:
    ::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import Data, Name


        X = Literal["x"]
        Y = Literal["y"]


        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]
            name: Name[str] = "default"

"""


# runtime functions
def get_dims(type_like: Any) -> Dims:
    """Parse a type-like object and get dims."""
    type_like = unannotate(type_like)

    if type_like == () or type_like is NoneType:
        return ()

    if isinstance(type_like, ForwardRef):
        return (type_like.__forward_arg__,)

    if isinstance(type_like, str):
        return (type_like,)

    origin = get_origin(type_like)
    args = get_args(type_like)

    if origin is tuple:
        return tuple(chain(*map(get_dims, args)))

    if origin is Literal:
        return tuple(map(str, args))

    raise ValueError(f"Could not parse {type_like}.")


def get_dtype(type_like: Any) -> Dtype:
    """Parse a type-like object and get dtype."""
    type_like = unannotate(type_like)

    if type_like is Any or type_like is NoneType:
        return None

    if isinstance(type_like, type):
        return type_like.__name__

    if isinstance(type_like, ForwardRef):
        return type_like.__forward_arg__

    if isinstance(type_like, str):
        return type_like

    origin = get_origin(type_like)
    args = get_args(type_like)

    if origin is Literal and len(args) == 1:
        return str(args[0])

    raise ValueError(f"Could not parse {type_like}.")


def unannotate(obj: T) -> T:
    """Recursively remove Annotated types."""
    if get_origin(obj) is Annotated:
        obj = get_args(obj)[0]

    origin = get_origin(obj)

    if origin is None:
        return obj

    args = map(unannotate, get_args(obj))
    args = tuple(filter(None, args))

    try:
        return origin[args]
    except TypeError:
        import typing

        name = origin.__name__.capitalize()
        return getattr(typing, name)[args]
