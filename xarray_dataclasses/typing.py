__all__ = ["Attr", "Coord", "Coordof", "Data", "Dataof", "Name"]


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
    Tuple,
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
    runtime_checkable,
)


# constants
class FieldType(Enum):
    """Annotation for xarray-related fields."""

    ATTR = auto()  #: Attribute field of DataArray or Dataset.
    COORD = auto()  #: Coordinate field of DataArray or Dataset.
    COORDOF = auto()  #: Coordinate field of DataArray or Dataset.
    DATA = auto()  #: Data (variable) field of DataArray or Dataset.
    DATAOF = auto()  #: Data (variable) field of DataArray or Dataset.
    NAME = auto()  #: Name field of DataArray.

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


@runtime_checkable
class DataClass(Protocol):
    """Type hint for dataclass objects."""

    __init__: Callable[..., None]
    __dataclass_fields__: Dict[str, Field[Any]]


@runtime_checkable
class ArrayLike(Protocol[TDims, TDtype]):
    """Type hint for array-like objects."""

    astype: Callable[..., Any]
    ndim: Any


Attr = Annotated[T, FieldType.ATTR]
"""Type hint for an attribute member of DataArray or Dataset.

Examples:
    ::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import Attr, Data


        X = Literal["x"]
        Y = Literal["y"]


        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]
            units: Attr[str] = "cd / m^2"

"""

Coord = Annotated[Union[ArrayLike[TDims, TDtype], TDtype], FieldType.COORD]
"""Type hint for a coordinate member of DataArray or Dataset.

Examples:
    ::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import Coord, Data


        X = Literal["x"]
        Y = Literal["y"]


        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]
            x: Coord[X, int] = 0
            y: Coord[Y, int] = 0

"""

Coordof = Annotated[Union[T, ArrayLike[Any, Any], Any], FieldType.COORDOF]
"""Type hint for a coordinate member of DataArray or Dataset.

Unlike ``Coord``, it receives a dataclass that defines a DataArray class.

Examples:
    ::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import Data, Coordof


        X = Literal["x"]
        Y = Literal["y"]


        @dataclass
        class XAxis:
            data: Data[X, int]


        @dataclass
        class YAxis:
            data: Data[Y, int]


        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]
            x: Coordof[XAxis] = 0
            y: Coordof[YAxis] = 0

"""

Data = Annotated[Union[ArrayLike[TDims, TDtype], TDtype], FieldType.DATA]
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
        class Image:
            red: Data[tuple[X, Y], float]
            green: Data[tuple[X, Y], float]
            blue: Data[tuple[X, Y], float]

"""

Dataof = Annotated[Union[T, ArrayLike[Any, Any], Any], FieldType.DATAOF]
"""Type hint for data of DataArray or variable of Dataset.

Unlike ``Data``, it receives a dataclass that defines a DataArray class.

Examples:
    ::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import Coordof, Data, Dataof


        X = Literal["x"]
        Y = Literal["y"]


        @dataclass
        class XAxis:
            data: Data[X, int]


        @dataclass
        class YAxis:
            data: Data[Y, int]


        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]
            x: Coordof[XAxis] = 0
            y: Coordof[YAxis] = 0


        @dataclass
        class ColorImage:
            red: Dataof[Image]
            green: Dataof[Image]
            red: Dataof[Image]

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
            name: Name[str] = "image"

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
