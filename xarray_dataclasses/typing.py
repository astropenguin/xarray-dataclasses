"""Submodule for type hints to define fields of dataclasses.

Note:
    The following codes are supposed in examples::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import AsDataArray, AsDataset
        from xarray_dataclasses import Attr, Coord, Data, Name
        from xarray_dataclasses import Coordof, Dataof


        X = Literal["x"]
        Y = Literal["y"]

"""
__all__ = ["Attr", "Coord", "Coordof", "Data", "Dataof", "Name"]


# standard library
from dataclasses import Field
from enum import auto, Enum
from typing import Any, Dict, Optional, Tuple, TypeVar, Union


# dependencies
import xarray as xr
from typing_extensions import (
    Annotated,
    get_args,
    get_origin,
    get_type_hints,
    Literal,
    Protocol,
    runtime_checkable,
)


# constants
class FieldType(Enum):
    """Annotation for xarray-related field types."""

    ATTR = auto()
    """Annotation for an attribute field type."""

    COORD = auto()
    """Annotation for a coordinate field type."""

    COORDOF = auto()
    """Annotation for a coordinate field type."""

    DATA = auto()
    """Annotation for a data (variable) field type."""

    DATAOF = auto()
    """Annotation for a data (variable) field type."""

    NAME = auto()
    """Annotation for a name field type."""

    def annotates(self, type: Any) -> bool:
        """Check if a field type is annotated."""
        return self in get_args(type)[1:]


# type hints
T = TypeVar("T")
TDims = TypeVar("TDims", covariant=True)
TDtype = TypeVar("TDtype", covariant=True)
Dims = Tuple[str, ...]
Dtype = Optional[str]
NoneType = type(None)
Reference = Union[xr.DataArray, xr.Dataset, None]


@runtime_checkable
class ArrayLike(Protocol[TDims, TDtype]):
    """Type hint for an array-like object."""

    def astype(self: T, dtype: Any) -> T:
        """Method for converting data type of the object."""
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions of the object."""
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the object."""
        ...


class DataClass(Protocol):
    """Type hint for a dataclass or its object."""

    __dataclass_fields__: Dict[str, Field[Any]]


TDataClass = TypeVar("TDataClass", bound=DataClass)


Attr = Annotated[T, FieldType.ATTR]
"""Type hint to define attribute fields (``Attr[T]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]
            long_name: Attr[str] = "luminance"
            units: Attr[str] = "cd / m^2"

Hint:
    The following field names are specially treated when plotting.

    - ``long_name`` or ``standard_name``: Coordinate name.
    - ``units``: Coordinate units.

Reference:
    https://xarray.pydata.org/en/stable/user-guide/plotting.html

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

Coordof = Annotated[Union[TDataClass, Any], FieldType.COORDOF]
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

Dataof = Annotated[Union[TDataClass, Any], FieldType.DATAOF]
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
def get_class(hint: Any) -> Any:
    """Return a class parsed from a type hint."""
    return get_first(unannotate(hint))


def get_dims(hint: Any) -> Dims:
    """Return dims parsed from a type hint."""
    t_dims = get_args(get_class(hint))[0]

    if is_str_literal(t_dims):
        return (get_first(t_dims),)

    args = get_args(t_dims)

    if args == () or args == ((),):
        return ()

    if all(map(is_str_literal, args)):
        return tuple(map(get_first, args))

    raise ValueError(f"Could not parse dims from {hint!r}.")


def get_dtype(hint: Any) -> Dtype:
    """Return dtype parsed from a type hint."""
    t_dtype = get_args(get_class(hint))[1]

    if t_dtype is Any:
        return None

    if t_dtype is NoneType:
        return None

    if isinstance(t_dtype, type):
        return t_dtype.__name__

    if is_str_literal(t_dtype):
        return get_first(t_dtype)

    raise ValueError(f"Could not parse dtype from {hint!r}.")


def get_first(hint: Any) -> Any:
    """Return the first argument in a type hint."""
    return get_args(hint)[0]


def is_str_literal(hint: Any) -> bool:
    """Check if a type hint is Literal[str]."""
    args = get_args(hint)
    origin = get_origin(hint)

    if origin is not Literal:
        return False

    if not len(args) == 1:
        return False

    return isinstance(args[0], str)


def unannotate(hint: Any) -> Any:
    """Recursively remove Annotated type hints."""

    class Temp:
        __annotations__ = dict(hint=hint)

    return get_type_hints(Temp)["hint"]
