"""Submodule for type hints to define fields of dataclasses.

Note:
    The following imports are supposed in the examples below::

        from dataclasses import dataclass
        from typing import Literal
        from xarray_dataclasses import AsDataArray, AsDataset
        from xarray_dataclasses import Attr, Coord, Data, Name
        from xarray_dataclasses import Coordof, Dataof

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
    """Annotation of xarray-related field hints."""

    ATTR = auto()
    """Annotation of attribute field hints."""

    COORD = auto()
    """Annotation of coordinate field hints."""

    COORDOF = auto()
    """Annotation of coordinate field hints."""

    DATA = auto()
    """Annotation of data (variable) field hints."""

    DATAOF = auto()
    """Annotation of data (variable) field hints."""

    NAME = auto()
    """Annotation of name field hints."""

    def annotates(self, hint: Any) -> bool:
        """Check if a field hint is annotated."""
        return self in get_args(hint)[1:]


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
    """Type hint of array-like objects."""

    def astype(self: T, dtype: Any) -> T:
        """Method to convert data type of the object."""
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
    """Type hint of dataclasses or their objects."""

    __dataclass_fields__: Dict[str, Field[Any]]


TDataClass = TypeVar("TDataClass", bound=DataClass)


Attr = Annotated[T, FieldType.ATTR]
"""Type hint to define attribute fields (``Attr[T]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[Literal["x"], Literal["y"]], float]
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
"""Type hint to define coordinate fields (``Coord[TDims, TDtype]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[Literal["x"], Literal["y"]], float]
            mask: Coord[tuple[Literal["x"], Literal["y"]], bool]
            x: Coord[Literal["x"], int] = 0
            y: Coord[Literal["y"], int] = 0

Hint:
    A coordinate field whose name is the same as ``TDims``
    (e.g. ``x: Coord[Literal["x"], int]``) can define a dimension.

"""

Coordof = Annotated[Union[TDataClass, Any], FieldType.COORDOF]
"""Type hint to define coordinate fields (``Coordof[TDataClass]``).

Unlike ``Coord``, it specifies a dataclass that defines a DataArray class.
This is useful when users want to add metadata to dimensions for plotting.

Example:
    ::

        @dataclass
        class XAxis:
            data: Data[Literal["x"], int]
            long_name: Attr[str] = "x axis"


        @dataclass
        class YAxis:
            data: Data[Literal["y"], int]
            long_name: Attr[str] = "y axis"


        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[Literal["x"], Literal["y"]], float]
            x: Coordof[Literal["x"]Axis] = 0
            y: Coordof[Literal["y"]Axis] = 0

Hint:
    A class used in ``Coordof`` does not need to inherit ``AsDataArray``.

"""

Data = Annotated[Union[ArrayLike[TDims, TDtype], TDtype], FieldType.DATA]
"""Type hint to define data fields (``Coordof[TDims, TDtype]``).

Examples:
    Exactly one data field is allowed in a DataArray class
    (the second and subsequent data fields are just ignored)::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[Literal["x"], Literal["y"]], float]

    Multiple data fields are allowed in a Dataset class::

        @dataclass
        class ColorImage(AsDataset):
            red: Data[tuple[Literal["x"], Literal["y"]], float]
            green: Data[tuple[Literal["x"], Literal["y"]], float]
            blue: Data[tuple[Literal["x"], Literal["y"]], float]

"""

Dataof = Annotated[Union[TDataClass, Any], FieldType.DATAOF]
"""Type hint to define data fields (``Coordof[TDataClass]``).

Unlike ``Data``, it specifies a dataclass that defines a DataArray class.
This is useful when users want to reuse a dataclass in a Dataset class.

Example:
    ::

        @dataclass
        class Image:
            data: Data[tuple[Literal["x"], Literal["y"]], float]
            x: Coord[Literal["x"], int] = 0
            y: Coord[Literal["y"], int] = 0


        @dataclass
        class ColorImage(AsDataset):
            red: Dataof[Image]
            green: Dataof[Image]
            blue: Dataof[Image]

Hint:
    A class used in ``Dataof`` does not need to inherit ``AsDataArray``.

"""

Name = Annotated[T, FieldType.NAME]
"""Type hint to define name fields (``Name[T]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[Literal["x"], Literal["y"]], float]
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
