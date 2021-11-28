"""Submodule for type hints to define fields of dataclasses.

Note:
    The following code is supposed in the examples below::

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
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union


# dependencies
from typing_extensions import (
    Annotated,
    Literal,
    Protocol,
    get_args,
    get_origin,
    get_type_hints,
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

    __init__: Callable[..., None]
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
"""Type hint to define coordinate fields (``Coord[TDims, TDtype]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]
            mask: Coord[tuple[X, Y], bool]
            x: Coord[X, int] = 0
            y: Coord[Y, int] = 0

Hint:
    A coordinate field whose name is the same as ``TDims``
    (e.g. ``x: Coord[X, int]``) can define a dimension.

"""

Coordof = Annotated[Union[TDataClass, Any], FieldType.COORDOF]
"""Type hint to define coordinate fields (``Coordof[TDataClass]``).

Unlike ``Coord``, it specifies a dataclass that defines a DataArray class.
This is useful when users want to add metadata to dimensions for plotting.

Example:
    ::

        @dataclass
        class XAxis:
            data: Data[X, int]
            long_name: Attr[str] = "x axis"


        @dataclass
        class YAxis:
            data: Data[Y, int]
            long_name: Attr[str] = "y axis"


        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]
            x: Coordof[XAxis] = 0
            y: Coordof[YAxis] = 0

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
            data: Data[tuple[X, Y], float]

    Multiple data fields are allowed in a Dataset class::

        @dataclass
        class ColorImage(AsDataset):
            red: Data[tuple[X, Y], float]
            green: Data[tuple[X, Y], float]
            blue: Data[tuple[X, Y], float]

"""

Dataof = Annotated[Union[TDataClass, Any], FieldType.DATAOF]
"""Type hint to define data fields (``Coordof[TDataClass]``).

Unlike ``Data``, it specifies a dataclass that defines a DataArray class.
This is useful when users want to reuse a dataclass in a Dataset class.

Example:
    ::

        @dataclass
        class Image:
            data: Data[tuple[X, Y], float]
            x: Coord[X, int] = 0
            y: Coord[Y, int] = 0


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
            data: Data[tuple[X, Y], float]
            name: Name[str] = "image"

"""


# runtime functions
def get_dims(hint: Any) -> Dims:
    """Return dims parsed from a type hint."""
    t_dims = get_inner(hint, 0, 0)

    if is_str_literal(t_dims):
        return (get_inner(t_dims, 0),)

    args = get_args(t_dims)

    if args == () or args == ((),):
        return ()

    if all(map(is_str_literal, args)):
        return tuple(map(get_inner, args, [0] * len(args)))

    raise ValueError(f"Could not parse dims from {hint!r}.")


def get_dtype(hint: Any) -> Dtype:
    """Return dtype parsed from a type hint."""
    t_dtype = get_inner(hint, 0, 1)

    if t_dtype is Any:
        return None

    if t_dtype is type(None):
        return None

    if isinstance(t_dtype, type):
        return t_dtype.__name__

    if is_str_literal(t_dtype):
        return get_inner(t_dtype, 0)

    raise ValueError(f"Could not parse dtype from {hint!r}.")


def get_inner(hint: Any, *indexes: int) -> Any:
    """Return an inner type hint by indexes."""
    if not indexes:
        return hint

    index, indexes = indexes[0], indexes[1:]
    return get_inner(get_args(hint)[index], *indexes)


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
