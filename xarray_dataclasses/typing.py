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
from enum import Enum
from typing import (
    Any,
    ClassVar,
    Dict,
    Hashable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)


# dependencies
import xarray as xr
from more_itertools import collapse
from typing_extensions import (
    Annotated,
    Literal,
    ParamSpec,
    Protocol,
    get_args,
    get_origin,
    get_type_hints,
    runtime_checkable,
)


# constants
class FieldType(Enum):
    """Annotation of xarray-related field hints."""

    ATTR = "attr"
    """Annotation of attribute field hints."""

    COORD = "coord"
    """Annotation of coordinate field hints."""

    COORDOF = "coordof"
    """Annotation of coordinate field hints."""

    DATA = "data"
    """Annotation of data (variable) field hints."""

    DATAOF = "dataof"
    """Annotation of data (variable) field hints."""

    NAME = "name"
    """Annotation of name field hints."""

    def annotates(self, hint: Any) -> bool:
        """Check if a field hint is annotated."""
        return self in get_args(hint)[1:]


# type hints
P = ParamSpec("P")
T = TypeVar("T")
TDataClass = TypeVar("TDataClass", bound="DataClass[Any]")
TDims = TypeVar("TDims", covariant=True)
TDtype = TypeVar("TDtype", covariant=True)
TName = TypeVar("TName", bound=Hashable)

DataClassFields = Dict[str, Field[Any]]
DataType = Union[xr.DataArray, xr.Dataset]
Dims = Tuple[str, ...]
Dtype = Optional[str]
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]
Sizes = Dict[str, int]


@runtime_checkable
class ArrayLike(Protocol[TDims, TDtype]):
    """Type hint for array-like objects."""

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


class DataClass(Protocol[P]):
    """Type hint for dataclass objects."""

    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        ...

    __dataclass_fields__: ClassVar[DataClassFields]


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

Name = Annotated[TName, FieldType.NAME]
"""Type hint to define name fields (``Name[TName]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]
            name: Name[str] = "image"

"""


# runtime functions
def get_dims(type_: Any) -> Dims:
    """Parse a type and return dims.

    Example:
        All of the following expressions will be ``True``::

            get_dims(tuple[()]) == ()
            get_dims(Literal[A]) == (A,)
            get_dims(tuple[Literal[A], Literal[B]]) == (A, B)
            get_dims(ArrayLike[A, ...]) == get_dims(A)

    """
    args = get_args(type_)
    origin = get_origin(type_)

    if origin is ArrayLike:
        return get_dims(args[0])

    if origin is tuple or origin is Tuple:
        return tuple(collapse(map(get_dims, args)))

    if origin is Literal:
        return (args[0],)

    if type_ == () or type_ == ((),):
        return ()

    raise ValueError(f"Could not convert {type_!r} to dims.")


def get_dtype(type_: Any) -> Dtype:
    """Parse a type and return dtype.

    Example:
        All of the following expressions will be ``True``::

            get_dtype(Any) == None
            get_dtype(NoneType) == None
            get_dtype(A) == A.__name__
            get_dtype(Literal[A]) == A
            get_dtype(ArrayLike[..., A]) == get_dtype(A)

    """
    args = get_args(type_)
    origin = get_origin(type_)

    if origin is ArrayLike:
        return get_dtype(args[1])

    if origin is Literal:
        return args[0]

    if type_ is Any or type_ is type(None):
        return None

    if isinstance(type_, type):
        return type_.__name__

    raise ValueError(f"Could not convert {type_!r} to dtype.")


def get_field_type(type_: Any) -> FieldType:
    """Parse a type and return a field type if it exists."""
    if FieldType.ATTR.annotates(type_):
        return FieldType.ATTR

    if FieldType.COORD.annotates(type_):
        return FieldType.COORD

    if FieldType.DATA.annotates(type_):
        return FieldType.DATA

    if FieldType.NAME.annotates(type_):
        return FieldType.NAME

    raise TypeError(f"Could not find any field type in {type_!r}.")


def get_inner(hint: Any, *indexes: int) -> Any:
    """Return an inner type hint by indexes."""
    if not indexes:
        return hint

    index, indexes = indexes[0], indexes[1:]
    return get_inner(get_args(hint)[index], *indexes)


def is_str_literal(hint: Any) -> bool:
    """Check if a type hint is Literal[str]."""
    args: Any = get_args(hint)
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
