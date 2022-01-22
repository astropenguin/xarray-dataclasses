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
from enum import Enum, auto
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
    """Annotations for xarray-related field types."""

    ATTR = auto()
    """Annotation for attribute field types."""

    COORD = auto()
    """Annotation for coordinate field types."""

    COORDOF = auto()
    """Annotation for coordinate field types."""

    DATA = auto()
    """Annotation for data (variable) field types."""

    DATAOF = auto()
    """Annotation for data (variable) field types."""

    NAME = auto()
    """Annotation for name field types."""

    def annotates(self, type: Any) -> bool:
        """Check if a type is annotated by FieldType."""
        return self in get_args(type)[1:]


# type hints
DataClassFields = Dict[str, Field[Any]]
DataType = Union[xr.DataArray, xr.Dataset]
Dims = Tuple[str, ...]
Dtype = Optional[str]
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]
Sizes = Dict[str, int]
P = ParamSpec("P")
T = TypeVar("T")
TDims = TypeVar("TDims", covariant=True)
TDtype = TypeVar("TDtype", covariant=True)
TName = TypeVar("TName", bound=Hashable)


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


TDataClass = TypeVar("TDataClass", bound=DataClass[Any])


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
def get_dims(hint: Any) -> Dims:
    """Return dims parsed from a type hint."""
    t_dims = get_inner(hint, 0, 0)

    if is_str_literal(t_dims):
        return (get_inner(t_dims, 0),)

    args: Any = get_args(t_dims)

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
