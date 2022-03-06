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
    Collection,
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
)


# type hints (private)
PInit = ParamSpec("PInit")
TAttr = TypeVar("TAttr")
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


class Labeled(Protocol[TDims]):
    """Type hint for labeled objects."""

    pass


class Collection(Labeled[TDims], Collection[TDtype], Protocol):
    """Type hint for labeled collection objects."""

    pass


class DataClass(Protocol[PInit]):
    """Type hint for dataclass objects."""

    def __init__(self, *args: PInit.args, **kwargs: PInit.kwargs) -> None:
        ...

    __dataclass_fields__: ClassVar[DataClassFields]


# type hints (public)
class FieldType(Enum):
    """Annotation of xarray-related field hints."""

    ATTR = "attr"
    """Annotation of attribute field hints."""

    COORD = "coord"
    """Annotation of coordinate field hints."""

    DATA = "data"
    """Annotation of data (variable) field hints."""

    NAME = "name"
    """Annotation of name field hints."""

    def annotates(self, hint: Any) -> bool:
        """Check if a field hint is annotated."""
        return self in get_args(hint)[1:]


Attr = Annotated[TAttr, FieldType.ATTR]
"""Type hint to define attribute fields (``Attr[TAttr]``).

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

Coord = Annotated[Union[Collection[TDims, TDtype], TDtype], FieldType.COORD]
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

Coordof = Annotated[Union[TDataClass, Any], FieldType.COORD]
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

Data = Annotated[Union[Collection[TDims, TDtype], TDtype], FieldType.DATA]
"""Type hint to define data fields (``Coordof[TDims, TDtype]``).

Example:
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

Dataof = Annotated[Union[TDataClass, Any], FieldType.DATA]
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

    if origin is Collection:
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

    if origin is Collection:
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


def get_repr_type(type_: Any) -> Any:
    """Parse a type and return an representative type.

    Example:
        All of the following expressions will be ``True``::

            get_repr_type(A) == A
            get_repr_type(Annotated[A, ...]) == A
            get_repr_type(Union[A, B, ...]) == A
            get_repr_type(Optional[A]) == A

    """

    class Temporary:
        __annotations__ = dict(type=type_)

    unannotated = get_type_hints(Temporary)["type"]

    if get_origin(unannotated) is Union:
        return get_args(unannotated)[0]

    return unannotated
