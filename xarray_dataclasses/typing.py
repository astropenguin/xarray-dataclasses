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
from dataclasses import Field, is_dataclass
from enum import Enum
from itertools import chain
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import (
    Annotated,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    get_args,
    get_origin,
    get_type_hints,
)


# type hints (private)
PInit = ParamSpec("PInit")
T = TypeVar("T")
TDataClass = TypeVar("TDataClass", bound="DataClass[Any]")
TDims = TypeVar("TDims", covariant=True)
TDType = TypeVar("TDType", covariant=True)
THashable = TypeVar("THashable", bound=Hashable)

AnyArray: TypeAlias = "np.ndarray[Any, Any]"
AnyDType: TypeAlias = "np.dtype[Any]"
AnyField: TypeAlias = "Field[Any]"
AnyXarray = Union[xr.DataArray, xr.Dataset]
Dims = Tuple[str, ...]
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]
Sizes = Dict[str, int]


class DataClass(Protocol[PInit]):
    """Type hint for dataclass objects."""

    def __init__(self, *args: PInit.args, **kwargs: PInit.kwargs) -> None:
        ...

    __dataclass_fields__: ClassVar[Dict[str, AnyField]]


class Labeled(Generic[TDims]):
    """Type hint for labeled objects."""

    pass


# type hints (public)
class FType(Enum):
    """Annotations for typing dataclass fields."""

    ATTR = "attr"
    """Annotation for attribute fields."""

    COORD = "coord"
    """Annotation for coordinate fields."""

    DATA = "data"
    """Annotation for data (variable) fields."""

    NAME = "name"
    """Annotation for name fields."""

    OTHER = "other"
    """Annotation for other fields."""

    @classmethod
    def annotates(cls, tp: Any) -> bool:
        """Check if any ftype annotates a type hint."""
        if get_origin(tp) is not Annotated:
            return False

        return any(isinstance(arg, cls) for arg in get_args(tp))


Attr = Annotated[T, FType.ATTR]
"""Type hint for attribute fields (``Attr[T]``).

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

Coord = Annotated[Union[Labeled[TDims], Collection[TDType], TDType], FType.COORD]
"""Type hint for coordinate fields (``Coord[TDims, TDType]``).

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

Coordof = Annotated[Union[TDataClass, Any], FType.COORD]
"""Type hint for coordinate fields (``Coordof[TDataClass]``).

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

Data = Annotated[Union[Labeled[TDims], Collection[TDType], TDType], FType.DATA]
"""Type hint for data fields (``Coordof[TDims, TDType]``).

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

Dataof = Annotated[Union[TDataClass, Any], FType.DATA]
"""Type hint for data fields (``Coordof[TDataClass]``).

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

Name = Annotated[THashable, FType.NAME]
"""Type hint for name fields (``Name[THashable]``).

Example:
    ::

        @dataclass
        class Image(AsDataArray):
            data: Data[tuple[X, Y], float]
            name: Name[str] = "image"

"""


# runtime functions
def deannotate(tp: Any) -> Any:
    """Recursively remove annotations in a type hint."""

    class Temporary:
        __annotations__ = dict(type=tp)

    return get_type_hints(Temporary)["type"]


def find_annotated(tp: Any) -> Iterable[Any]:
    """Generate all annotated types in a type hint."""
    args = get_args(tp)

    if get_origin(tp) is Annotated:
        yield tp
        yield from find_annotated(args[0])
    else:
        yield from chain(*map(find_annotated, args))


def get_annotated(tp: Any) -> Any:
    """Extract the first ftype-annotated type."""
    for annotated in filter(FType.annotates, find_annotated(tp)):
        return deannotate(annotated)

    raise TypeError("Could not find any ftype-annotated type.")


def get_annotations(tp: Any) -> Tuple[Any, ...]:
    """Extract annotations of the first ftype-annotated type."""
    for annotated in filter(FType.annotates, find_annotated(tp)):
        return get_args(annotated)[1:]

    raise TypeError("Could not find any ftype-annotated type.")


def get_dataclass(tp: Any) -> Type[DataClass[Any]]:
    """Extract a dataclass."""
    try:
        dataclass = get_args(get_annotated(tp))[0]
    except TypeError:
        raise TypeError(f"Could not find any dataclass in {tp!r}.")

    if not is_dataclass(dataclass):
        raise TypeError(f"Could not find any dataclass in {tp!r}.")

    return dataclass


def get_dims(tp: Any) -> Dims:
    """Extract data dimensions (dims)."""
    try:
        dims = get_args(get_args(get_annotated(tp))[0])[0]
    except TypeError:
        raise TypeError(f"Could not find any dims in {tp!r}.")

    args = get_args(dims)
    origin = get_origin(dims)

    if origin is Literal:
        return (str(args[0]),)

    if not (origin is tuple or origin is Tuple):
        raise TypeError(f"Could not find any dims in {tp!r}.")

    if args == () or args == ((),):
        return ()

    if not all(get_origin(arg) is Literal for arg in args):
        raise TypeError(f"Could not find any dims in {tp!r}.")

    return tuple(str(get_args(arg)[0]) for arg in args)


def get_dtype(tp: Any) -> Optional[AnyDType]:
    """Extract a NumPy data type (dtype)."""
    try:
        dtype = get_args(get_args(get_annotated(tp))[1])[0]
    except TypeError:
        raise TypeError(f"Could not find any dtype in {tp!r}.")

    if dtype is Any or dtype is type(None):
        return

    if get_origin(dtype) is Literal:
        dtype = get_args(dtype)[0]

    return np.dtype(dtype)


def get_ftype(tp: Any, default: FType = FType.OTHER) -> FType:
    """Extract an ftype if found or return given default."""
    try:
        return get_annotations(tp)[0]
    except TypeError:
        return default
