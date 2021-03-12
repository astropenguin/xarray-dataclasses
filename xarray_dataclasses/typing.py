__all__ = ["Attr", "Coord", "Data", "Name"]


# standard library
from enum import auto, Enum
from functools import wraps
from typing import (
    Any,
    ForwardRef,
    Generic,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Annotated, get_args, get_origin, Literal


# constants (internal)
class Xarray(Enum):
    """Identifiers of type hints for xarray."""

    ATTR = auto()  #: Attribute member of DataArray or Dataset.
    COORD = auto()  #: Coordinate member of DataArray or Dataset.
    DATA = auto()  #: Data of DataArray or variable of Dataset.
    NAME = auto()  #: Name of DataArray.


# type variables (internal)
T = TypeVar("T")  #: Type variable for data types.
D = TypeVar("D")  #: Type variable for dimensions.


# type hints (internal)
class ndarray(Generic[T]):
    """Generic version of numpy.ndarray."""

    @wraps(np.ndarray)
    def __new__(cls, *args, **kwargs):
        return np.ndarray(*args, **kwargs)


class DataArray(Generic[T, D]):
    """Generic version of xarray.DataArray."""

    @wraps(xr.DataArray)
    def __new__(cls, *args, **kwargs):
        return xr.DataArray(*args, **kwargs)


DataArrayLike = Union[DataArray[T, D], ndarray[T], Sequence[T], T]


# type hints (public)
Attr = Annotated[T, Xarray.ATTR]
"""Type hint for attribute member of DataArray or Dataset.

Examples:
    ::

        from typing import Literal
        from xarray_dataclasses import dataarrayclass, Data, Attr


        X = Literal["x"]
        Y = Literal["y"]


        @dataarrayclass
        class Image:
            data: Data[float, tuple[X, Y]]
            dpi: Attr[int] = 300

"""

Coord = Annotated[DataArrayLike[T, D], Xarray.COORD]
"""Type hint for coordinate member of DataArray or Dataset.

Examples:
    ::

        from typing import Literal
        from xarray_dataclasses import dataarrayclass, Data, Coord


        X = Literal["x"]
        Y = Literal["y"]


        @dataarrayclass
        class Image:
            data: Data[float, tuple[X, Y]]
            weight: Coord[float, tuple[X, Y]] = 1.0
            x: Coord[int, X] = 0
            y: Coord[int, Y] = 0

"""

Data = Annotated[DataArrayLike[T, D], Xarray.DATA]
"""Type hint for data of DataArray or variable of Dataset.

Examples:
    ::

        from typing import Literal
        from xarray_dataclasses import dataarrayclass, Data


        X = Literal["x"]
        Y = Literal["y"]


        @dataarrayclass
        class Image:
            data: Data[float, tuple[X, Y]]

    ::

        from typing import Literal
        from xarray_dataclasses import datasetclass, Data


        X = Literal["x"]
        Y = Literal["y"]


        @datasetclass
        class Images:
            red: Data[float, tuple[X, Y]]
            green: Data[float, tuple[X, Y]]
            blue: Data[float, tuple[X, Y]]

"""

Name = Annotated[T, Xarray.NAME]
"""Type hint for name of DataArray.

Examples:
    ::

        from typing import Literal
        from xarray_dataclasses import dataarrayclass, Data, Name


        X = Literal["x"]
        Y = Literal["y"]


        @dataarrayclass
        class Image:
            data: Data[float, tuple[X, Y]]
            name: Name[str] = "default"

"""


# runtime functions (internal)
def is_attr(type_: Any) -> bool:
    """Check if type is Attr[T]."""
    return _has_xarray_id(type_, Xarray.ATTR)


def is_coord(type_: Any) -> bool:
    """Check if type is Coord[T, D]."""
    return _has_xarray_id(type_, Xarray.COORD)


def is_data(type_: Any) -> bool:
    """Check if type is Data[T, D]."""
    return _has_xarray_id(type_, Xarray.DATA)


def is_name(type_: Any) -> bool:
    """Check if type is Name[T]."""
    return _has_xarray_id(type_, Xarray.NAME)


def get_dims(type_: Type[DataArrayLike]) -> Tuple[str, ...]:
    """Extract dimensions from DataArrayLike[T, D]."""
    if get_origin(type_) is Annotated:
        type_ = get_args(type_)[0]

    dtype, dims = get_args(get_args(type_)[0])

    if get_origin(dims) is tuple:
        return tuple(_unwrap(dim) for dim in get_args(dims))
    else:
        return (_unwrap(dims),)


def get_dtype(type_: Type[DataArrayLike]) -> Optional[np.dtype]:
    """Extract data type from DataArrayLike[T, D]."""
    if get_origin(type_) is Annotated:
        type_ = get_args(type_)[0]

    dtype, dims = get_args(get_args(type_)[0])

    if dtype is Any or dtype is None:
        return None
    else:
        return np.dtype(_unwrap(dtype))


# helper functions (internal)
def _has_xarray_id(type_: Any, id: Xarray) -> bool:
    """Check if type has given identifier of xarray."""
    args = get_args(type_)
    return (len(args) > 1) and (args[1] is id)


def _unwrap(type_: T) -> Union[T, str]:
    """Extract string from a type hint if possible."""
    if get_origin(type_) is Literal:
        return str(get_args(type_)[0])

    if isinstance(type_, ForwardRef):
        return str(type_.__forward_arg__)

    return type_
