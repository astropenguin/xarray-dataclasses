__all__ = ["Attr", "Coord", "Data", "Name"]


# standard library
from enum import auto, Enum
from typing import (
    Any,
    ForwardRef,
    Hashable,
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
from typing_extensions import (
    Annotated,
    get_args,
    get_origin,
    Literal,
    Protocol,
)


# constants (internal)
class Xarray(Enum):
    """Identifiers of type hints for xarray."""

    ATTR = auto()  #: Attribute member of DataArray or Dataset.
    COORD = auto()  #: Coordinate member of DataArray or Dataset.
    DATA = auto()  #: Data of DataArray or variable of Dataset.
    NAME = auto()  #: Name of DataArray.


# type variables (internal)
T = TypeVar("T", covariant=True)  #: Type variable for data types.
D = TypeVar("D", covariant=True)  #: Type variable for dimensions.


# type hints (internal)
class ndarray(Protocol[T]):
    """Protocol version of numpy.ndarray."""

    __class__: Type[np.ndarray]  # type: ignore


class DataArray(Protocol[T, D]):
    """Protocol version of xarray.DataArray."""

    __class__: Type[xr.DataArray]  # type: ignore


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


def get_dims(type_: Type[DataArrayLike]) -> Tuple[Hashable, ...]:
    """Extract dimensions from DataArrayLike[T, D]."""
    if get_origin(type_) is Annotated:
        type_ = get_args(type_)[0]

    dtype, dims = get_args(get_args(type_)[0])

    if isinstance(dims, ForwardRef):
        return (dims.__forward_arg__,)

    if get_origin(dims) is Literal:
        return get_args(dims)

    if get_origin(dims) is tuple:
        return tuple(get_dims(dim)[0] for dim in get_args(dims))

    raise TypeError("Invalid type for extracting dims.")


def get_dtype(type_: Type[DataArrayLike]) -> Optional[np.dtype]:
    """Extract data type from DataArrayLike[T, D]."""
    if get_origin(type_) is Annotated:
        type_ = get_args(type_)[0]

    dtype, dims = get_args(get_args(type_)[0])

    if dtype is Any or dtype is None:
        return None

    if get_origin(dtype) is Literal:
        return np.dtype(get_args(dtype)[0])

    return np.dtype(dtype)


# helper functions (internal)
def _has_xarray_id(type_: Any, id: Xarray) -> bool:
    """Check if type has given identifier of xarray."""
    args = get_args(type_)
    return (len(args) > 1) and (args[1] is id)
