__all__ = ["Attr", "Coord", "Data", "Name"]


# standard library
from enum import auto, Enum
from functools import wraps
from typing import ForwardRef, Generic, Sequence, Tuple, TypeVar, Union


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Annotated, get_args, get_origin, Literal, TypeAlias


# constants
class Xarray(Enum):
    """Identification for public type hints."""

    ATTR = auto()  #: Attribute member of DataArray or Dataset.
    COORD = auto()  #: Coordinate member of DataArray or Dataset.
    DATA = auto()  #: Data of DataArray or variable of Dataset.
    NAME = auto()  #: Name of DataArray.


# type variables
T = TypeVar("T")  #: Type variable for data types.
D = TypeVar("D")  #: Type valiable for dimensions.


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
def is_dataarraylike(alias: TypeAlias) -> bool:
    """Check if type alias is DataArrayLike[...]."""
    if not get_args(alias):
        return False

    alias, *metadata = get_args(alias)
    return metadata[0] in (Xarray.COORD, Xarray.DATA)


def get_dims(alias: TypeAlias) -> Tuple[str, ...]:
    """Extract dimensions from type alias."""
    if not is_dataarraylike(alias):
        raise ValueError("Invalid type hint.")

    dtype, dims = get_args(get_args(get_args(alias)[0])[0])

    # dims -> ForwardRef(string)
    if isinstance(dims, ForwardRef):
        return tuple(dims.__forward_arg__.split(","))

    # dims -> Literal[string]
    if get_origin(dims) == Literal:
        return get_args(dims)

    # dims -> Tuple[Literal[string], ...]
    return tuple(get_args(dim)[0] for dim in get_args(dims))


def get_dtype(alias: TypeAlias) -> np.dtype:
    """Extract data type from type alias."""
    if not is_dataarraylike(alias):
        raise ValueError("Invalid type hint.")

    dtype, dims = get_args(get_args(get_args(alias)[0])[0])

    # dtype -> ForwardRef(string)
    if isinstance(dtype, ForwardRef):
        return np.dtype(dtype.__forward_arg__)

    # dtype -> type itself
    return np.dtype(dtype)
