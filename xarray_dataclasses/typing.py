__all__ = ["Attr", "Coord", "Data", "Name"]


# standard library
from dataclasses import Field
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    Generic,
    cast,
    Dict,
    ForwardRef,
    List,
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
    Final,
    get_args,
    get_origin,
    Literal,
    Protocol,
)


# constants (internal)
class Xarray(Enum):
    """Identifiers for type hints of xarray-dataclasses."""

    ATTR = auto()  #: Attribute member of DataArray or Dataset.
    COORD = auto()  #: Coordinate member of DataArray or Dataset.
    DATA = auto()  #: Data of DataArray or variable of Dataset.
    NAME = auto()  #: Name of DataArray.

    def annotates(self, type_: Any) -> bool:
        """Check if type is annotated by the identifier."""
        args = get_args(type_)
        return len(args) > 1 and self in args[1:]


# type hints (internal)
D = TypeVar("D", covariant=True)
T = TypeVar("T", covariant=True)
DS = TypeVar("DS", bound=xr.Dataset)

DTypeLike = Union[np.dtype, type, str, None]
FieldDict = Dict[str, Field]
NoneType: Final[type] = type(None)


class ndarray(Protocol[T]):
    """Protocol version of numpy.ndarray."""

    __array__: Callable[..., np.ndarray]


class DataArray(Protocol[D, T]):
    """Protocol version of xarray.DataArray."""

    __array__: Callable[..., np.ndarray]


class DataClass(Protocol):
    """Type hint for dataclass instance."""

    __init__: Callable[..., None]
    __dataclass_fields__: FieldDict


class WithClass(Generic[DS]):
    __dataset_class__: Type[DS]


class DataClassX(DataClass, WithClass[DS]):
    ...


DataArrayLike = Union[DataArray[D, T], ndarray[T], Sequence[T], T]


# type hints (public)
Attr = Annotated[T, Xarray.ATTR]
"""Type hint for an attribute member of DataArray or Dataset.

Examples:
    ::

        from typing import Literal
        from xarray_dataclasses import dataarrayclass, Data, Attr


        X = Literal["x"]
        Y = Literal["y"]


        @dataarrayclass
        class Image:
            data: Data[tuple[X, Y], float]
            dpi: Attr[int] = 300

"""

Coord = Annotated[DataArrayLike[D, T], Xarray.COORD]
"""Type hint for a coordinate member of DataArray or Dataset.

Examples:
    ::

        from typing import Literal
        from xarray_dataclasses import dataarrayclass, Data, Coord


        X = Literal["x"]
        Y = Literal["y"]


        @dataarrayclass
        class Image:
            data: Data[tuple[X, Y], float]
            weight: Coord[tuple[X, Y], float] = 1.0
            x: Coord[X, int] = 0
            y: Coord[Y, int] = 0

"""

Data = Annotated[DataArrayLike[D, T], Xarray.DATA]
"""Type hint for data of DataArray or variable of Dataset.

Examples:
    ::

        from typing import Literal
        from xarray_dataclasses import dataarrayclass, Data


        X = Literal["x"]
        Y = Literal["y"]


        @dataarrayclass
        class Image:
            data: Data[tuple[X, Y], float]

    ::

        from typing import Literal
        from xarray_dataclasses import datasetclass, Data


        X = Literal["x"]
        Y = Literal["y"]


        @datasetclass
        class Images:
            red: Data[tuple[X, Y], float]
            green: Data[tuple[X, Y], float]
            blue: Data[tuple[X, Y], float]

"""

Name = Annotated[T, Xarray.NAME]
"""Type hint for a name of DataArray.

Examples:
    ::

        from typing import Literal
        from xarray_dataclasses import dataarrayclass, Data, Name


        X = Literal["x"]
        Y = Literal["y"]


        @dataarrayclass
        class Image:
            data: Data[tuple[X, Y], float]
            name: Name[str] = "default"

"""


# runtime functions (internal)
def is_attr(type_: Any) -> bool:
    """Check if type is Attr[T]."""
    return Xarray.ATTR.annotates(type_)


def is_coord(type_: Any) -> bool:
    """Check if type is Coord[D, T]."""
    return Xarray.COORD.annotates(type_)


def is_data(type_: Any) -> bool:
    """Check if type is Data[D, T]."""
    return Xarray.DATA.annotates(type_)


def is_name(type_: Any) -> bool:
    """Check if type is Name[T]."""
    return Xarray.NAME.annotates(type_)


def get_dims(type_: Type[DataArrayLike[D, T]]) -> Tuple[str, ...]:
    """Extract dimensions (dims) from DataArrayLike[D, T]."""
    if get_origin(type_) is Annotated:
        type_ = get_args(type_)[0]

    dims_ = get_args(get_args(type_)[0])[0]

    if get_origin(dims_) is tuple:
        dims_ = get_args(dims_)
    else:
        dims_ = (dims_,)

    dims: List[str] = []

    for dim_ in dims_:
        if dim_ == () or dim_ is NoneType:
            continue

        if isinstance(dim_, ForwardRef):
            dims.append(dim_.__forward_arg__)
            continue

        if get_origin(dim_) is Literal:
            dims.append(str(get_args(dim_)[0]))
            continue

        raise TypeError("Could not extract dimension.")

    return tuple(dims)


def get_dtype(type_: Type[DataArrayLike[D, T]]) -> DTypeLike:
    """Extract a data type (dtype) from DataArrayLike[D, T]."""
    if get_origin(type_) is Annotated:
        type_ = get_args(type_)[0]

    dtype_ = get_args(get_args(type_)[0])[1]

    if dtype_ is Any:
        return None

    if isinstance(dtype_, ForwardRef):
        return dtype_.__forward_arg__

    if get_origin(dtype_) is Literal:
        return get_args(dtype_)[0]

    return cast(type, dtype_)
