__all__ = ["Attr", "Coord", "Coordof", "Data", "Dataof", "Other"]


# standard library
import types
from dataclasses import Field
from enum import Enum, auto
from itertools import chain
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generic,
    Hashable,
    Iterable,
    Optional,
    Tuple,
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
    get_args,
    get_origin,
    get_type_hints,
)


# type hints (private)
P = ParamSpec("P")
T = TypeVar("T")
TDataClass = TypeVar("TDataClass", bound="DataClass[Any]")
TDataArray = TypeVar("TDataArray", bound=xr.DataArray)
TDataset = TypeVar("TDataset", bound=xr.Dataset)
TDims = TypeVar("TDims")
TDType = TypeVar("TDType")
TXarray = TypeVar("TXarray", bound="Xarray")
Xarray = Union[xr.DataArray, xr.Dataset]


class DataClass(Protocol[P]):
    """Type hint for dataclass objects."""

    __dataclass_fields__: Dict[str, "Field[Any]"]

    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        ...


class XarrayClass(Protocol[P, TXarray]):
    """Type hint for dataclass objects with a xarray factory."""

    __dataclass_fields__: Dict[str, "Field[Any]"]
    __xarray_factory__: Callable[..., TXarray]

    def __init__(self, *args: P.args, **kwargs: P.kwargs) -> None:
        ...


class Dims(Generic[TDims]):
    """Empty class for storing type of dimensions."""

    pass


class Tag(Enum):
    """Annotations for typing dataclass fields."""

    ATTR = auto()
    """Annotation for attribute fields."""

    COORD = auto()
    """Annotation for coordinate fields."""

    DATA = auto()
    """Annotation for data fields."""

    OTHER = auto()
    """Annotation for other fields."""

    @classmethod
    def annotates(cls, tp: Any) -> bool:
        """Check if any tag annotates a type hint."""
        return any(isinstance(arg, cls) for arg in get_args(tp))


# type hints (public)
Attr = Annotated[T, Tag.ATTR]
"""Type hint for attribute fields (``Attr[T]``)."""

Coord = Annotated[Union[Dims[TDims], Collection[TDType]], Tag.COORD]
"""Type hint for coordinate fields (``Coord[TDims, TDType]``)."""

Coordof = Annotated[TDataClass, Tag.COORD]
"""Type hint for coordinate fields (``Coordof[TDataClass, TDType]``)."""

Data = Annotated[Union[Dims[TDims], Collection[TDType]], Tag.DATA]
"""Type hint for data fields (``Data[TDims, TDType]``)."""

Dataof = Annotated[TDataClass, Tag.DATA]
"""Type hint for data fields (``Dataof[TDataClass, TDType]``)."""

Other = Annotated[T, Tag.OTHER]
"""Type hint for other fields (``Other[T]``)."""


# runtime functions
def deannotate(tp: Any) -> Any:
    """Recursively remove annotations in a type hint."""

    class Temporary:
        __annotations__ = dict(tp=tp)

    return get_type_hints(Temporary)["tp"]


def find_annotated(tp: Any) -> Iterable[Any]:
    """Generate all annotated types in a type hint."""
    args = get_args(tp)

    if get_origin(tp) is Annotated:
        yield tp
        yield from find_annotated(args[0])
    else:
        yield from chain(*map(find_annotated, args))


def get_annotated(tp: Any) -> Any:
    """Extract the first tag-annotated type."""
    for annotated in filter(Tag.annotates, find_annotated(tp)):
        return deannotate(annotated)

    raise TypeError("Could not find any tag-annotated type.")


def get_annotations(tp: Any) -> Tuple[Any, ...]:
    """Extract annotations of the first tag-annotated type."""
    for annotated in filter(Tag.annotates, find_annotated(tp)):
        return get_args(annotated)[1:]

    raise TypeError("Could not find any tag-annotated type.")


def get_dims(tp: Any) -> Optional[Tuple[str, ...]]:
    """Extract dimensions if found or return None."""
    try:
        dims = get_args(get_args(get_annotated(tp))[0])[0]
    except (IndexError, TypeError):
        return None

    args = get_args(dims)
    origin = get_origin(dims)

    if args == () or args == ((),):
        return ()

    if origin is Literal:
        return (str(args[0]),)

    if not (origin is tuple or origin is Tuple):
        raise TypeError(f"Could not find any dims in {tp!r}.")

    if not all(get_origin(arg) is Literal for arg in args):
        raise TypeError(f"Could not find any dims in {tp!r}.")

    return tuple(str(get_args(arg)[0]) for arg in args)


def get_dtype(tp: Any) -> Optional[str]:
    """Extract a data type if found or return None."""
    try:
        dtype = get_args(get_args(get_annotated(tp))[1])[0]
    except (IndexError, TypeError):
        return None

    if dtype is Any or dtype is type(None):
        return None

    if is_union_type(dtype):
        dtype = get_args(dtype)[0]

    if get_origin(dtype) is Literal:
        dtype = get_args(dtype)[0]

    return np.dtype(dtype).name


def get_name(tp: Any, default: Hashable = None) -> Hashable:
    """Extract a name if found or return given default."""
    try:
        name = get_annotations(tp)[1]
    except (IndexError, TypeError):
        return default

    if name is Ellipsis:
        return default

    try:
        hash(name)
    except TypeError:
        raise ValueError("Could not find any valid name.")

    return name


def get_tag(tp: Any, default: Tag = Tag.OTHER) -> Tag:
    """Extract a tag if found or return given default."""
    try:
        return get_annotations(tp)[0]
    except (IndexError, TypeError):
        return default


def is_union_type(tp: Any) -> bool:
    """Check if a type hint is a union type."""
    if get_origin(tp) is Union:
        return True

    UnionType = getattr(types, "UnionType", None)
    return UnionType is not None and isinstance(tp, UnionType)
