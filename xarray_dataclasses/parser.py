__all__ = ["parse"]


# standard library
from dataclasses import dataclass, Field
from itertools import chain
from typing import Any, Dict, ForwardRef, List, Optional, Tuple, TypeVar


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Annotated, get_args, get_origin, Literal, Protocol


# submodules
from .typing import ArrayLike, DataClassLike, FieldType
from .utils import make_generic


# for Python 3.7 and 3.8
make_generic(Field)


# type hints
Dims = Tuple[str, ...]
Dtype = Optional[str]
NoneType = type(None)
T = TypeVar("T")


class ParsedField(Protocol):
    """Dataclass for parsed field information."""

    name: str
    """Name of a field."""
    type: Dict[str, Any]
    """Parsed type of a field."""
    value: Any
    """Assigned value of a field."""

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "ParsedField":
        """Create an instance from a field and a value."""
        ...

    def instantiate(self) -> Any:
        """Convert a value to an instance with given type information."""
        ...


# dataclasses
@dataclass(frozen=True)
class DataArray:
    """Dataclass for parsed DataArray information."""

    name: str
    """Name of a field."""
    type: Dict[str, Any]
    """Parsed type of a field."""
    value: Any
    """Assigned value of a field."""

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "DataArray":
        """Create an instance from a Coord/Data-type field and a value."""
        dims, dtype = get_args(get_args(unannotate(field.type))[0])
        type = dict(dims=parse_dims(dims), dtype=parse_dtype(dtype))
        return cls(field.name, type, value)

    def instantiate(self) -> xr.DataArray:
        """Convert a value to a DataArray with given dims and dtype."""
        return to_dataarray(self.value, **self.type)


@dataclass(frozen=True)
class GeneralType:
    """Dataclass for parsed general-type information."""

    name: str
    """Name of a field."""
    type: Dict[str, Any]
    """Parsed type of a field."""
    value: Any
    """Assigned value of a field."""

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "GeneralType":
        """Create an instance from a general-type field and a value."""
        type = dict(type=unannotate(field.type))
        return cls(field.name, type, value)

    def instantiate(self) -> Any:
        """Do not convert but just return a value."""
        return self.value


@dataclass(frozen=True)
class ParsedDataClass:
    """Dataclass for parsed dataclass or dataclass instance."""

    attr: List[ParsedField]
    """Parsed Attr-typed field(s) information."""
    coord: List[ParsedField]
    """Parsed Coord-typed field(s) information."""
    data: List[ParsedField]
    """Parsed Data-typed field(s) information."""
    name: List[ParsedField]
    """Parsed Name-typed field(s) information."""

    def __post_init__(self):
        """Validate the number of fields in ``data`` and ``name``."""
        if len(self.data) == 0:
            raise RuntimeError("Could not find any Data-typed fields.")

        if len(self.name) > 1:
            raise RuntimeError("Found more than one Name-typed fields.")

    @classmethod
    def from_dataclass(cls, dataclass: DataClassLike) -> "ParsedDataClass":
        """Create an instance from a dataclass or dataclass instance."""
        attr: List[ParsedField] = []
        coord: List[ParsedField] = []
        data: List[ParsedField] = []
        name: List[ParsedField] = []

        for field in dataclass.__dataclass_fields__.values():
            value = getattr(dataclass, field.name, field.default)

            if FieldType.ATTR.annotates(field.type):
                attr.append(GeneralType.from_field(field, value))
            elif FieldType.COORD.annotates(field.type):
                coord.append(DataArray.from_field(field, value))
            elif FieldType.DATA.annotates(field.type):
                data.append(DataArray.from_field(field, value))
            elif FieldType.NAME.annotates(field.type):
                name.append(GeneralType.from_field(field, value))

        return cls(attr, coord, data, name)


# main features
def parse(dataclass: DataClassLike) -> ParsedDataClass:
    """Parse a dataclass or dataclass instance."""
    return ParsedDataClass.from_dataclass(dataclass)


# helper features
def parse_dims(type_like: Any) -> Dims:
    """Parse a type-like object and get dims."""
    type_like = unannotate(type_like)

    if type_like == () or type_like is NoneType:
        return ()

    if isinstance(type_like, ForwardRef):
        return (type_like.__forward_arg__,)

    if isinstance(type_like, str):
        return (type_like,)

    origin = get_origin(type_like)
    args = get_args(type_like)

    if origin is tuple:
        return tuple(chain(*map(parse_dims, args)))

    if origin is Literal:
        return tuple(map(str, args))

    raise ValueError(f"Could not parse {type_like}.")


def parse_dtype(type_like: Any) -> Dtype:
    """Parse a type-like object and get dtype."""
    type_like = unannotate(type_like)

    if type_like is Any or type_like is NoneType:
        return None

    if isinstance(type_like, type):
        return type_like.__name__

    if isinstance(type_like, ForwardRef):
        return type_like.__forward_arg__

    if isinstance(type_like, str):
        return type_like

    origin = get_origin(type_like)
    args = get_args(type_like)

    if origin is Literal and len(args) == 1:
        return str(args[0])

    raise ValueError(f"Could not parse {type_like}.")


def to_dataarray(
    data: Any,
    dims: Dims,
    dtype: Dtype,
) -> xr.DataArray:
    """Convert data to a DataArray with given dims and dtype."""
    if not isinstance(data, ArrayLike):
        data = np.asarray(data)  # type: ignore

    if dtype is not None:
        data = data.astype(dtype, copy=True)  # type: ignore

    if data.ndim == 0:
        data = data.reshape([1] * len(dims))  # type: ignore

    return xr.DataArray(data, dims=dims)


def unannotate(obj: T) -> T:
    """Recursively remove Annotated types."""
    if get_origin(obj) is Annotated:
        obj = get_args(obj)[0]

    origin = get_origin(obj)

    if origin is None:
        return obj

    args = map(unannotate, get_args(obj))
    args = tuple(filter(None, args))

    try:
        return origin[args]
    except TypeError:
        import typing

        name = origin.__name__.capitalize()
        return getattr(typing, name)[args]
