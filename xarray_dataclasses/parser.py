# standard library
from dataclasses import dataclass, Field
from itertools import chain
from typing import Any, ForwardRef, List, Optional, Tuple, Type, TypeVar, Union


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Annotated, get_args, get_origin, Literal


# submodules
from .typing import ArrayLike, DataClass, FieldType
from .utils import make_generic


# for Python 3.7 and 3.8
make_generic(Field)


# type hints
DataClassLike = Union[Type[DataClass], DataClass]
Dims = Tuple[str, ...]
Dtype = Optional[str]
NoneType = type(None)
ParsedType = str
T = TypeVar("T")


# dataclasses
@dataclass(frozen=True)
class ParsedDataArray:
    """Dataclass for parsed DataArray information."""

    dims: Dims
    """Parsed dimensions of DataArray."""
    dtype: Dtype
    """Parsed data type of DataArray."""

    @classmethod
    def from_type(cls, type: Type[Any]) -> "ParsedDataArray":
        """Create an instance from a Data or Coord type."""
        dims, dtype = get_args(get_args(unannotate(type))[0])
        return cls(parse_dims(dims), parse_dtype(dtype))

    def to_dataarray(self, data: Any) -> xr.DataArray:
        """Convert data to a DataArray with given dims and dtype."""
        return to_dataarray(data, self.dims, self.dtype)


@dataclass(frozen=True)
class ParsedField:
    """Dataclass for parsed field information."""

    name: str
    """Name of a field."""
    type: Union[ParsedDataArray, ParsedType]
    """Parsed type of a field."""
    value: Any
    """Assigned value of a field."""

    def __post_init__(self):
        """Remove Annotated type from ``type``."""
        super().__setattr__("type", unannotate(self.type))

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "ParsedField":
        """Create an instance from a field and a value."""

        if FieldType.COORD.annotates(field.type):
            type = ParsedDataArray.from_type(field.type)
            return cls(field.name, type, value)

        if FieldType.DATA.annotates(field.type):
            type = ParsedDataArray.from_type(field.type)
            return cls(field.name, type, value)

        type = f"{field.type.__module__}.{field.type.__qualname__}"
        return cls(field.name, type, value)


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
            parsed_field = ParsedField.from_field(field, value)

            if FieldType.ATTR.annotates(field):
                attr.append(parsed_field)
            elif FieldType.COORD.annotates(field):
                coord.append(parsed_field)
            elif FieldType.DATA.annotates(field):
                data.append(parsed_field)
            elif FieldType.NAME.annotates(field):
                name.append(parsed_field)

        return cls(attr, coord, data, name)


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
