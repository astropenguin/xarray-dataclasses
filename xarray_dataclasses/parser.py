__all__ = ["parse"]


# standard library
from dataclasses import dataclass, Field
from typing import Any, Dict, List, Optional, Tuple, TypeVar


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import get_args, Protocol


# submodules
from .typing import (
    ArrayLike,
    DataClassLike,
    FieldType,
    get_dims,
    get_dtype,
    unannotate,
)


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
        t_dims, t_dtype = get_args(get_args(unannotate(field.type))[0])
        type = dict(dims=get_dims(t_dims), dtype=get_dtype(t_dtype))
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
def to_dataarray(
    data: Any,
    dims: Dims,
    dtype: Dtype,
    template: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """Convert data to a DataArray with given dims and dtype."""
    if not isinstance(data, ArrayLike):
        data = np.asarray(data)

    if dtype is not None:
        data = data.astype(dtype, copy=True)

    if template is not None:
        template = to_subspace(template, dims)

    if data.ndim == len(dims):
        dataarray = xr.DataArray(data, dims=dims)

        if template is None:
            return dataarray
        else:
            return dataarray.broadcast_like(template)

    if data.ndim == 0 and template is not None:
        return xr.DataArray(data).expand_dims(template.sizes)

    raise ValueError(f"Could not convert {data} with {dims} and {dtype}.")


def to_subspace(dataarray: xr.DataArray, dims: Dims) -> xr.DataArray:
    """Return the subspace of a DataArray with given dims."""
    indexers = {dim: 0 for dim in dataarray.dims if dim not in dims}
    return dataarray.isel(indexers)
