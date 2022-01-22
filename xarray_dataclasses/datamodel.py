__all__ = ["DataModel"]


# standard library
from dataclasses import dataclass, field, is_dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import (
    Literal,
    ParamSpec,
    get_args,
    get_origin,
    get_type_hints,
)


# submodules
from .typing import (
    ArrayLike,
    DataClass,
    DataType,
    FieldType,
    Dims,
    Dtype,
)


# constants
class EntryType(Enum):
    """Type labels for datamodel entries."""

    ATTR = auto()
    """Type label for attribute entries."""

    COORD = auto()
    """Type label for coordinate entries."""

    DATA = auto()
    """Type label for data entries."""

    NAME = auto()
    """Type label for name entries."""


# type hints
P = ParamSpec("P")
DataClassLike = Union[Type[DataClass[P]], DataClass[P]]
Entries = Dict[str, Union["AttrEntry", "DataEntry"]]


# dataclasses
@dataclass(frozen=True)
class AttrEntry:
    type: Literal[EntryType.ATTR, EntryType.NAME]
    name: Any = None
    default: Any = None

    def __call__(self) -> Any:
        ...


@dataclass(frozen=True)
class DataEntry:
    type: Literal[EntryType.COORD, EntryType.DATA]
    dims: Any = None
    dtype: Any = None
    base: Any = None
    name: Any = None
    default: Any = None

    def __call__(self, reference: Optional[DataType] = None) -> xr.DataArray:
        ...


@dataclass(frozen=True)
class DataModel:
    entries: Entries = field(default_factory=dict)

    @property
    def attrs(self) -> List[AttrEntry]:
        ...

    @property
    def coords(self) -> List[DataEntry]:
        ...

    @property
    def data_vars(self) -> List[DataEntry]:
        ...

    @property
    def data_vars_items(self) -> List[Tuple[str, DataEntry]]:
        ...

    @property
    def names(self) -> List[AttrEntry]:
        ...

    @classmethod
    def from_dataclass(cls, dataclass: DataClassLike[P]) -> "DataModel":
        """Create a datamodel from a dataclass or its object."""
        eval_dataclass(dataclass)
        return cls(get_entries(dataclass))


# runtime functions
def eval_dataclass(dataclass: DataClassLike[P]) -> None:
    """Evaluate field types of a dataclass."""
    if not is_dataclass(dataclass):
        raise TypeError("Not a dataclass or its object.")

    fields = dataclass.__dataclass_fields__.values()

    # do nothing if field types are already evaluated
    if not any(isinstance(field.type, str) for field in fields):
        return

    # otherwise, replace field types with evaluated types
    if not isinstance(dataclass, type):
        dataclass = type(dataclass)

    types = get_type_hints(dataclass, include_extras=True)

    for field in fields:
        field.type = types[field.name]


def get_dims(type_: Any) -> Dims:
    """Parse a type and return dims."""
    args = get_args(type_)
    origin = get_origin(type_)

    if origin not in (tuple, Tuple, Literal):
        raise ValueError(f"Could not convert {type_!r} to dims.")

    if args == () or args == ((),):
        return ()

    if origin is Literal:
        return (args[0],)

    return tuple(get_dims(arg)[0] for arg in args)


def get_dtype(type_: Any) -> Dtype:
    """Parse a type and return dtype."""
    if type_ is Any or type_ is type(None):
        return None

    if isinstance(type_, type):
        return type_.__name__

    if get_origin(type_) is Literal:
        return get_args(type_)[0]

    raise ValueError(f"Could not convert {type_!r} to dtype.")


def get_entries(dataclass: DataClassLike[P]) -> Entries:
    """Parse a dataclass and return entries."""
    entries: Entries = {}

    for field in dataclass.__dataclass_fields__.values():
        default = getattr(dataclass, field.name, field.default)
        etype = get_entry_type(field.type)
        rtype = get_repr_type(field.type)

        if etype == EntryType.ATTR or etype == EntryType.NAME:
            entries[field.name] = AttrEntry(
                type=etype,
                name=field.name,
                default=default,
            )
        elif is_dataclass(rtype):
            entries[field.name] = DataEntry(
                type=etype,
                base=rtype,
                name=field.name,
                default=default,
            )
        else:
            entries[field.name] = DataEntry(
                type=etype,
                dims=get_dims(get_args(rtype)[0]),
                dtype=get_dtype(get_args(rtype)[1]),
                name=field.name,
                default=default,
            )

    return entries


def get_entry_type(type_: Any) -> EntryType:
    """Parse a type and return a corresponding entry type."""
    if FieldType.ATTR.annotates(type_):
        return EntryType.ATTR

    if FieldType.COORD.annotates(type_):
        return EntryType.COORD

    if FieldType.COORDOF.annotates(type_):
        return EntryType.COORD

    if FieldType.DATA.annotates(type_):
        return EntryType.DATA

    if FieldType.DATAOF.annotates(type_):
        return EntryType.DATA

    if FieldType.NAME.annotates(type_):
        return EntryType.NAME

    raise TypeError("Could not find any FieldType annotations.")


def get_repr_type(type_: Any) -> Any:
    """Parse a type and return an representative type."""

    class Temporary:
        __annotations__ = dict(type=type_)

    unannotated = get_type_hints(Temporary)["type"]
    inner_types = get_args(unannotated)

    return inner_types[0] if inner_types else unannotated


def typedarray(
    data: Any,
    dims: Dims,
    dtype: Dtype,
    reference: Optional[DataType] = None,
) -> xr.DataArray:
    """Create a DataArray object with given dims and dtype.

    Args:
        data: Data to be converted to the DataArray object.
        dims: Dimensions of the DataArray object.
        dtype: Data type of the DataArray object.
        reference: DataArray or Dataset object as a reference of shape.

    Returns:
        DataArray object with given dims and dtype.

    """
    if isinstance(data, ArrayLike):
        array = cast(np.ndarray, data)
    else:
        array = np.asarray(data)

    if dtype is not None:
        array = array.astype(dtype, copy=False)

    if array.ndim == len(dims):
        dataarray = xr.DataArray(array, dims=dims)
    elif array.ndim == 0 and reference is not None:
        dataarray = xr.DataArray(array)
    else:
        raise ValueError(
            "Could not create a DataArray object from data. "
            f"Mismatch between shape {array.shape} and dims {dims}."
        )

    if reference is None:
        return dataarray

    diff_dims = set(reference.dims) - set(dims)
    subspace = reference.isel({dim: 0 for dim in diff_dims})
    return dataarray.broadcast_like(subspace)
