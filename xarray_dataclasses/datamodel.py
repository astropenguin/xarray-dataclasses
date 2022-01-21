__all__ = ["DataModel"]


# standard library
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union, cast


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import Literal


# submodules
from .typing import ArrayLike, DataType, FieldType, Dims, Dtype


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
Reference = Optional[DataType]


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

    def __call__(self, reference: Reference = None) -> xr.DataArray:
        ...


@dataclass(frozen=True)
class DataModel:
    entries: Dict[str, Union[AttrEntry, DataEntry]]
    options: Dict[str, Any]

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
    def from_dataclass(cls, dataclass: Any) -> "DataModel":
        ...


# runtime functions
def get_entry_type(type: Any) -> EntryType:
    """Parse a type and return a corresponding entry type."""
    if FieldType.ATTR.annotates(type):
        return EntryType.ATTR

    if FieldType.COORD.annotates(type):
        return EntryType.COORD

    if FieldType.COORDOF.annotates(type):
        return EntryType.COORD

    if FieldType.DATA.annotates(type):
        return EntryType.DATA

    if FieldType.DATAOF.annotates(type):
        return EntryType.DATA

    if FieldType.NAME.annotates(type):
        return EntryType.NAME

    raise TypeError("Could not find any FieldType annotation.")


def typedarray(
    data: Any,
    dims: Dims,
    dtype: Dtype,
    reference: Reference = None,
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
