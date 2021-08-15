__all__ = ["parse"]


# standard library
from dataclasses import dataclass, Field
from typing import Any, List, Optional, Type, Union


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import get_args


# submodules
from .typing import (
    ArrayLike,
    DataClass,
    Dims,
    Dtype,
    FieldType,
    get_dims,
    get_dtype,
    TDataArray,
    TDataset,
    unannotate,
)


# type hints
DataClassLike = Union[Type[DataClass], DataClass]
Reference = Union[xr.DataArray, xr.Dataset]


# dataclasses
@dataclass(frozen=True)
class DataArray:
    """Parsed DataArray information."""

    name: str
    """Variable name for a DataArray."""
    value: Any
    """Value to be assigned to a DataArray."""
    dims: Dims
    """Dimensions of a DataArray."""
    dtype: Dtype
    """Data type of a DataArray."""

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "DataArray":
        """Create an instance from a dataclass field."""
        t_dims, t_dtype = get_args(get_args(unannotate(field.type))[0])
        return cls(field.name, value, get_dims(t_dims), get_dtype(t_dtype))

    def __call__(self, reference: Optional[Reference] = None) -> xr.DataArray:
        """Return a typed DataArray from the parsed information."""
        return typed_dataarray(self.value, self.dims, self.dtype, reference)


@dataclass(frozen=True)
class GeneralType:
    """Parsed general-type information."""

    name: str
    """Variable name for a type."""
    value: Any
    """Value to be assigned to a type."""
    type: Type[Any]
    """Type or type-hint object."""

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "GeneralType":
        """Create an instance from a dataclass field."""
        return cls(field.name, value, unannotate(field.type))

    def __call__(self) -> Any:
        """Return the value (without casting)."""
        return self.value


@dataclass(frozen=True)
class DataStructure:
    """Parsed dataclass information."""

    attr: List[GeneralType]
    """Parsed attr-type information."""
    coord: List[DataArray]
    """Parsed coord-type information."""
    data: List[DataArray]
    """Parsed data-type information."""
    name: List[GeneralType]
    """Parsed name-type information."""

    @classmethod
    def from_dataclass(cls, dataclass: DataClassLike) -> "DataStructure":
        """Create an instance from a dataclass."""
        attr: List[GeneralType] = []
        coord: List[DataArray] = []
        data: List[DataArray] = []
        name: List[GeneralType] = []

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

    def to_dataarray(
        self,
        dataarray_factory: Type[TDataArray] = xr.DataArray,
    ) -> TDataArray:
        """Return a DataArray from the parsed information."""
        return to_dataarray(self, dataarray_factory)

    def to_dataset(
        self,
        dataset_factory: Type[TDataset] = xr.Dataset,
    ) -> TDataset:
        """Create a Dataset from the parsed information."""
        return to_dataset(self, dataset_factory)


# runtime functions
def parse(dataclass: DataClassLike) -> DataStructure:
    """Create a data structure from a dataclass."""
    return DataStructure.from_dataclass(dataclass)


def to_dataarray(
    data_structure: DataStructure,
    dataarray_factory: Type[TDataArray] = xr.DataArray,
) -> TDataArray:
    """Create a DataArray from a parsed dataclass."""
    dataarray = dataarray_factory(data_structure.data[0]())

    for coord in data_structure.coord:
        dataarray.coords.update({coord.name: coord(dataarray)})

    for attr in data_structure.attr:
        dataarray.attrs.update({attr.name: attr()})

    for name in data_structure.name:
        dataarray.name = name()

    return dataarray


def to_dataset(
    data_structure: DataStructure,
    dataset_factory: Type[TDataset] = xr.Dataset,
) -> TDataset:
    """Create a Dataset from a parsed dataclass."""
    dataset = dataset_factory()

    for data in data_structure.data:
        dataset.update({data.name: data()})

    for coord in data_structure.coord:
        dataset.coords.update({coord.name: coord(dataset)})

    for attr in data_structure.attr:
        dataset.attrs.update({attr.name: attr()})

    return dataset


def typed_dataarray(
    data: Any,
    dims: Dims,
    dtype: Dtype,
    reference: Optional[Reference] = None,
) -> xr.DataArray:
    """Convert data to a DataArray with given dims and dtype."""
    if not isinstance(data, ArrayLike):
        data = np.asarray(data)

    if dtype is not None:
        data = data.astype(dtype, copy=True)

    if data.ndim == len(dims):
        dataarray = xr.DataArray(data, dims=dims)
    elif data.ndim == 0 and reference is not None:
        dataarray = xr.DataArray(data)
    else:
        raise ValueError(f"Could not convert {data} with {dims}.")

    if reference is None:
        return dataarray
    else:
        return dataarray.broadcast_like(subspace(reference, dims))


def subspace(reference: Reference, dims: Dims) -> Reference:
    """Return the subspace of a DataArray or a Dataset."""
    diff_dims = set(reference.dims) - set(dims)
    return reference.isel({dim: 0 for dim in diff_dims})
