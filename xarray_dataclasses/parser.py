__all__ = ["parse"]


# standard library
from dataclasses import InitVar, dataclass, Field
from typing import Any, List, Optional, Type, Union


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Protocol, TypedDict


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
from .utils import resolve_class


# type hints
DataClassLike = Union[Type[DataClass], DataClass]
Reference = Union[xr.DataArray, xr.Dataset]
Types = TypedDict("Types", dims=Dims, dtype=Dtype)


# dataclasses
@dataclass(frozen=True)
class GeneralType:
    """Representation for general-type variables."""

    name: str  #: Name of a variable.
    type: str  #: Type (full path) of a variable.
    value: Any  #: Value to be assigned to a vabiable.

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "GeneralType":
        """Create an instance from a dataclass field."""
        type = resolve_class(unannotate(field.type))
        return cls(field.name, type, value)


class DataArrayType(Protocol):
    """Reperesentation for DataArray variables."""

    name: str  #: Name of a variable
    type: Union[Types, str]  #: Type (full path or dict) of a variable.
    value: Any  #: Value to be assigned to a variable.

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "DataArrayType":
        """Create an instance from a dataclass field."""
        ...

    def __call__(self, reference: Optional[Reference] = None) -> xr.DataArray:
        """Create a typed DataArray from the representation."""


@dataclass(frozen=True)
class DataType:
    """Representation for DataArray variables with dims and dtypes."""

    name: str  #: Name of a variable
    type: Types  #: Type (dict) of a variable.
    value: Any  #: Value to be assigned to a variable.

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "DataType":
        """Create an instance from a dataclass field."""
        type = unannotate(field.type).__args__[0]

        types: Types = {
            "dims": get_dims(type.__args__[0]),
            "dtype": get_dtype(type.__args__[1]),
        }

        return cls(field.name, types, value)

    def __call__(self, reference: Optional[Reference] = None) -> xr.DataArray:
        """Create a typed DataArray from the representation."""
        dims, dtype = self.type["dims"], self.type["dtype"]
        return typedarray(self.value, dims, dtype, reference)


@dataclass(frozen=True)
class ClassType:
    """Representation for DataArray variables with dataclass."""

    name: str  #: Name of a variable
    type: str  #: Type (full path) of a variable.
    value: Any  #: Value to be assigned to a variable.
    dataclass: InitVar[Type[DataClass]]  #: Runtime dataclass of a variable.

    def __post_init__(self, dataclass: InitVar[Type[DataClass]]) -> None:
        super().__setattr__("dataclass", dataclass)

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "ClassType":
        """Create an instance from a dataclass field."""
        type = unannotate(field.type).__args__[0]
        return cls(field.name, resolve_class(type), value, type)

    def __call__(self, reference: Optional[Reference] = None) -> xr.DataArray:
        """Create a typed DataArray from the representation."""
        if isinstance(self.value, self.dataclass):
            return parse(self.value).to_dataarray(reference)
        else:
            return parse(self.dataclass(self.value)).to_dataarray(reference)


@dataclass(frozen=True)
class Structure:
    """Structure of a dataclass or its instance."""

    attr: List[GeneralType]  #: Representations of attr-type variables.
    coord: List[DataArrayType]  #: Representations of coord-type variables.
    data: List[DataArrayType]  #: Representations of data-type variables.
    name: List[GeneralType]  #: Representations of name-type variables.

    @classmethod
    def from_dataclass(cls, dataclass: DataClassLike) -> "Structure":
        """Create an instance from a dataclass or its instance."""
        attr: List[GeneralType] = []
        coord: List[DataArrayType] = []
        data: List[DataArrayType] = []
        name: List[GeneralType] = []

        for field in dataclass.__dataclass_fields__.values():
            value = getattr(dataclass, field.name, field.default)

            if FieldType.ATTR.annotates(field.type):
                attr.append(GeneralType.from_field(field, value))
            elif FieldType.COORD.annotates(field.type):
                coord.append(DataType.from_field(field, value))
            elif FieldType.COORDOF.annotates(field.type):
                coord.append(ClassType.from_field(field, value))
            elif FieldType.DATA.annotates(field.type):
                data.append(DataType.from_field(field, value))
            elif FieldType.DATAOF.annotates(field.type):
                data.append(ClassType.from_field(field, value))
            elif FieldType.NAME.annotates(field.type):
                name.append(GeneralType.from_field(field, value))

        return cls(attr, coord, data, name)

    def to_dataarray(
        self,
        reference: Optional[Reference] = None,
        dataarray_factory: Type[TDataArray] = xr.DataArray,
    ) -> TDataArray:
        """Create a typed DataArray from the structure."""
        return to_dataarray(self, reference, dataarray_factory)

    def to_dataset(
        self,
        reference: Optional[Reference] = None,
        dataset_factory: Type[TDataset] = xr.Dataset,
    ) -> TDataset:
        """Create a typed Dataset from the structure."""
        return to_dataset(self, reference, dataset_factory)


# runtime functions
def parse(dataclass: DataClassLike) -> Structure:
    """Create a structure from a dataclass."""
    return Structure.from_dataclass(dataclass)


def to_dataarray(
    structure: Structure,
    reference: Optional[Reference] = None,
    dataarray_factory: Type[TDataArray] = xr.DataArray,
) -> TDataArray:
    """Create a typed DataArray from a structure."""
    dataarray = dataarray_factory(structure.data[0](reference))

    for coord in structure.coord:
        dataarray.coords.update({coord.name: coord(dataarray)})

    for attr in structure.attr:
        dataarray.attrs.update({attr.name: attr.value})

    for name in structure.name:
        dataarray.name = name.value

    return dataarray


def to_dataset(
    structure: Structure,
    reference: Optional[Reference] = None,
    dataset_factory: Type[TDataset] = xr.Dataset,
) -> TDataset:
    """Create a typed Dataset from a structure."""
    dataset = dataset_factory()

    for data in structure.data:
        dataset.update({data.name: data(reference)})

    for coord in structure.coord:
        dataset.coords.update({coord.name: coord(dataset)})

    for attr in structure.attr:
        dataset.attrs.update({attr.name: attr.value})

    return dataset


def typedarray(
    data: Any,
    dims: Dims,
    dtype: Dtype,
    reference: Optional[Reference] = None,
) -> xr.DataArray:
    """Create a typed DataArray with given dims and dtype."""
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
        diff_dims = set(reference.dims) - set(dims)
        subspace = reference.isel({dim: 0 for dim in diff_dims})
        return dataarray.broadcast_like(subspace)
