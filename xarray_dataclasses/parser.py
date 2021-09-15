# standard library
from dataclasses import dataclass, field, Field, InitVar, is_dataclass
from typing import Any, Callable, Generic, List, TypeVar, Union


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import get_args, TypedDict


# submodules
from .typing import (
    ArrayLike,
    DataClass,
    Dims,
    Dtype,
    FieldType,
    get_dims,
    get_dtype,
    unannotate,
)
from .utils import resolve_class


# type hints
R = TypeVar("R")
RDataArray = TypeVar("RDataArray", bound=xr.DataArray)
RDataset = TypeVar("RDataset", bound=xr.Dataset)
Reference = Union[xr.DataArray, xr.Dataset, None]
Factory = Callable[[Any, Reference], R]


class DataArrayType(TypedDict):
    """Type hint for a DataArray type."""

    dims: Dims
    dtype: Dtype


# field models
def default_factory(value: R, reference: Reference) -> R:
    """Default factory that just returns an input value."""
    return value


@dataclass
class FieldModel(Generic[R]):
    """Base model for the dataclass fields."""

    name: str
    """Name of the field."""

    type: Any
    """Type of the field."""

    value: Any
    """Value assigned to the field."""

    factory: InitVar[Factory[R]] = default_factory
    """Factory function to create an object."""

    def __post_init__(self, factory: Factory[R]) -> None:
        self.factory = factory

    def __call__(self, reference: Reference = None) -> R:
        return self.factory(self.value, reference)


@dataclass
class Data(FieldModel[xr.DataArray]):
    """Model for the coord or data fields."""

    type: DataArrayType
    """Type of the field."""

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "Data":
        """Create a field model from a dataclass field and a value."""
        args = get_args(get_args(unannotate(field.type))[0])

        dims = get_dims(args[0])
        dtype = get_dtype(args[1])
        type: DataArrayType = {"dims": dims, "dtype": dtype}

        def factory(value: Any, reference: Reference) -> xr.DataArray:
            return typedarray(value, dims, dtype, reference)

        return cls(field.name, type, value, factory)


@dataclass
class Dataof(FieldModel[xr.DataArray]):
    """Model for the coordof or dataof fields."""

    type: str
    """Type of the field."""

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "Dataof":
        """Create a field model from a dataclass field and a value."""
        dataclass = get_args(unannotate(field.type))[0]
        type = resolve_class(dataclass)

        def factory(value: Any, reference: Reference) -> xr.DataArray:
            if not is_dataclass(value):
                value = dataclass(value)

            return DataModel.from_dataclass(value).to_dataarray(reference)

        return cls(field.name, type, value, factory)


@dataclass
class General(FieldModel[Any]):
    """Model for the attribute or name fields."""

    type: str
    """Type of the field."""

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "General":
        """Create a field model from a dataclass field and a value."""
        type = resolve_class(unannotate(field.type))
        return cls(field.name, type, value)


# data models
@dataclass(frozen=True)
class DataModel:
    """Model for dataclasses or their objects."""

    attr: List[General] = field(default_factory=list)
    """Model of the attribute fields."""

    coord: List[Union[Data, Dataof]] = field(default_factory=list)
    """Model of the coordinate fields."""

    data: List[Union[Data, Dataof]] = field(default_factory=list)
    """Model of the data fields."""

    name: List[General] = field(default_factory=list)
    """Model of the name fields."""

    @classmethod
    def from_dataclass(cls, dataclass: DataClass) -> "DataModel":
        """Create a data model from a dataclass or its object."""
        model = cls()

        for field_ in dataclass.__dataclass_fields__.values():
            value = getattr(dataclass, field_.name, field_.default)

            if FieldType.ATTR.annotates(field_.type):
                model.attr.append(General.from_field(field_, value))
            elif FieldType.COORD.annotates(field_.type):
                model.coord.append(Data.from_field(field_, value))
            elif FieldType.COORDOF.annotates(field_.type):
                model.coord.append(Dataof.from_field(field_, value))
            elif FieldType.DATA.annotates(field_.type):
                model.data.append(Data.from_field(field_, value))
            elif FieldType.DATAOF.annotates(field_.type):
                model.data.append(Dataof.from_field(field_, value))
            elif FieldType.NAME.annotates(field_.type):
                model.name.append(General.from_field(field_, value))

        return model

    def to_dataarray(
        self,
        reference: Reference = None,
        dataarray_factory: Callable[..., RDataArray] = xr.DataArray,
    ) -> RDataArray:
        """Create a DataArray object from the data model."""
        dataarray = dataarray_factory(self.data[0](reference))

        for coord in self.coord:
            dataarray.coords.update({coord.name: coord(dataarray)})

        for attr in self.attr:
            dataarray.attrs.update({attr.name: attr()})

        for name in self.name:
            dataarray.name = name()

        return dataarray

    def to_dataset(
        self,
        reference: Reference = None,
        dataset_factory: Callable[..., RDataset] = xr.Dataset,
    ) -> RDataset:
        """Create a Dataset object from the data model."""
        dataset = dataset_factory()

        for data in self.data:
            dataset.update({data.name: data(reference)})

        for coord in self.coord:
            dataset.coords.update({coord.name: coord(dataset)})

        for attr in self.attr:
            dataset.attrs.update({attr.name: attr()})

        return dataset


# runtime functions
def typedarray(
    data: Any,
    dims: Dims,
    dtype: Dtype,
    reference: Reference = None,
) -> xr.DataArray:
    """Create a DataArray object with given dims and dtype."""
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
