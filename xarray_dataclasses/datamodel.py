__all__ = ["DataModel"]


# standard library
from dataclasses import dataclass, field, Field, InitVar, is_dataclass
from typing import Any, Callable, cast, Generic, List, TypeVar, Union


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import TypedDict


# submodules
from .deprecated import get_type_hints
from .typing import (
    ArrayLike,
    DataClass,
    Dims,
    Dtype,
    FieldType,
    get_class,
    get_dims,
    get_dtype,
    Reference,
    unannotate,
)
from .utils import resolve_class


# type hints
R = TypeVar("R")
Factory = Callable[[Any, Reference], R]


class DataArrayDict(TypedDict):
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
        """Add a factory to the field model."""
        self.factory = factory

    def __call__(self, reference: Reference = None) -> R:
        """Create an object from the value and a reference."""
        return self.factory(self.value, reference)


@dataclass
class Data(FieldModel[xr.DataArray]):
    """Model for the coord or data fields."""

    type: DataArrayDict
    """Type of the field."""

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "Data":
        """Create a field model from a dataclass field and a value."""
        dims = get_dims(field.type)
        dtype = get_dtype(field.type)
        type: DataArrayDict = {"dims": dims, "dtype": dtype}

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
        dataclass = get_class(field.type)
        type = resolve_class(dataclass)

        def factory(value: Any, reference: Reference) -> xr.DataArray:
            from .dataarray import asdataarray

            if not is_dataclass(value):
                value = dataclass(value)

            return asdataarray(value, reference)

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
        eval_field_types(dataclass)

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


# runtime functions
def eval_field_types(dataclass: DataClass) -> None:
    """Evaluate field types of a dataclass or its object."""
    hints = get_type_hints(dataclass, include_extras=True)  # type: ignore

    for field_ in dataclass.__dataclass_fields__.values():
        if isinstance(field_.type, str):
            field_.type = hints[field_.name]


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
