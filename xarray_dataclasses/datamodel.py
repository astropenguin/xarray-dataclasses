__all__ = ["DataModel"]


# standard library
from dataclasses import Field, dataclass, field, is_dataclass
from typing import Any, List, Type, Union, cast


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
    get_dims,
    get_dtype,
    get_inner,
    unannotate,
)


# type hints
DataType = TypedDict("DataType", dims=Dims, dtype=Dtype)
Reference = Union[xr.DataArray, xr.Dataset, None]


# field models
@dataclass
class Data:
    """Model for the coord or data fields."""

    name: str
    type: DataType
    value: Any

    def __call__(self, reference: Reference = None) -> xr.DataArray:
        """Create a DataArray object from the value and a reference."""
        return typedarray(
            self.value,
            self.type["dims"],
            self.type["dtype"],
            reference,
        )

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "Data":
        """Create a field model from a dataclass field and a value."""
        return cls(
            field.name,
            {
                "dims": get_dims(field.type),
                "dtype": get_dtype(field.type),
            },
            value,
        )


@dataclass
class Dataof:
    """Model for the coordof or dataof fields."""

    name: str
    type: Type[DataClass]
    value: Any

    def __call__(self, reference: Reference = None) -> xr.DataArray:
        """Create a DataArray object from the value and a reference."""
        from .dataarray import asdataarray

        if is_dataclass(self.value):
            return asdataarray(self.value, reference)
        else:
            return asdataarray(self.type(self.value), reference)

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "Dataof":
        """Create a field model from a dataclass field and a value."""
        return cls(field.name, get_inner(unannotate(field.type), 0), value)


@dataclass
class General:
    """Model for the attribute or name fields."""

    name: str
    type: Any
    value: Any

    def __call__(self) -> Any:
        """Just return the value."""
        return self.value

    @classmethod
    def from_field(cls, field: Field[Any], value: Any) -> "General":
        """Create a field model from a dataclass field and a value."""
        return cls(field.name, get_inner(unannotate(field.type)), value)


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
