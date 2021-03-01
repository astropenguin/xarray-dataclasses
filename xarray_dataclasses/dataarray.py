__all__ = ["dataarrayclass", "asdataarray"]


# standard library
from dataclasses import dataclass, Field
from typing import Any, Optional, Union


# third-party packages
import numpy as np
import xarray as xr
from .field import FieldKind, set_fields
from .typing import DataClass, DataClassDecorator


# main features
def dataarrayclass(
    cls: Optional[type] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> Union[DataClass, DataClassDecorator]:
    """Convert class to a DataArray class."""

    set_options = dataclass(
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
    )

    def to_dataclass(cls: type) -> DataClass:
        set_fields(cls)
        set_options(cls)
        return cls

    if cls is not None:
        return to_dataclass(cls)  # DataClass
    else:
        return to_dataclass  # DataClassDecorator


def asdataarray(obj: DataClass) -> xr.DataArray:
    """Convert dataclass instance to a DataArray instance."""
    fields = obj.__dataclass_fields__
    dataarray = fields["data"].type(obj.data)

    for field in fields.values():
        value = getattr(obj, field.name)
        set_value(dataarray, field, value)

    return dataarray


# helper features
def set_value(dataarray: xr.DataArray, field: Field, value: Any) -> xr.DataArray:
    """Set value to a DataArray instance according to given field."""
    kind = field.metadata["xarray"].kind

    if kind == FieldKind.DATA:
        return dataarray

    if kind == FieldKind.ATTR:
        dataarray.attrs[field.name] = value
        return dataarray

    if kind == FieldKind.NAME:
        dataarray.name = value
        return dataarray

    if kind == FieldKind.COORD:
        try:
            coord = field.type(value)
        except ValueError:
            shape = tuple(dataarray.sizes[dim] for dim in field.type.dims)
            coord = field.type(np.full(shape, value))

        dataarray.coords[field.name] = coord
        return dataarray

    raise ValueError(f"Unsupported field kind: {kind}")
