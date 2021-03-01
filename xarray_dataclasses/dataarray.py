__all__ = ["asdataarray", "dataarrayclass", "is_dataarrayclass"]


# standard library
from dataclasses import dataclass, Field, is_dataclass
from typing import Any, Optional, Union


# third-party packages
import numpy as np
from .field import FieldKind, set_fields, XarrayMetadata
from .typing import DataArray, DataClass, DataClassDecorator, Order, Shape
from .utils import copy_wraps


# constants
C_ORDER: Order = "C"
DATA: str = "data"
XARRAY: str = "xarray"


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
        set_shorthands(cls)
        return cls

    if cls is not None:
        return to_dataclass(cls)  # DataClass
    else:
        return to_dataclass  # DataClassDecorator


def asdataarray(obj: DataClass) -> DataArray:
    """Convert dataclass instance to a DataArray instance."""
    fields = obj.__dataclass_fields__
    dataarray = fields[DATA].type(obj.data)

    for field in fields.values():
        value = getattr(obj, field.name)
        set_value(dataarray, field, value)

    return dataarray


def is_dataarrayclass(obj: Any) -> bool:
    """Check if object is a DataArray class or its instance."""
    # obj must be a dataclass or its instance
    if not is_dataclass(obj):
        return False

    # all fields must have an xarray-related metadata
    fields = obj.__dataclass_fields__

    for field in fields.values():
        metadata = field.metadata.get(XARRAY)

        if not isinstance(metadata, XarrayMetadata):
            return False

    # at least data field must be defined
    return DATA in fields


# helper features
def set_value(dataarray: DataArray, field: Field, value: Any) -> DataArray:
    """Set value to a DataArray instance according to given field."""
    kind = field.metadata[XARRAY].kind

    if kind == FieldKind.DATA:
        return dataarray

    if kind == FieldKind.NAME:
        dataarray.name = value
        return dataarray

    if kind == FieldKind.ATTR:
        dataarray.attrs[field.name] = value
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


def set_shorthands(cls: DataClass) -> DataClass:
    """Set shorthand methods to a DataArray class."""

    # create methods
    DataType = cls.__dataclass_fields__[DATA].type

    @copy_wraps(cls.__init__)
    def new(cls, *args, **kwargs):
        return asdataarray(cls(*args, **kwargs))

    new.__annotations__["return"] = DataType

    def empty(cls, shape: Shape, order: Order = C_ORDER, **kwargs) -> DataType:
        data = np.empty(shape, order=order)
        return asdataarray(cls(data=data, **kwargs))

    def zeros(cls, shape: Shape, order: Order = C_ORDER, **kwargs) -> DataType:
        data = np.zeros(shape, order=order)
        return asdataarray(cls(data=data, **kwargs))

    def ones(cls, shape: Shape, order: Order = C_ORDER, **kwargs) -> DataType:
        data = np.ones(shape, order=order)
        return asdataarray(cls(data=data, **kwargs))

    def full(
        cls, shape: Shape, fill_value: Any, order: Order = C_ORDER, **kwargs
    ) -> DataType:
        data = np.full(shape, fill_value, order=order)
        return asdataarray(cls(data=data, **kwargs))

    # add docstrings to methods
    def doc(code: str) -> str:
        return f"Shorthand for asdataarray({cls.__name__}({code}))."

    new.__doc__ = doc("*args, **kwargs")
    empty.__doc__ = doc("data=numpy.empty(shape), **kwargs")
    zeros.__doc__ = doc("data=numpy.zeros(shape), **kwargs")
    ones.__doc__ = doc("data=numpy.ones(shape), **kwargs")
    full.__doc__ = doc("data=numpy.full(shape, fill_value), **kwargs")

    # set methods to a class
    cls.new = classmethod(new)
    cls.empty = classmethod(empty)
    cls.zeros = classmethod(zeros)
    cls.ones = classmethod(ones)
    cls.full = classmethod(full)

    return cls
