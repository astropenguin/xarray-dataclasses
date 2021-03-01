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
    """Class decorator to create a DataArray class.

    Args:
        cls: Class to be decorated.
        init: Same as the ``init`` parameter of ``dataclass()``.
        repr: Same as the ``repr`` parameter of ``dataclass()``.
        eq: Same as the ``eq`` parameter of ``dataclass()``.
        order: Same as the ``order`` parameter of ``dataclass()``.
        unsafe_hash: Same as the ``unsafe_hash`` parameter of ``dataclass()``.
        frozen: Same as the ``frozen`` parameter of ``dataclass()``.

    Returns:
        DataArray class or class decorator with fixed parameters.

    Examples:
        To create a DataArray class to represent images::

            from xarray_dataclasses import DataArray, dataarrayclass


            @dataarrayclass
            class Image:
                \"\"\"DataArray class to represent images.\"\"\"

                data: DataArray[('x', 'y'), float]
                x: DataArray['x', int]
                y: DataArray['y', int]

        To create a DataArray instance::

            image = Image.new([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])

        To create a DataArray instance filled with ones::

            ones = Image.ones((2, 2), x=[0, 1], y=[0, 1])

    """

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
    """Check if object is a DataArray class or its instance.

    It returns ``True`` if ``obj`` fulfills all the
    following conditions or ``False`` otherwise.

    1. ``obj`` is a Python's native dataclass or its instance.
    2. ``obj`` has a data field whose type is DataArray.
    3. All fields in ``obj`` have an xarray-related metadata.

    Args:
        obj: Object to be checked.

    Returns:
        ``True`` if ``obj`` fulfills the conditions above
        or ``False`` otherwise.

    """
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
