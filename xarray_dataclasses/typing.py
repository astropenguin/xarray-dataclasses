__all__ = ["DataArray"]


# standard library
from typing import Any, Hashable, Mapping, Optional, Sequence, Tuple, Union


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import TypeAlias


# type aliases
Attrs: TypeAlias = Optional[Mapping]
Coords: TypeAlias = Optional[Union[Sequence[tuple], Mapping[Hashable, Any]]]
Dims: TypeAlias = Union[Sequence[Hashable], Hashable]
Dtype: TypeAlias = Optional[Union[type, str]]
Name: TypeAlias = Optional[Hashable]


# main features
class DataArrayMeta(type):
    """Metaclass of the type hint for xarray.DataArray."""

    def __getitem__(cls, options: Tuple[Dims, Dtype]) -> "DataArrayMeta":
        try:
            dims, dtype = options
        except (ValueError, TypeError):
            raise ValueError("Both dims and dtype must be specified.")

        if isinstance(dims, str):
            dims = (dims,)

        if dims is not None:
            dims = tuple(dims)

        if dtype is not None:
            dtype = np.dtype(dtype)

        if dims is None and dtype is None:
            name = cls.__name__
        else:
            name = f"{cls.__name__}[{dims!s}, {dtype!s}]"

        return type(name, (cls,), dict(dims=dims, dtype=dtype))

    def __instancecheck__(cls, inst: Any) -> bool:
        if not isinstance(inst, xr.DataArray):
            return False

        if cls.dims is None:
            is_equal_dims = True  # Do not evaluate.
        else:
            is_equal_dims = inst.dims == cls.dims

        if cls.dtype is None:
            is_equal_dtype = True  # Do not evaluate.
        else:
            is_equal_dtype = inst.dtype == cls.dtype

        return is_equal_dims and is_equal_dtype


class DataArray(metaclass=DataArrayMeta):
    """Type hint for xarray.DataArray.

    As shown in the examples, it enables to specify fixed dimension(s)
    (``dims``) and datatype (``dtype``) of ``xarray.DataArray``.
    Users can use it to create a ``DataArray`` instance with fixed
    dimension(s) and datatype in the same manner as ``xarray.DataArray``.

    Args:
        data: Values of a ``DataArray`` instance.
            They are cast to ``dtype`` if it is specified in a hint.
        coords: Coordinates of a ``DataArray`` instance.
        dims: Dimension(s) of a ``DataArray`` instance.
            It is ignored if ``dims`` is specified in a hint.
        name: Name of a ``DataArray`` instance.
        attrs: Attributes of a ``DataArray`` instance.

    Returns:
        ``DataArray`` instance with fixed ``dims`` and ``dtype``.

    Examples:
        To fix ``dims`` to be ``('x', 'y')``::

            DataArray[('x', 'y'), None]

        To fix ``dtype`` to be ``float``::

            DataArray[None, float]

        To fix both ``dims`` and ``dtype``::

            DataArray[('x', 'y'), float]

        Not to fix neither ``dims`` nor ``dtype``::

            DataArray # or DataArray[None, None]

        A type can be instantiated::

            DataArray["x", float]([0, 1, 2])

            # <xarray.DataArray (x: 3)>
            # array([0., 1., 2.])
            # Dimensions without coordinates: x

    """

    dims: Dims = None  #: Dimensions to be fixed in DataArray instances.
    dtype: Dtype = None  #: Datatype to be fixed in DataArray instances.

    def __new__(
        cls,
        data: Any,
        coords: Coords = None,
        dims: Dims = None,
        name: Name = None,
        attrs: Attrs = None,
    ) -> xr.DataArray:
        """Create a DataArray instance with fixed dims and dtype."""
        data = np.array(data, cls.dtype)
        dims = dims if cls.dims is None else cls.dims

        return xr.DataArray(data, coords, dims, name, attrs)
