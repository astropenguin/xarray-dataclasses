__all__ = ["DataArray"]


# standard library
from typing import Any, Optional, Sequence, Tuple, Union


# dependencies
import numpy as np
import xarray as xr


# type aliases
Dims = Optional[Sequence[str]]
Dtype = Optional[Union[type, str]]


# main features
class DataArrayMeta(type):
    """Metaclass for the DataArray class."""

    def __getitem__(cls, options: Tuple[Dims, Dtype]) -> type:
        try:
            dims, dtype = options
        except (ValueError, TypeError):
            raise ValueError("Both dims and dtype must be specified.")

        if isinstance(dims, str):
            dims = (dims,)

        if isinstance(dims, list):
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
            is_equal_dims = True  # do not evaluate
        else:
            is_equal_dims = inst.dims == cls.dims

        if cls.dtype is None:
            is_equal_dtype = True  # do not evaluate
        else:
            is_equal_dtype = inst.dtype == cls.dtype

        return is_equal_dims and is_equal_dtype


class DataArray(metaclass=DataArrayMeta):
    """Type hint for ``xarray.DataArray``.

    Args:
        data: Values for a ``DataArray`` instance.
        kwargs: Options passed to ``DataArray()``.

    Returns:
        ``DataArray`` with fixed ``dims`` and ``dtype``.

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

    def __new__(cls, data: Any, **kwargs) -> xr.DataArray:
        data = np.asarray(data, dtype=cls.dtype)
        return xr.DataArray(data, dims=cls.dims, **kwargs)
