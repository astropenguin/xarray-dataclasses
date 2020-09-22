__all__ = ["DataArray"]


# standard library
from typing import Any, Optional, Sequence, Tuple, Type, Union


# dependencies
import numpy as np
import xarray as xr


# type aliases
Dims = Optional[Sequence[str]]
Dtype = Optional[Union[Type, str]]


# main features
class DataArrayMeta(type):
    """Metaclass for the DataArray class."""

    def __getitem__(cls, options: Tuple[Dims, Dtype]) -> Type:
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
    """Type hint for xarray.DataArray."""

    dims: Dims = None  #: Dimensions to be fixed in DataArray instances.
    dtype: Dtype = None  #: Datatype to be fixed in DataArray instances.

    def __new__(cls, data: Any, **kwargs) -> xr.DataArray:
        data = np.asarray(data, dtype=cls.dtype)
        return xr.DataArray(data, dims=cls.dims, **kwargs)
