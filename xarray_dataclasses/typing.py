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


class DataArray(metaclass=DataArrayMeta):
    dims: Dims = None
    dtype: Dtype = None

    def __new__(cls, data: Any, **kwargs) -> xr.DataArray:
        data = np.asarray(data, dtype=cls.dtype)
        return xr.DataArray(data, dims=cls.dims, **kwargs)
