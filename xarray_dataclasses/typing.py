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
    dims: Dims = None
    dtype: Dtype = None

    def __getitem__(cls, options: Tuple[Dims, Dtype]) -> Type:
        dims, dtype = options

        if isinstance(dims, str):
            dims = (dims,)

        if isinstance(dims, list):
            dims = tuple(dims)

        if dtype is not None:
            dtype = np.dtype(dtype)

        namespace = dict(dims=dims, dtype=dtype)
        return type(cls.__name__, (cls,), namespace)

    def __str__(cls) -> str:
        if cls.dims is None and cls.dtype is None:
            return cls.__name__
        else:
            return f"{cls.__name__}[{cls.dims!s}, {cls.dtype!s}]"

    def __repr__(cls) -> str:
        return f"{__name__}.{cls!s}"


class DataArray(metaclass=DataArrayMeta):
    def __new__(cls, data: Any, **kwargs) -> xr.DataArray:
        data = np.asarray(data, dtype=cls.dtype)
        return xr.DataArray(data, dims=cls.dims, **kwargs)
