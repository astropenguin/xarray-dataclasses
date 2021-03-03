from __future__ import annotations


__all__ = ["DataArray"]


# standard library
from dataclasses import Field, _DataclassParams
from typing import Any, Callable, Dict, Optional, Tuple, Union


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Protocol


# type hints (dataclasses)
class DataClass(Protocol):
    """Type hint for dataclasses."""

    __dataclass_fields__: Dict[str, Field]
    __dataclass_params__: _DataclassParams


DataClassDecorator = Callable[[type], DataClass]


# type hints (xarray)
Dtype = Optional[Union[np.dtype, type, str]]
Hints = type("Hints", (), xr.DataArray.__init__.__annotations__)


class DataArrayMeta(type):
    """Metaclass of the type hint for DataArray."""

    def __getitem__(cls, key: Tuple[Hints.dims, Dtype]) -> DataArrayMeta:
        """Define the behavior of DataArray[dims, dtype]."""
        try:
            dims, dtype = key
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

        namespace = cls.__dict__.copy()
        namespace.update(dims=dims, dtype=dtype)
        return DataArrayMeta(name, (cls,), namespace)

    def __instancecheck__(cls, obj: Any) -> bool:
        """Define the behavior of isinstance(obj, DataArray)."""
        if not isinstance(obj, xr.DataArray):
            return False

        if cls.dims is None:
            is_equal_dims = True  # Do not evaluate.
        else:
            is_equal_dims = obj.dims == cls.dims

        if cls.dtype is None:
            is_equal_dtype = True  # Do not evaluate.
        else:
            is_equal_dtype = obj.dtype == cls.dtype

        return is_equal_dims and is_equal_dtype


class DataArray(xr.DataArray, metaclass=DataArrayMeta):
    """Type hint for xarray.DataArray."""

    __slots__: Tuple[()] = ()
    dims: Hints.Dims = None
    dtype: Dtype = None

    def __new__(
        cls,
        data: Hints.data,
        coords: Hints.coords = None,
        dims: Hints.dims = None,
        name: Hints.name = None,
        attrs: Hints.attrs = None,
    ) -> xr.DataArray:
        data = np.array(data, cls.dtype)
        dims = cls.dims if cls.dims is not None else dims
        return xr.DataArray(data, coords, dims, name, attrs)
