# standard library
from functools import wraps
from inspect import signature
from typing import Callable


# sub-modules/packages
from .typing import DataArray, Dims, Dtype


# helper features
def add_dataarray_init(
    cls: type,
    dims: Dims,
    dtype: Dtype,
    func: Callable,
) -> type:
    sig = signature(func)
    Typed = DataArray[dims, dtype]

    @staticmethod
    @wraps(func)
    def wrapper(*args, **kwargs) -> Typed:
        for key in kwargs.keys():
            if key not in sig.parameters:
                kwargs.pop(key)

        return Typed(func(*args, **kwargs))

    cls.__dataarray_init__ = wrapper
    return cls
