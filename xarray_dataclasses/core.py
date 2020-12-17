# standard library
from functools import wraps
from inspect import signature
from typing import Callable


# sub-modules/packages
from .typing import DataArray, Dims, Dtype


# helper features
def get_dataarray_init(func: Callable, dims: Dims, dtype: Dtype) -> Callable:
    """Returns a DataArray initializer with fixed dims and dtype."""
    sig = signature(func)
    TypedArray = DataArray[dims, dtype]

    for par in sig.parameters.values():
        if par.annotation == par.empty:
            raise ValueError("Type hint must be specified for all args.")

        if par.kind == par.VAR_POSITIONAL:
            raise ValueError("Positional args cannot be used.")

        if par.kind == par.VAR_KEYWORD:
            raise ValueError("Keyword args cannot be used.")

    @wraps(func)
    def dataarray_init(*args, **kwargs) -> TypedArray:
        for key in kwargs.keys():
            if key not in sig.parameters:
                kwargs.pop(key)

        return TypedArray[dims, dtype](func(*args, **kwargs))

    return dataarray_init

    cls.__dataarray_init__ = wrapper
    return cls
