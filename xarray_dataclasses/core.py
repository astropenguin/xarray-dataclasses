# standard library
from functools import wraps
from inspect import signature
from typing import Callable


# sub-modules/packages
from .typing import DataArray, Dims, Dtype


# helper features
def get_initializer(func: Callable, dims: Dims, dtype: Dtype) -> Callable:
    """Create a DataArray initializer with fixed dims and dtype."""
    sig = signature(func)
    TypedArray = DataArray[dims, dtype]

    for par in sig.parameters.values():
        if par.annotation == par.empty:
            raise ValueError("Type hint must be specified for all args.")

        if par.kind == par.VAR_POSITIONAL:
            raise ValueError("Variadic positional args cannot be used.")

        if par.kind == par.VAR_KEYWORD:
            raise ValueError("Variadic keyword args cannot be used.")

    @wraps(func)
    def wrapper(*args, **kwargs) -> TypedArray:
        for key in kwargs.keys():
            if key not in sig.parameters:
                kwargs.pop(key)

        return TypedArray(func(*args, **kwargs))

    return wrapper


def update_annotations(cls: type, based_on: Callable) -> None:
    """Update class annotations based on a DataArray initializer."""
    leading_annotations = {}
    trailing_annotations = {}

    for par in signature(based_on).parameters.values():
        if par.kind == par.KEYWORD_ONLY:
            trailing_annotations[par.name] = par.annotation
        else:
            leading_annotations[par.name] = par.annotation

    cls.__annotations__ = {
        **leading_annotations,
        **cls.__annotations__,
        **trailing_annotations,
    }


def update_defaults(cls: type, based_on: Callable) -> None:
    """Update class defaults based on a DataArray initializer."""
    for par in signature(based_on).parameters.values():
        if not par.default == par.empty:
            setattr(cls, par.name, par.default)
