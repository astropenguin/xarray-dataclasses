__all__ = ["copy_func", "copy_wraps"]


# standard library
from copy import copy, deepcopy
from functools import wraps, WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES
from types import FunctionType
from typing import Callable, Sequence, TypeVar


# type variables (internal)
T = TypeVar("T")


# utility functions (internal)
def copy_func(func: FunctionType, deep: bool = False) -> FunctionType:
    """Copy a function as a different object.

    Args:
        func: Function object to be copied.
        deep: If ``True``, mutable attributes of ``func`` are deep-copied.

    Returns:
        Function as a different object from the original one.

    """
    copied = FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )

    # mutable attributes are copied by the given method
    copier = deepcopy if deep else copy
    copied.__annotations__ = copier(func.__annotations__)
    copied.__dict__ = copier(func.__dict__)
    copied.__kwdefaults__ = copier(func.__kwdefaults__)

    # immutable attributes are not copied (just assigned)
    copied.__doc__ = func.__doc__
    copied.__module__ = func.__module__
    copied.__name__ = func.__name__
    copied.__qualname__ = func.__qualname__

    return copied


def copy_wraps(
    wrapped: FunctionType,
    assigned: Sequence[str] = WRAPPER_ASSIGNMENTS,
    updated: Sequence[str] = WRAPPER_UPDATES,
) -> Callable[[T], T]:
    """Same as functools.wraps but uses a copied function."""
    return wraps(copy_func(wrapped), assigned, updated)
