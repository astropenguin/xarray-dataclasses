__all__ = ["copy_class", "copy_func", "copy_wraps", "extend_class"]


# standard library
from copy import copy, deepcopy
from functools import wraps, WRAPPER_ASSIGNMENTS, WRAPPER_UPDATES
from types import FunctionType
import types
from typing import Callable, List, Sequence, Tuple, Type, TypeVar, cast


# type hints (internal)
T = TypeVar("T")


# utility functions (internal)
def copy_class(cls: type, prefix: str = "Copied") -> type:
    """Copy a class as a new one whose name starts with prefix."""

    if cls.__name__.startswith(prefix):
        raise ValueError("Could not copy a copied class.")

    name = prefix + cls.__name__

    if cls.__bases__ == (object,):
        bases: Tuple[type, ...] = ()
    else:
        bases = cls.__bases__

    return type(name, bases, cls.__dict__.copy())


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


def extend_class(cls: type, mixin: type) -> type:
    """Extend a class with a mix-in class."""
    if issubclass(cls, mixin):
        return cls
    if cls.__bases__ == (object,):
        bases: Tuple[type, ...] = (mixin,)
    else:
        # ensure consistent MRO
        collect: List[type] = []
        for i, b in enumerate(cls.__bases__):
            if issubclass(mixin, b):
                collect.append(mixin)
                collect.extend(cls.__bases__[i:])
                break
            collect.append(b)
        else:
            collect.append(mixin)
        bases = tuple(collect)

    return type(cls.__name__, bases, cls.__dict__.copy())
