# standard library
import re
from copy import copy, deepcopy
from pprint import pformat
from types import FunctionType
from typing import Any, Pattern, Type, TypeVar


# constants
CLASS_REPR: Pattern[str] = re.compile(r"^<class '(.+)'>$")


# type hints
T = TypeVar("T", bound=FunctionType)


# runtime functions
def copy_function(function: T, deep: bool = False) -> T:
    """Copy a function as a different object.

    Args:
        function: Function object to be copied.
        deep: If True, mutable attributes are deep-copied.

    Returns:
        Copied function.

    """
    copied = type(function)(
        function.__code__,
        function.__globals__,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )

    # mutable attributes are copied by the given method
    copier = deepcopy if deep else copy
    copied.__annotations__ = copier(function.__annotations__)
    copied.__dict__ = copier(function.__dict__)
    copied.__kwdefaults__ = copier(function.__kwdefaults__)

    # immutable attributes are not copied
    copied.__doc__ = function.__doc__
    copied.__module__ = function.__module__
    copied.__name__ = function.__name__
    copied.__qualname__ = function.__qualname__

    return copied


def resolve_class(cls: Type[Any]) -> str:
    """Return the prettified representation of a class."""
    class_repr = pformat(cls)
    match = CLASS_REPR.search(class_repr)

    if match:
        return match.group(1)
    else:
        return class_repr
