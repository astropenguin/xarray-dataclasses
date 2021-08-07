# standard library
from typing import TypeVar


# third-party packages
from typing_extensions import Annotated, get_args, get_origin


# type hints
T = TypeVar("T")


# helper features
def unannotate(obj: T) -> T:
    """Recursively remove Annotated types."""
    if get_origin(obj) is Annotated:
        obj = get_args(obj)[0]

    origin = get_origin(obj)

    if origin is None:
        return obj

    args = map(unannotate, get_args(obj))
    args = tuple(filter(None, args))

    try:
        return origin[args]
    except TypeError:
        import typing

        name = origin.__name__.capitalize()
        return getattr(typing, name)[args]
