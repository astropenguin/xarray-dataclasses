__all__ = ["copy_class", "extend_class", "make_generic"]


# standard library
from typing import Sequence, Type, TypeVar


# type hints (internal)
T = TypeVar("T")
GenericAlias = type(Sequence[int])


# utility functions (internal)
def copy_class(cls: type, prefix: str = "Copied") -> type:
    """Copy a class as a new one whose name starts with prefix."""

    if cls.__name__.startswith(prefix):
        raise ValueError("Could not copy a copied class.")

    name = prefix + cls.__name__

    if cls.__bases__ == (object,):
        bases = ()
    else:
        bases = cls.__bases__

    return type(name, bases, cls.__dict__.copy())


def extend_class(cls: type, mixin: type) -> type:
    """Extend a class with a mix-in class."""
    if cls.__bases__ == (object,):
        bases = (mixin,)
    else:
        bases = (*cls.__bases__, mixin)

    return type(cls.__name__, bases, cls.__dict__.copy())


def make_generic(cls: Type[T]) -> Type[T]:
    """Make a class generic (only for type check)."""
    try:
        cls.__class_getitem__  # type: ignore
    except AttributeError:
        cls.__class_getitem__ = classmethod(GenericAlias)  # type: ignore

    return cls
