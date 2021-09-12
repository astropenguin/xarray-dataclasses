# standard library
import re
from pprint import pformat
from typing import Any, Pattern, Type, TypeVar


# constants
COPIED_CLASS: str = "__xrdc_copied_class__"
CLASS_REPR: Pattern[str] = re.compile(r"^<class '(.+)'>$")


# type hints
T = TypeVar("T")


# runtime functions
def copy_class(cls: Type[T]) -> Type[T]:
    """Copy a class as a new one unless it is already copied."""

    if hasattr(cls, COPIED_CLASS):
        raise RuntimeError("Could not copy an already copied class.")

    if cls.__bases__ == (object,):
        bases = ()
    else:
        bases = cls.__bases__

    namespace = {
        **cls.__dict__.copy(),
        **{COPIED_CLASS: True},
    }

    return type(cls.__name__, bases, namespace)


def resolve_class(cls: Type[Any]) -> str:
    """Return the prettified representation of a class."""
    class_repr = pformat(cls)
    match = CLASS_REPR.search(class_repr)

    if match:
        return match.group(1)
    else:
        return class_repr
