# standard library
import re
from pprint import pformat
from typing import Any, Pattern, Type


# constants
CLASS_REPR: Pattern[str] = re.compile(r"^<class '(.+)'>$")


# runtime functions
def resolve_class(cls: Type[Any]) -> str:
    """Return the prettified representation of a class."""
    class_repr = pformat(cls)
    match = CLASS_REPR.search(class_repr)

    if match:
        return match.group(1)
    else:
        return class_repr
