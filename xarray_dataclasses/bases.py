__all__ = [
    "DataArrayClass",
    "is_dataarrayclass",
]


# standard library
from dataclasses import is_dataclass
from typing import Any


# dependencies
from .typing import DataArray


# constants
DATA = "data"


# main features
class DataArrayClass:
    """Base class for dataclasses."""

    def __init_subclass__(cls) -> None:
        if not is_dataarrayclass(cls):
            raise ValueError("Not a valid dataarrayclass.")


def is_dataarrayclass(obj: Any) -> bool:
    """Return True if obj is a valid dataarrayclass."""
    if not is_dataclass(obj):
        return False

    fields = obj.__dataclass_fields__

    if DATA not in fields:
        return False

    return issubclass(fields[DATA].type, DataArray)
