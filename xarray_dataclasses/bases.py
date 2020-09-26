__all__ = ["is_dataarrayclass"]


# standard library
from dataclasses import is_dataclass
from typing import Any


# dependencies
from .typing import DataArray


# constants
DATA = "data"


# main features
def is_dataarrayclass(obj: Any) -> bool:
    """Return True if obj is a valid dataarrayclass."""
    if not is_dataclass(obj):
        return False

    fields = obj.__dataclass_fields__

    if DATA not in fields:
        return False

    return issubclass(fields[DATA].type, DataArray)
