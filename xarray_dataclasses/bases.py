__all__ = [
    "DataArrayClass",
    "is_dataarrayclass",
]


# standard library
from dataclasses import dataclass, is_dataclass
from typing import Any


# dependencies
from .typing import DataArray


# constants
DATA = "data"


# main features
@dataclass
class DataArrayClass:
    """Base class for dataclasses."""

    data: DataArray

    def __init_subclass__(cls) -> None:
        dataclass(cls)


def is_dataarrayclass(obj: Any) -> bool:
    """Return True if obj is a valid dataarrayclass."""
    if not is_dataclass(obj):
        return False

    fields = obj.__dataclass_fields__

    if DATA not in fields:
        return False

    return issubclass(fields[DATA].type, DataArray)
