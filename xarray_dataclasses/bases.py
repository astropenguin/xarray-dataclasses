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

# helper features
def get_data_field(obj: DataArrayClass) -> Field:
    """Return data field if it exists and is valid.

    Args:
        obj: Dataarrayclass or its instance.

    Returns:
        Data field if it exists and is valid.

    Raises:
        KeyError: Raised if data field does not exist.
        ValueError: Raised if data field has an invalid type.

    """
    try:
        data_field = obj.__dataclass_fields__[DATA]
    except KeyError:
        raise KeyError("Data field does not exist.")

    if not issubclass(data_field.type, DataArray):
        raise ValueError("Data field has an invalid type.")

    return data_field


def get_coords_fields(obj: DataArrayClass) -> Dict[str, Field]:
    """Return fields of DataArray coordinates.

    Args:
        obj: Dataarrayclass or its instance.

    Returns:
        Fields of DataArray coordinates.

    """
    items = obj.__dataclass_fields__.items()
    return {k: v for k, v in items if issubclass(v.type, DataArray)}
