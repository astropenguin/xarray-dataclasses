__all__ = [
    "DataArrayClass",
    "is_dataarrayclass",
]


# standard library
from dataclasses import dataclass, Field, is_dataclass
from typing import Any, Dict


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
    """Check if obj is a dataarrayclass or its instance.

    It returns ``True`` if ``obj`` fulfills all the
    following conditions or ``False`` otherwise.

    1. ``obj`` is a Python's dataclass.
    2. ``obj`` has a valid data field.
    3. All ``dims`` of coords are subsets of data's.

    Args:
        obj: Object to be checked.

    Returns:
        ``True`` if ``obj`` is a valid dataarrayclass
        or its instance or ``False`` otherwise.

    """
    if not is_dataclass(obj):
        return False

    try:
        data_field = get_data_field(obj)
    except (KeyError, ValueError):
        return False

    try:
        data_dims = set(data_field.type.dims)
    except TypeError:
        data_dims = set()  # type.dims is None

    for field in get_coords_fields(obj).values():
        try:
            coord_dims = set(field.type.dims)
        except TypeError:
            return False  # type.dims is None

        if not coord_dims <= data_dims:
            return False

    return True


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
