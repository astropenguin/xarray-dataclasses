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
class DataArrayClassMeta(type):
    """Metaclass of ``DataArrayClass``."""

    def __instancecheck__(cls, inst: Any) -> bool:
        if not isinstance(inst, get_data_field(cls).type):
            return False

        for name, field in get_coords_fields(cls).items():
            if name not in inst.coords:
                return False

            if not isinstance(inst.coords[name], field.type):
                return False

        return True


@dataclass
class DataArrayClass(metaclass=DataArrayClassMeta):
    """Base class for dataclasses."""

    data: DataArray  #: Values for a ``DataArray`` instance.

    def __init_subclass__(cls) -> None:
        dataclass(cls)


def is_dataarrayclass(obj: Any) -> bool:
    """Check if obj is a dataarrayclass or its instance.

    It returns ``True`` if ``obj`` fulfills all the
    following conditions or ``False`` otherwise.

    1. ``obj`` is a Python's native dataclass.
    2. ``obj`` has a data field whose type is DataArray.
    3. All ``dims`` of coords are subsets of data ``dims``.

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

    data_dims = data_field.type.dims

    if data_dims is None:
        data_dims_set = set()
    else:
        data_dims_set = set(data_dims)

    for field in get_coords_fields(obj).values():
        coord_dims = field.type.dims

        # Coord dims must be fixed.
        if coord_dims is None:
            return False

        # Coord dims must be empty if data dims is not fixed.
        # Otherwise, it must be a subset of data dims.
        if not set(coord_dims).issubset(data_dims_set):
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

    def condition(name, field):
        return name != DATA and issubclass(field.type, DataArray)

    fields = obj.__dataclass_fields__
    return {k: v for k, v in fields.items() if condition(k, v)}
