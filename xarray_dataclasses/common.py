# standard library
from dataclasses import Field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Protocol


# submodules
from .typing import (
    DataArrayLike,
    get_dims,
    get_dtype,
    is_attr,
    is_coord,
    is_data,
    is_name,
)


# type hints
class DataClass(Protocol):
    """Type hint for dataclass and its instance."""

    __dataclass_fields__: Dict[str, Field]


# runtime function (internal)
def get_attrs(inst: DataClass) -> Dict[Hashable, Any]:
    """Return attrs for a DataArray or Dataset instance."""
    return {f.name: v for f, v in _gen_fields(inst, is_attr)}


def get_coords(
    inst: DataClass, bound_to: Union[xr.DataArray, xr.Dataset]
) -> Dict[Hashable, xr.DataArray]:
    """Return coords for a DataArray or Dataset instance.

    Args:
        inst: A dataclass instance.
        bound_to: A DataArray or Dataset instance to which coords are bound.

    Returns:
        Dictionary of DataArray instances to be bounded.

    """
    sizes = bound_to.sizes
    coords: Dict[Hashable, xr.DataArray] = {}

    for field, value in _gen_fields(inst, is_coord):
        try:
            coord = _dataarray(field.type, value)
        except ValueError:
            shape = tuple(sizes[dim] for dim in get_dims(field.type))
            coord = _dataarray(field.type, np.full(shape, value))

        coords[field.name] = coord

    return coords


def get_data(inst: DataClass) -> xr.DataArray:
    """Return data for a DataArray instance."""
    try:
        field, value = _get_one(dict(_gen_fields(inst, is_data)))
    except ValueError:
        raise ValueError("Exactly one Data-type value is allowed.")

    return _dataarray(field.type, value)


def get_name(inst: DataClass) -> Hashable:
    """Return name for a DataArray instance."""
    try:
        return _get_one(dict(_gen_fields(inst, is_name)))
    except ValueError:
        raise ValueError("Exactly one Name-type value is allowed.")


# helper functions (internal)
def _gen_fields(
    inst: DataClass, type_filter: Optional[Callable[..., bool]] = None
) -> Iterable[Tuple[Field, Any]]:
    """Generate field-value pairs from a dataclass instance.

    Args:
        inst: An instance of dataclass.
        type_filter: If specified, only field-value pairs
            s.t. ``type_filter(field.type) == True`` are yielded.

    Yields:
        Field-value pairs as tuple.

    """
    for name, field in inst.__dataclass_fields__.items():
        if type_filter is None or type_filter(field.type):
            yield field, getattr(inst, name)


def _get_one(obj: Mapping) -> Any:
    """Return value of mapping if it has exactly one entry."""
    if len(obj) != 1:
        raise ValueError("obj must have exactly one entry.")

    return next(iter(obj.values()))


def _dataarray(type_: Type[DataArrayLike], obj: DataArrayLike) -> xr.DataArray:
    """Convert object to a DataArray instance according to given type."""
    data = np.asarray(obj, dtype=get_dtype(type_))
    return xr.DataArray(data, dims=get_dims(type_))
