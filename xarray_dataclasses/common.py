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


# type hints (internal)
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
        bound_to: A DataArray or Dataset instance to bind.

    Returns:
        Dictionary of DataArray instances to be bounded.

    """
    sizes = bound_to.sizes
    fields = _gen_fields(inst, is_coord)
    return {f.name: _to_dataarray(v, f.type, sizes) for f, v in fields}


def get_data(inst: DataClass) -> xr.DataArray:
    """Return data for a DataArray instance."""
    field, value = next(iter(_gen_fields(inst, is_data)))
    return _to_dataarray(value, field.type)


def get_name(inst: DataClass) -> Hashable:
    """Return name for a DataArray instance."""
    return next(iter(_gen_fields(inst, is_name)))[1]


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


def _to_dataarray(
    data: DataArrayLike,
    type_: Type[DataArrayLike],
    sizes: Optional[Mapping[Hashable, int]] = None,
) -> xr.DataArray:
    """Create a DataArray instance from DataArrayLike object.

    Args:
        data: DataArrayLike object.
        type_: Type of ``data``. Must be DataArrayLike[T, D].
        sizes: If specified, it is used for broadcasting ``data``.

    Returns:
        DataArray instance whose dtype and dims follow ``type_``.

    Raises:
        ValueError: Raised if ``sizes`` are not specified
            when broadcasting ``data`` is necessary.

    """
    dims = get_dims(type_)
    dtype = get_dtype(type_)

    try:
        return xr.DataArray(np.asarray(data, dtype), dims=dims)
    except ValueError as error:
        if sizes is None:
            raise error

        shape = tuple(sizes[dim] for dim in dims)
        return xr.DataArray(np.full(shape, data, dtype), dims=dims)
