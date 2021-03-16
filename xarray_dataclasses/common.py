# standard library
from dataclasses import Field, MISSING
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
    TypeVar,
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


# type variables (internal)
T = TypeVar("T")


# type hints (internal)
ClassDecorator = Union[Type[T], Callable[[type], Type[T]]]


class DataClass(Protocol):
    """Type hint for dataclass instance."""

    __dataclass_fields__: Dict[str, Field]

    def __init__(self, *args, **kwargs) -> None:
        ...


# runtime function (internal)
def get_attrs(inst: DataClass) -> Dict[Hashable, Any]:
    """Return Attr-typed values for a DataArray or Dataset instance."""
    return {f.name: v for f, v in _gen_fields(inst, is_attr)}


def get_coords(
    inst: DataClass,
    bound_to: Union[xr.DataArray, xr.Dataset],
) -> Dict[Hashable, xr.DataArray]:
    """Return Coord-typed values for a DataArray or Dataset instance.

    Args:
        inst: Dataclass instance.
        bound_to: DataArray or Dataset instance to bind.

    Returns:
        Dictionary of DataArray instances to be bounded.

    """
    sizes = bound_to.sizes
    fields = _gen_fields(inst, is_coord)
    return {f.name: _to_dataarray(v, f.type, sizes) for f, v in fields}


def get_data(inst: DataClass) -> xr.DataArray:
    """Return Data-typed value for a DataArray instance."""
    fields = _gen_fields(inst, is_data)
    data = {f.name: _to_dataarray(v, f.type) for f, v in fields}

    if len(data) > 1:
        raise ValueError("Unique Data-typed value is allowed.")

    if len(data) == 0:
        raise ValueError("Could not find any Data-typed values.")

    return next(iter(data.values()))


def get_data_name(cls: Type[DataClass]) -> str:
    """Return name of Data-typed field for a DataArray instance."""
    fields = dict(_gen_fields(cls, is_data))

    if len(fields) > 1:
        raise ValueError("Unique Data-typed field is allowed.")

    if len(fields) == 0:
        raise ValueError("Could not find any Data-typed fields.")

    return next(iter(fields)).name


def get_data_vars(inst: DataClass) -> Dict[Hashable, xr.DataArray]:
    """Return Data-typed values for a Dataset instance."""
    fields = _gen_fields(inst, is_data)
    data_vars: Dict[Hashable, xr.DataArray]
    data_vars = {f.name: _to_dataarray(v, f.type) for f, v in fields}

    if len(data_vars) == 0:
        raise ValueError("Could not find any Data-typed values.")

    return data_vars


def get_name(inst: DataClass) -> Optional[Hashable]:
    """Return Name-typed value for a DataArray instance."""
    names = {f.name: v for f, v in _gen_fields(inst, is_name)}

    if len(names) > 1:
        raise ValueError("Unique Name-typed value is allowed.")

    if len(names) == 0:
        return None

    return next(iter(names.values()))


# helper functions (internal)
def _gen_fields(
    obj: Union[DataClass, Type[DataClass]],
    type_filter: Optional[Callable[..., bool]] = None,
) -> Iterable[Tuple[Field, Any]]:
    """Generate field-value pairs from a dataclass instance.

    Args:
        obj: Dataclass or its instance.
        type_filter: If specified, only field-value pairs
            s.t. ``type_filter(field.type) == True`` are yielded.

    Yields:
        Field-value pairs as tuple.

    """
    for name, field in obj.__dataclass_fields__.items():
        if type_filter is None or type_filter(field.type):
            yield field, getattr(obj, name, MISSING)


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
