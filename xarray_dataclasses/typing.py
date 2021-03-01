__all__ = ["DataArray"]


# standard library
from dataclasses import Field, _DataclassParams
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Literal, Protocol


# type hints (dataclasses)
class DataClass(Protocol):
    """Type hint for dataclasses."""

    __dataclass_fields__: Dict[str, Field]
    __dataclass_params__: _DataclassParams


DataClassDecorator = Callable[[type], DataClass]


# type hints (numpy)
Dtype = Optional[Union[np.dtype, type, str]]
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]


# type hints (xarray)
Attrs = Optional[Mapping]
Coords = Optional[Union[Sequence[Tuple], Mapping[Hashable, Any]]]
Dims = Optional[Union[Sequence[Hashable], Hashable]]
Name = Optional[Hashable]


class DataArrayMeta(type):
    """Metaclass of the type hint for xarray.DataArray."""

    def __getitem__(cls, options: Tuple[Dims, Dtype]) -> "DataArrayMeta":
        try:
            dims, dtype = options
        except (ValueError, TypeError):
            raise ValueError("Both dims and dtype must be specified.")

        if isinstance(dims, str):
            dims = (dims,)

        if dims is not None:
            dims = tuple(dims)

        if dtype is not None:
            dtype = np.dtype(dtype)

        if dims is None and dtype is None:
            name = cls.__name__
        else:
            name = f"{cls.__name__}[{dims!s}, {dtype!s}]"

        namespace = cls.__dict__.copy()
        namespace.update(dims=dims, dtype=dtype)

        return DataArrayMeta(name, (cls,), namespace)

    def __instancecheck__(cls, inst: Any) -> bool:
        if not isinstance(inst, xr.DataArray):
            return False

        if cls.dims is None:
            is_equal_dims = True  # Do not evaluate.
        else:
            is_equal_dims = inst.dims == cls.dims

        if cls.dtype is None:
            is_equal_dtype = True  # Do not evaluate.
        else:
            is_equal_dtype = inst.dtype == cls.dtype

        return is_equal_dims and is_equal_dtype


class DataArray(xr.DataArray, metaclass=DataArrayMeta):
    """Type hint for xarray.DataArray.

    As shown in the examples, it enables to specify fixed dimension(s)
    (``dims``) and datatype (``dtype``) of ``xarray.DataArray``.
    Users can use it to create a ``DataArray`` instance with fixed
    dimension(s) and datatype in the same manner as ``xarray.DataArray``.

    Args:
        data: Values of a ``DataArray`` instance.
            They are cast to ``dtype`` if it is specified in a hint.
        coords: Coordinates of a ``DataArray`` instance.
        dims: Dimension(s) of a ``DataArray`` instance.
            It is ignored if ``dims`` is specified in a hint.
        name: Name of a ``DataArray`` instance.
        attrs: Attributes of a ``DataArray`` instance.

    Returns:
        ``DataArray`` instance with fixed ``dims`` and ``dtype``.

    Examples:
        To fix ``dims`` to be ``('x', 'y')``::

            DataArray[('x', 'y'), None]

        To fix ``dtype`` to be ``float``::

            DataArray[None, float]

        To fix both ``dims`` and ``dtype``::

            DataArray[('x', 'y'), float]

        Not to fix neither ``dims`` nor ``dtype``::

            DataArray # or DataArray[None, None]

        A type can be instantiated::

            DataArray["x", float]([0, 1, 2])

            # <xarray.DataArray (x: 3)>
            # array([0., 1., 2.])
            # Dimensions without coordinates: x

    """

    __slots__: Tuple[str, ...] = ()  #: Do not allow to add any values.
    dims: Dims = None  #: Dimensions to be fixed in DataArray instances.
    dtype: Dtype = None  #: Datatype to be fixed in DataArray instances.

    def __new__(
        cls,
        data: Any,
        coords: Coords = None,
        dims: Dims = None,
        name: Name = None,
        attrs: Attrs = None,
    ) -> xr.DataArray:
        """Create a DataArray instance with fixed dims and dtype."""
        data = np.array(data, cls.dtype)
        dims = dims if cls.dims is None else cls.dims
        return xr.DataArray(data, coords, dims, name, attrs)
