__all__ = ["DataModel"]


# standard library
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple, Type, Union, cast


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import Literal, ParamSpec, get_type_hints


# submodules
from .typing import ArrayLike, DataClass, DataType, Dims, Dtype


# type hints
P = ParamSpec("P")
AnyDataClass = Union[Type[DataClass[P]], DataClass[P]]
AnyEntry = Union["AttrEntry", "DataEntry"]


# constants
class MissingType:
    """Singleton that indicates missing data."""

    _instance = None

    def __new__(cls) -> "MissingType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"


MISSING = MissingType()


# runtime classes
@dataclass(frozen=True)
class AttrEntry:
    """Entry of an attribute (i.e. metadata)."""

    name: Hashable
    """Name that the attribute is accessed by."""

    tag: Literal["attr", "name"]
    """Function of the attribute (either attr or name)."""

    type: Any = None
    """Type or type hint of the attribute."""

    value: Any = MISSING
    """Actual value of the attribute."""

    cast: bool = False
    """Whether the value is cast to the type."""

    def __call__(self) -> Any:
        """Create an object according to the entry."""
        ...


@dataclass(frozen=True)
class DataEntry:
    """Entry of a data variable."""

    name: Hashable
    """Name that the attribute is accessed by."""

    tag: Literal["coord", "data"]
    """Function of the data (either coord or data)."""

    dims: Dims = cast(Dims, None)
    """Dimensions of the DataArray that the data is cast to."""

    dtype: Dtype = cast(Dtype, None)
    """Data type of the DataArray that the data is cast to."""

    base: Optional[Type[Any]] = None
    """Base dataclass that converts the data to a DataArray."""

    value: Any = MISSING
    """Actual value of the data."""

    cast: bool = True
    """Whether the value is cast to the data type."""

    def __call__(self, reference: Optional[DataType] = None) -> xr.DataArray:
        """Create a DataArray object according to the entry."""
        ...


@dataclass(frozen=True)
class DataModel:
    """Data representation (data model) inside the package."""

    entries: Dict[str, AnyEntry] = field(default_factory=dict)
    """Entries of data variable(s) and attribute(s)."""

    @property
    def attrs(self) -> List[AttrEntry]:
        """Return a list of attribute entries."""
        ...

    @property
    def coords(self) -> List[DataEntry]:
        """Return a list of coordinate entries."""
        ...

    @property
    def data_vars(self) -> List[DataEntry]:
        """Return a list of data variable entries."""
        ...

    @property
    def data_vars_items(self) -> List[Tuple[str, DataEntry]]:
        """Return a list of data variable entries with keys."""
        ...

    @property
    def names(self) -> List[AttrEntry]:
        """Return a list of name entries."""
        ...

    @classmethod
    def from_dataclass(cls, dataclass: AnyDataClass[P]) -> "DataModel":
        """Create a data model from a dataclass or its object."""
        ...


# runtime functions
def eval_dataclass(dataclass: AnyDataClass[P]) -> None:
    """Evaluate field types of a dataclass."""
    if not is_dataclass(dataclass):
        raise TypeError("Not a dataclass or its object.")

    fields = dataclass.__dataclass_fields__.values()

    # do nothing if field types are already evaluated
    if not any(isinstance(field.type, str) for field in fields):
        return

    # otherwise, replace field types with evaluated types
    if not isinstance(dataclass, type):
        dataclass = type(dataclass)

    types = get_type_hints(dataclass, include_extras=True)

    for field in fields:
        field.type = types[field.name]


def typedarray(
    data: Any,
    dims: Dims,
    dtype: Dtype,
    reference: Optional[DataType] = None,
) -> xr.DataArray:
    """Create a DataArray object with given dims and dtype.

    Args:
        data: Data to be converted to the DataArray object.
        dims: Dimensions of the DataArray object.
        dtype: Data type of the DataArray object.
        reference: DataArray or Dataset object as a reference of shape.

    Returns:
        DataArray object with given dims and dtype.

    """
    if isinstance(data, ArrayLike):
        array = cast(np.ndarray, data)
    else:
        array = np.asarray(data)

    if dtype is not None:
        array = array.astype(dtype, copy=False)

    if array.ndim == len(dims):
        dataarray = xr.DataArray(array, dims=dims)
    elif array.ndim == 0 and reference is not None:
        dataarray = xr.DataArray(array)
    else:
        raise ValueError(
            "Could not create a DataArray object from data. "
            f"Mismatch between shape {array.shape} and dims {dims}."
        )

    if reference is None:
        return dataarray

    diff_dims = set(reference.dims) - set(dims)
    subspace = reference.isel({dim: 0 for dim in diff_dims})
    return dataarray.broadcast_like(subspace)
