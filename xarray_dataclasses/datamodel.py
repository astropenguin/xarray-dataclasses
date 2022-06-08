"""Submodule for data expression inside the package."""
__all__ = ["DataModel"]


# standard library
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple, Type, Union, cast


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import Literal, ParamSpec, get_type_hints


# submodules
from .typing import (
    AnyDType,
    AnyField,
    DataClass,
    AnyXarray,
    Dims,
    FType,
    get_annotated,
    get_dataclass,
    get_dims,
    get_dtype,
    get_ftype,
)


# type hints
PInit = ParamSpec("PInit")
AnyDataClass = Union[Type[DataClass[PInit]], DataClass[PInit]]
AnyEntry = Union["AttrEntry", "DataEntry"]


# runtime classes
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
        if self.value is MISSING:
            raise ValueError("Value is missing.")

        return self.value


@dataclass(frozen=True)
class DataEntry:
    """Entry of a data variable."""

    name: Hashable
    """Name that the attribute is accessed by."""

    tag: Literal["coord", "data"]
    """Function of the data (either coord or data)."""

    dims: Dims = cast(Dims, None)
    """Dimensions of the DataArray that the data is cast to."""

    dtype: Optional[AnyDType] = None
    """Data type of the DataArray that the data is cast to."""

    base: Optional[Type[DataClass[Any]]] = None
    """Base dataclass that converts the data to a DataArray."""

    value: Any = MISSING
    """Actual value of the data."""

    cast: bool = True
    """Whether the value is cast to the data type."""

    def __post_init__(self) -> None:
        """Update the entry if a base dataclass exists."""
        if self.base is None:
            return

        model = DataModel.from_dataclass(self.base)

        setattr = object.__setattr__
        setattr(self, "dims", model.data_vars[0].dims)
        setattr(self, "dtype", model.data_vars[0].dtype)

        if model.names:
            setattr(self, "name", model.names[0].value)

    def __call__(self, reference: Optional[AnyXarray] = None) -> xr.DataArray:
        """Create a DataArray object according to the entry."""
        from .dataarray import asdataarray

        if self.value is MISSING:
            raise ValueError("Value is missing.")

        if self.base is None:
            return get_typedarray(self.value, self.dims, self.dtype, reference)

        if is_dataclass(self.value):
            return asdataarray(self.value, reference)
        else:
            return asdataarray(self.base(self.value), reference)


@dataclass(frozen=True)
class DataModel:
    """Data representation (data model) inside the package."""

    entries: Dict[str, AnyEntry] = field(default_factory=dict)
    """Entries of data variable(s) and attribute(s)."""

    @property
    def attrs(self) -> List[AttrEntry]:
        """Return a list of attribute entries."""
        return [v for v in self.entries.values() if v.tag == "attr"]

    @property
    def coords(self) -> List[DataEntry]:
        """Return a list of coordinate entries."""
        return [v for v in self.entries.values() if v.tag == "coord"]

    @property
    def data_vars(self) -> List[DataEntry]:
        """Return a list of data variable entries."""
        return [v for v in self.entries.values() if v.tag == "data"]

    @property
    def data_vars_items(self) -> List[Tuple[str, DataEntry]]:
        """Return a list of data variable entries with keys."""
        return [(k, v) for k, v in self.entries.items() if v.tag == "data"]

    @property
    def names(self) -> List[AttrEntry]:
        """Return a list of name entries."""
        return [v for v in self.entries.values() if v.tag == "name"]

    @classmethod
    def from_dataclass(cls, dataclass: AnyDataClass[PInit]) -> "DataModel":
        """Create a data model from a dataclass or its object."""
        model = cls()
        eval_dataclass(dataclass)

        for field in dataclass.__dataclass_fields__.values():
            value = getattr(dataclass, field.name, MISSING)
            entry = get_entry(field, value)

            if entry is not None:
                model.entries[field.name] = entry

        return model


# runtime functions
def eval_dataclass(dataclass: AnyDataClass[PInit]) -> None:
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


def get_entry(field: AnyField, value: Any) -> Optional[AnyEntry]:
    """Create an entry from a field and its value."""
    ftype = get_ftype(field.type)

    if ftype is FType.ATTR or ftype is FType.NAME:
        return AttrEntry(
            name=field.name,
            tag=ftype.value,
            value=value,
            type=get_annotated(field.type),
        )

    if ftype is FType.COORD or ftype is FType.DATA:
        try:
            return DataEntry(
                name=field.name,
                tag=ftype.value,
                base=get_dataclass(field.type),
                value=value,
            )
        except TypeError:
            return DataEntry(
                name=field.name,
                tag=ftype.value,
                dims=get_dims(field.type),
                dtype=get_dtype(field.type),
                value=value,
            )


def get_typedarray(
    data: Any,
    dims: Dims,
    dtype: Optional[AnyDType],
    reference: Optional[AnyXarray] = None,
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
    try:
        data.__array__
    except AttributeError:
        data = np.asarray(data)

    if dtype is not None:
        data = data.astype(dtype, copy=False)

    if data.ndim == len(dims):
        dataarray = xr.DataArray(data, dims=dims)
    elif data.ndim == 0 and reference is not None:
        dataarray = xr.DataArray(data)
    else:
        raise ValueError(
            "Could not create a DataArray object from data. "
            f"Mismatch between shape {data.shape} and dims {dims}."
        )

    if reference is None:
        return dataarray
    else:
        ddims = set(reference.dims) - set(dims)
        reference = reference.isel({dim: 0 for dim in ddims})
        return dataarray.broadcast_like(reference)
