__all__ = ["asdataset", "datasetclass"]


# standard library
from dataclasses import dataclass
from typing import Optional, Type


# third-party packages
import xarray as xr
from typing_extensions import Protocol


# submodules
from .common import (
    ClassDecorator,
    DataClass,
    get_attrs,
    get_coords,
    get_data_vars,
)
from .utils import copy_wraps


# type hints (internal)
class DatasetClass(DataClass, Protocol):
    """Type hint for a Dataset-class instance."""

    new: classmethod


# runtime functions (public)
def asdataset(inst: DatasetClass) -> xr.Dataset:
    """Create a Dataset instance from a Dataset-class instance."""
    dataset = xr.Dataset(get_data_vars(inst))
    coords = get_coords(inst, dataset)

    dataset.coords.update(coords)
    dataset.attrs = get_attrs(inst)

    return dataset


def datasetclass(
    cls: Optional[type] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> ClassDecorator[DatasetClass]:
    """Class decorator to create a Dataset class."""

    set_options = dataclass(
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
    )

    def to_dataclass(cls: type) -> Type[DatasetClass]:
        set_options(cls)
        set_shorthands(cls)
        return cls

    if cls is None:
        return to_dataclass
    else:
        return to_dataclass(cls)


# runtime functions (internal)
def set_shorthands(cls: Type[DatasetClass]) -> None:
    """Set shorthand methods to a Dataset class."""

    @copy_wraps(cls.__init__)  # type: ignore
    def new(cls, *args, **kwargs):
        return asdataset(cls(*args, **kwargs))

    new.__annotations__["return"] = xr.Dataset
    new.__doc__ = (
        "Create a Dataset instance. This is a shorthand for "
        f"``asdataset({cls.__name__}(*args, **kwargs))``."
    )

    cls.new = classmethod(new)
