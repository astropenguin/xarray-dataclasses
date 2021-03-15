__all__ = ["asdataarray", "dataarrayclass", "is_dataarrayclass"]


# standard library
from dataclasses import dataclass, is_dataclass
from typing import Any, Callable, Optional, Sequence, Union


# third-party packages
import xarray as xr
from typing_extensions import Literal


# submodules
from .common import DataClass, get_attrs, get_coords, get_data, get_name
from .utils import copy_wraps


# type hints (internal)
Order = Literal["C", "F"]
Shape = Union[Sequence[int], int]


# runtime functions (public)
def asdataarray(inst: DataClass) -> xr.DataArray:
    """Create a DataArray instance from a DataArray class instance."""
    dataarray = get_data(inst)
    coords = get_coords(inst, dataarray)

    dataarray.coords.update(coords)
    dataarray.attrs = get_attrs(inst)
    dataarray.name = get_name(inst)

    return dataarray


def dataarrayclass(
    cls: Optional[type] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> Union[DataClass, Callable[[type], DataClass]]:
    """Class decorator to create a DataArray class."""

    set_options = dataclass(
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
    )

    def to_dataclass(cls: type) -> DataClass:
        set_options(cls)
        set_shorthands(cls)
        return cls  # type: ignore

    if cls is not None:
        return to_dataclass(cls)
    else:
        return to_dataclass


def is_dataarrayclass(obj: Any) -> bool:
    """Check if object is a DataArray class or its instance."""
    return is_dataclass(obj)


# runtime functions (internal)
def set_shorthands(cls: type) -> None:
    """Set shorthand methods to a DataArray class."""

    new = copy_wraps(cls.__init__)(_new)  # type: ignore
    new.__annotations__["return"] = xr.DataArray

    if _new.__doc__ is not None:
        new.__doc__ = _new.__doc__.format(cls=cls)

    cls.new = classmethod(new)  # type: ignore


# helper functions (internal)
def _new(cls, *args, **kwargs) -> xr.DataArray:
    """Shorthand for asdataarray({cls.__name__}(...))."""
    return asdataarray(cls(*args, **kwargs))
