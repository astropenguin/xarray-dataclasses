__all__ = ["asdataarray", "asdataset", "asxarray"]


# standard library
from typing import Any, Callable, overload


# dependencies
from xarray import DataArray, Dataset
from .typing import DataClass, DataClassOf, PAny, TDataArray, TDataset, TXarray


@overload
def asdataarray(
    obj: DataClassOf[PAny, TDataArray],
    /,
    *,
    factory: None = None,
) -> TDataArray: ...


@overload
def asdataarray(
    obj: DataClass[PAny],
    /,
    *,
    factory: Callable[..., TDataArray],
) -> TDataArray: ...


@overload
def asdataarray(
    obj: DataClass[PAny],
    /,
    *,
    factory: None = None,
) -> DataArray: ...


def asdataarray(obj: Any, /, *, factory: Any = None) -> Any:
    """Create a DataArray object from a dataclass object."""
    ...


@overload
def asdataset(
    obj: DataClassOf[PAny, TDataset],
    /,
    *,
    factory: None = None,
) -> TDataset: ...


@overload
def asdataset(
    obj: DataClass[PAny],
    /,
    *,
    factory: Callable[..., TDataset],
) -> TDataset: ...


@overload
def asdataset(
    obj: DataClass[PAny],
    /,
    *,
    factory: None = None,
) -> Dataset: ...


def asdataset(obj: Any, /, *, factory: Any = None) -> Any:
    """Create a Dataset object from a dataclass object."""
    ...


@overload
def asxarray(
    obj: DataClassOf[PAny, TXarray],
    /,
    *,
    factory: None = None,
) -> TXarray: ...


@overload
def asxarray(
    obj: DataClass[PAny],
    /,
    *,
    factory: Callable[..., TXarray],
) -> TXarray: ...


def asxarray(obj: Any, /, *, factory: Any = None) -> Any:
    """Create a DataArray/set object from a dataclass object."""
    ...
