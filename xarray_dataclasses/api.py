__all__ = ["asarray", "asdataarray", "asdataset", "asset", "asxarray"]


# standard library
from typing import Any, Callable, overload


# dependencies
from xarray import DataArray, Dataset
from .typing import P, DataClass, DataClassOf, TArray, TSet, TXarray


@overload
def asarray(obj: DataClass[P], /, *, factory: Callable[..., TArray]) -> TArray: ...


@overload
def asarray(obj: DataClassOf[TArray, P], /) -> TArray: ...


@overload
def asarray(obj: DataClass[P], /) -> DataArray: ...


def asarray(obj: Any, /, *, factory: Any = None) -> Any:
    pass


@overload
def asset(obj: DataClass[P], /, *, factory: Callable[..., TSet]) -> TSet: ...


@overload
def asset(obj: DataClassOf[TSet, P], /) -> TSet: ...


@overload
def asset(obj: DataClass[P], /) -> Dataset: ...


def asset(obj: Any, /, *, factory: Any = None) -> Any:
    pass


@overload
def asxarray(obj: DataClass[P], /, *, factory: Callable[..., TXarray]) -> TXarray: ...


@overload
def asxarray(obj: DataClassOf[TXarray, P], /) -> TXarray: ...


def asxarray(obj: Any, /, *, factory: Any = None) -> Any:
    pass


# function aliases
asdataarray = asarray
"""Alias of ``api.asdataarray``."""

asdataset = asset
"""Alias of ``api.asset``."""
