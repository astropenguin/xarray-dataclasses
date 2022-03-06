"""Submodule for customization of DataArray or Dataset creation."""
__all__ = ["DataOptions"]


# standard library
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


# submodules
from .typing import DataType


# type hints
TDataType = TypeVar("TDataType", bound=DataType)


# dataclasses
@dataclass(frozen=True)
class DataOptions(Generic[TDataType]):
    """Options for DataArray or Dataset creation."""

    factory: Callable[..., TDataType]
    """Factory function for DataArray or Dataset."""
