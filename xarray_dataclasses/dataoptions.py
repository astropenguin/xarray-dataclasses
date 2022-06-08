"""Submodule for customization of DataArray or Dataset creation."""
__all__ = ["DataOptions"]


# standard library
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar


# submodules
from .typing import AnyXarray


# type hints
TAnyXarray = TypeVar("TAnyXarray", bound=AnyXarray)


# dataclasses
@dataclass(frozen=True)
class DataOptions(Generic[TAnyXarray]):
    """Options for DataArray or Dataset creation."""

    factory: Callable[..., TAnyXarray]
    """Factory function for DataArray or Dataset."""
