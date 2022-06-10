# standard library
from dataclasses import dataclass
from typing import Any, Dict, Hashable, Optional, Type


# dependencies
from typing_extensions import Literal, TypeAlias


# submodules
from .typing import Dims, DataClass, AnyDType


# type hints
AnySpec: TypeAlias = "ArraySpec | ScalarSpec"


# runtime classes
@dataclass(frozen=True)
class ArraySpec:
    """Specification of an array."""

    name: Hashable
    """Name of the array."""

    role: Literal["coord", "data"]
    """Role of the array."""

    default: Any
    """Default value of the array."""

    dims: Dims
    """Dimensions of the array."""

    type: Optional[AnyDType]
    """Data type of the array."""

    origin: Optional[Type[DataClass[Any]]] = None
    """Dataclass of dims and type origins."""


@dataclass(frozen=True)
class ScalarSpec:
    """Specification of a scalar."""

    name: Hashable
    """Name of the scalar."""

    role: Literal["attr", "name"]
    """Role of the scalar."""

    default: Any
    """Default value of the scalar."""

    type: Any
    """Data type of the scalar."""


class Specs(Dict[str, AnySpec]):
    """Dictionary of any specifications."""

    @property
    def of_attr(self) -> Dict[str, ScalarSpec]:
        """Limit to attribute specifications."""
        return {k: v for k, v in self.items() if v.role == "attr"}

    @property
    def of_coord(self) -> Dict[str, ArraySpec]:
        """Limit to coordinate specifications."""
        return {k: v for k, v in self.items() if v.role == "coord"}

    @property
    def of_data(self) -> Dict[str, ArraySpec]:
        """Limit to data specifications."""
        return {k: v for k, v in self.items() if v.role == "data"}

    @property
    def of_name(self) -> Dict[str, ScalarSpec]:
        """Limit to name specifications."""
        return {k: v for k, v in self.items() if v.role == "name"}