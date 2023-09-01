__all__ = ["Spec"]


# standard library
from collections.abc import Callable, Hashable
from dataclasses import dataclass, is_dataclass, replace
from typing import Any, Generic, Literal, Optional


# dependencies
from typing_extensions import Self
from .typing import DataClass, TAny


# type hints
Intent = Literal[
    "attr",
    "attrs",
    "coord",
    "coords",
    "name",
    "root",
    "var",
    "vars",
]


@dataclass
class Spec(Generic[TAny]):
    """Specification for data creation."""

    id: str
    """Identifier of the specification."""

    name: Hashable
    """Name of the created data."""

    intent: Intent
    """Intent of the created data."""

    factory: Callable[..., TAny]
    """Factory for the data creation."""

    # optional (coord(s) and var(s) only)
    dims: Optional[tuple[str, ...]] = None
    """Data dimensions of the created data."""

    dtype: Optional[Any] = None
    """Data type of the created data."""

    # optional (any intent)
    default: Optional[Any] = None
    """Default value for the data creation."""

    origin: Optional[type[DataClass[Any]]] = None
    """Original dataclass of the specification."""

    specs: Optional["Specs[Any]"] = None
    """Sub-specifications of the specification."""

    def update(self, obj: Any) -> Self:
        """Update the specification by an object."""
        if self.specs is None or self.origin is None:
            return replace(self, default=obj)

        if not is_dataclass(obj):
            obj = self.origin(obj)

        return replace(self, specs=self.specs.update(obj))

    def __matmul__(self, obj: Any) -> Self:
        """Alias of self.update(obj)."""
        return self.update(obj)


class Specs(Generic[TAny], list[Spec[TAny]]):
    """List of specifications for data creation."""

    def of(self, intent: Intent) -> Self:
        """Select only specifications for given intent."""
        return type(self)(s for s in self if s.intent == intent)

    def update(self, obj: Any) -> Self:
        """Update the specifications by an object."""
        return type(self)(s.update(obj) for s in self)

