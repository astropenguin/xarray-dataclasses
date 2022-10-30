__all__ = ["Spec"]


# standard library
from dataclasses import dataclass, is_dataclass, replace
from dataclasses import Field as Field_, fields as fields_
from functools import lru_cache
from typing import Any, Callable, Hashable, List, Optional, Tuple, Type


# dependencies
from typing_extensions import Literal, get_type_hints


# submodules
from .typing import (
    P,
    DataClass,
    Role,
    Xarray,
    get_annotated,
    get_dims,
    get_dtype,
    get_name,
    get_role,
)


# runtime classes
@dataclass(frozen=True)
class Field:
    """Specification of a field."""

    id: str
    """Identifier of the field."""

    name: Hashable
    """Name of the field."""

    role: Literal["attr", "coord", "data"]
    """Role of the field."""

    default: Any
    """Default value of the field data."""

    type: Optional[Any] = None
    """Type (hint) of the field data."""

    dims: Optional[Tuple[str, ...]] = None
    """Dimensions of the field data."""

    dtype: Optional[str] = None
    """Data type of the field data."""

    def __post_init__(self) -> None:
        """Post updates for coordinate and data fields."""
        if not (self.role == "coord" or self.role == "data"):
            return None

        if is_dataclass(self.type):
            spec = Spec.from_dataclass(self.type)  # type: ignore
            field = spec.fields.of_data[0]
            object.__setattr__(self, "dims", field.dims)
            object.__setattr__(self, "dtype", field.dtype)
        else:
            object.__setattr__(self, "type", None)

    def update(self, obj: DataClass[P]) -> "Field":
        """Update the specification by a dataclass object."""
        return replace(
            self,
            name=format_name(self.name, obj),
            default=getattr(obj, self.id, self.default),
        )


class Fields(List[Field]):
    """List of field specifications (with selectors)."""

    @property
    def of_attr(self) -> "Fields":
        """Select only attribute field specifications."""
        return Fields(field for field in self if field.role == "attr")

    @property
    def of_coord(self) -> "Fields":
        """Select only coordinate field specifications."""
        return Fields(field for field in self if field.role == "coord")

    @property
    def of_data(self) -> "Fields":
        """Select only data field specifications."""
        return Fields(field for field in self if field.role == "data")

    def update(self, obj: DataClass[P]) -> "Fields":
        """Update the specifications by a dataclass object."""
        return Fields(field.update(obj) for field in self)


@dataclass(frozen=True)
class Spec:
    """Specification of a xarray dataclass."""

    fields: Fields
    """List of field specifications."""

    factory: Optional[Callable[..., Xarray]] = None
    """Factory for xarray data creation."""

    @classmethod
    def from_dataclass(cls, dataclass: Type[DataClass[P]]) -> "Spec":
        """Create a specification from a data class."""
        fields = Fields()

        for field_ in fields_(eval_types(dataclass)):
            field = convert_field(field_)

            if field is not None:
                fields.append(field)

        factory = getattr(dataclass, "__xarray_factory__", None)
        return cls(fields, factory)

    def update(self, obj: DataClass[P]) -> "Spec":
        """Update the specification by a dataclass object."""
        return replace(self, fields=self.fields.update(obj))

    def __matmul__(self, obj: DataClass[P]) -> "Spec":
        """Alias of the update method."""
        return self.update(obj)


# runtime functions
@lru_cache(maxsize=None)
def convert_field(field_: "Field_[Any]") -> Optional[Field]:
    """Convert a dataclass field to a field specification."""
    role = get_role(field_.type)

    if role is Role.OTHER:
        return None

    return Field(
        id=field_.name,
        name=get_name(field_.type, field_.name),
        role=role.name.lower(),  # type: ignore
        default=field_.default,
        type=get_annotated(field_.type),
        dims=get_dims(field_.type),
        dtype=get_dtype(field_.type),
    )


@lru_cache(maxsize=None)
def eval_types(dataclass: Type[DataClass[P]]) -> Type[DataClass[P]]:
    """Evaluate field types of a dataclass."""
    types = get_type_hints(dataclass, include_extras=True)

    for field_ in fields_(dataclass):
        field_.type = types[field_.name]

    return dataclass


def format_name(name: Hashable, obj: DataClass[P]) -> Hashable:
    """Format a name by a dataclass object."""
    if isinstance(name, tuple):
        return type(name)(format_name(elem, obj) for elem in name)

    if isinstance(name, str):
        return name.format(obj)

    return name
