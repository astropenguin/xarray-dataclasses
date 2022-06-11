__all__ = ["DataOptions", "DataSpec"]


# standard library
from dataclasses import dataclass, field, fields
from functools import lru_cache
from typing import Any, Dict, Generic, Hashable, Optional, Type, TypeVar


# dependencies
from typing_extensions import Literal, ParamSpec, TypeAlias, get_type_hints


# submodules
from .typing import (
    AnyDType,
    AnyField,
    AnyXarray,
    DataClass,
    Dims,
    Role,
    get_annotated,
    get_dataclass,
    get_dims,
    get_dtype,
    get_name,
    get_role,
)


# type hints
AnySpec: TypeAlias = "ArraySpec | ScalarSpec"
PInit = ParamSpec("PInit")
TReturn = TypeVar("TReturn", AnyXarray, None)


# runtime classes
@dataclass(frozen=True)
class ArraySpec:
    """Specification of an array."""

    name: Hashable
    """Name of the array."""

    role: Literal["coord", "data"]
    """Role of the array."""

    dims: Dims
    """Dimensions of the array."""

    dtype: Optional[AnyDType]
    """Data type of the array."""

    default: Any
    """Default value of the array."""

    origin: Optional[Type[DataClass[Any]]] = None
    """Dataclass as origins of name, dims, and dtype."""

    def __post_init__(self) -> None:
        """Update name, dims, and dtype if origin exists."""
        if self.origin is None:
            return

        dataspec = DataSpec.from_dataclass(self.origin)
        setattr = object.__setattr__

        for spec in dataspec.specs.of_name.values():
            setattr(self, "name", spec.default)
            break

        for spec in dataspec.specs.of_data.values():
            setattr(self, "dims", spec.dims)
            setattr(self, "dtype", spec.dtype)
            break


@dataclass(frozen=True)
class ScalarSpec:
    """Specification of a scalar."""

    name: Hashable
    """Name of the scalar."""

    role: Literal["attr", "name"]
    """Role of the scalar."""

    type: Any
    """Type (hint) of the scalar."""

    default: Any
    """Default value of the scalar."""


class SpecDict(Dict[str, AnySpec]):
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


@dataclass(frozen=True)
class DataOptions(Generic[TReturn]):
    """Options for xarray data creation."""

    factory: Type[TReturn]
    """Factory for xarray data creation."""


@dataclass(frozen=True)
class DataSpec:
    """Data specification of an xarray dataclass."""

    specs: SpecDict = field(default_factory=SpecDict)
    """Dictionary of any specifications."""

    options: DataOptions[Any] = DataOptions(type(None))
    """Options for xarray data creation."""

    @classmethod
    def from_dataclass(
        cls,
        dataclass: Type[DataClass[PInit]],
        dataoptions: Optional[DataOptions[Any]] = None,
    ) -> "DataSpec":
        """Create a data specification from a dataclass."""
        specs = SpecDict()

        for field in fields(eval_types(dataclass)):
            spec = get_spec(field)

            if spec is not None:
                specs[field.name] = spec

        if dataoptions is None:
            return cls(specs)
        else:
            return cls(specs, dataoptions)


# runtime functions
@lru_cache(maxsize=None)
def eval_types(dataclass: Type[DataClass[PInit]]) -> Type[DataClass[PInit]]:
    """Evaluate field types of a dataclass."""
    types = get_type_hints(dataclass, include_extras=True)

    for field in fields(dataclass):
        field.type = types[field.name]

    return dataclass


@lru_cache(maxsize=None)
def get_spec(field: AnyField) -> Optional[AnySpec]:
    """Convert a dataclass field to a specification."""
    name = get_name(field.type, field.name)
    role = get_role(field.type)

    if role is Role.DATA or role is Role.COORD:
        try:
            return ArraySpec(
                name=name,
                role=role.value,
                dims=(),  # dummy
                dtype=None,  # dummy
                default=field.default,
                origin=get_dataclass(field.type),
            )
        except TypeError:
            return ArraySpec(
                name=name,
                role=role.value,
                dims=get_dims(field.type),
                dtype=get_dtype(field.type),
                default=field.default,
            )

    if role is Role.ATTR or role is Role.NAME:
        return ScalarSpec(
            name=name,
            role=role.value,
            type=get_annotated(field.type),
            default=field.default,
        )
