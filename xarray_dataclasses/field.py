__all__ = ["set_fields", "XarrayField"]


# standard library
from dataclasses import dataclass, field, Field, MISSING
from enum import auto, Enum
from types import MappingProxyType
from typing import Any, Optional


# third-party packages
from typing_extensions import Annotated, get_args, get_origin
from .typing import DataArray


# data classes
class FieldKind(Enum):
    """Kind of xarray-related fields."""

    ATTR = auto()  #: Attribute member of a DataArray.
    COORD = auto()  #: Coordinate member of a DataArray.
    DATA = auto()  #: Data of a DataArray.
    NAME = auto()  #: Name of a DataArray.


@dataclass(frozen=True)
class XarrayField:
    """Metadata for xarray-related fields."""

    kind: FieldKind  #: Kind of a field.
    doc: Optional[str] = None  #: Docstring of a field.


# main features
def set_fields(cls: type) -> type:
    """Set dataclass fields to a class."""
    for name, hint in cls.__annotations__.items():
        set_field(cls, name, hint)

    return cls


# helper features
def infer_field_kind(name: str, hint: Any) -> FieldKind:
    """Infer field kind from given name and type hint."""
    if get_origin(hint) == Annotated:
        hint = get_args(hint)[0]

    if name.upper() == FieldKind.DATA.name:
        return FieldKind.DATA

    if name.upper() == FieldKind.NAME.name:
        return FieldKind.NAME

    if get_origin(hint) is not None:
        return FieldKind.ATTR

    if not issubclass(hint, DataArray):
        return FieldKind.ATTR

    return FieldKind.COORD


def set_field(cls: type, name: str, hint: Any) -> type:
    """Set dataclass field to a class with given name."""
    kind = infer_field_kind(name, hint)
    metadata = dict(xarray=XarrayField(kind))

    obj = getattr(cls, name, MISSING)

    if isinstance(obj, Field):
        metadata = {**obj.metadata, **metadata}
        obj.metadata = MappingProxyType(metadata)
        return cls

    setattr(cls, name, field(default=obj, metadata=metadata))
    return cls
