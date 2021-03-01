__all__ = ["set_fields"]


# standard library
from dataclasses import dataclass, field, Field, MISSING
from enum import auto, Enum
from types import MappingProxyType
from typing import Any, Optional


# third-party packages
from typing_extensions import Annotated, get_args, get_origin
from .typing import DataArray


# data classes
class Kind(Enum):
    ATTR = auto()  #: Attribute member of a DataArray.
    COORD = auto()  #: Coordinate member of a DataArray.
    DATA = auto()  #: Data of a DataArray.
    NAME = auto()  #: Name of a DataArray.


@dataclass(frozen=True)
class Xarray:
    kind: Kind
    doc: Optional[str] = None


# main features
def set_fields(cls: type) -> type:
    """Set dataclass fields to class."""
    for name, hint in cls.__annotations__.items():
        set_field(cls, name, hint)

    return cls


# helper features
def infer_kind(name: str, hint: Any) -> Kind:
    """Infer kind from name and type hint."""
    if get_origin(hint) == Annotated:
        hint = get_args(hint)[0]

    if name.upper() == Kind.DATA.name:
        return Kind.DATA

    if name.upper() == Kind.NAME.name:
        return Kind.NAME

    if get_origin(hint) is not None:
        return Kind.ATTR

    if not issubclass(hint, DataArray):
        return Kind.ATTR

    return Kind.COORD


def set_field(cls: type, name: str, hint: Any) -> type:
    """Set dataclass field to class with given name."""
    kind = infer_kind(name, hint)
    metadata = dict(xarray=Xarray(kind))

    obj = getattr(cls, name, MISSING)

    if isinstance(obj, Field):
        metadata = {**obj.metadata, **metadata}
        obj.metadata = MappingProxyType(metadata)
        return cls

    setattr(cls, name, field(default=obj, metadata=metadata))
    return cls
