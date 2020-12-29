# standard library
from dataclasses import Field, MISSING, _DataclassParams
from dataclasses import field as get_field
from enum import auto, Flag
from typing import Any, Dict, Type, TypeVar


# third-party packages
from typing_extensions import Annotated, Final, get_args, get_origin, Protocol


# sub-modules
from .typing import DataArray


# constants
DATA_FIELD: Final[str] = "data"
FIELD_KIND: Final[str] = "field_kind"
NAME_FIELD: Final[str] = "name"


class FieldKind(Flag):
    """Enum for specifying kinds of dataclass fields."""

    ATTR = auto()  #: Member in attributes of DataArray.
    COORD = auto()  #: Member in coordinates of DataArray.
    DATA = auto()  #: Data (values) of DataArray.
    NAME = auto()  #: Name of DataArray.


# type hints
C = TypeVar("C")


class DataClass(Protocol):
    """Type hint for dataclasses."""

    __dataclass_fields__: Dict[str, Field]
    __dataclass_params__: _DataclassParams


class DataArrayClass(Protocol):
    """Type hint for DataArray classes."""

    data: DataArray
    __dataclass_fields__: Dict[str, Field]
    __dataclass_params__: _DataclassParams


# helper features
def cast_fields(inst: DataArrayClass) -> DataArrayClass:
    """Cast dataclass fields of an instance."""

    def setattr(obj, name, value):
        """Local setattr function (for frozen instances)."""
        super(type(inst), inst).__setattr__(name, value)

    for name, field in inst.__dataclass_fields__.items():
        if field.metadata[FIELD_KIND] == FieldKind.ATTR:
            continue

        value = getattr(inst, name)
        setattr(inst, name, field.type(value))

    return inst


def infer_field_kind(name: str, hint: Any) -> FieldKind:
    """Return field kind inferred from name and type hint."""
    if get_origin(hint) == Annotated:
        hint = get_args(hint)[0]

    if name == DATA_FIELD:
        return FieldKind.DATA

    if name == NAME_FIELD:
        return FieldKind.NAME

    if get_origin(hint) is not None:
        return FieldKind.ATTR

    if not issubclass(hint, DataArray):
        return FieldKind.ATTR

    return FieldKind.COORD


def set_fields(cls: Type[C]) -> Type[C]:
    """Set dataclass fields to a class."""
    for name, hint in cls.__annotations__.items():
        default = getattr(cls, name, MISSING)
        metadata = {FIELD_KIND: infer_field_kind(name, hint)}

        field = get_field(default=default, metadata=metadata)
        setattr(cls, name, field)

    return cls
