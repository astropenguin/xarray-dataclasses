# standard library
from dataclasses import field, MISSING
from enum import auto, Flag
from typing import Any


# third-party packages
from typing_extensions import Annotated, Final, get_args, get_origin


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


# helper features
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


def set_fields(cls: type) -> None:
    """Set dataclass fields to a class."""
    for name, hint in cls.__annotations__.items():
        default = getattr(cls, name, MISSING)
        metadata = {FIELD_KIND: infer_field_kind(name, hint)}

        setattr(cls, name, field(default=default, metadata=metadata))
