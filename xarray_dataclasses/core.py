# standard library
from enum import auto, Flag
from typing import Any


# third-party packages
from typing_extensions import Annotated, Final, get_args, get_origin


# sub-modules
from .typing import DataArray


# constants
DATA_FIELD: Final[str] = "data"
NAME_FIELD: Final[str] = "name"


class FieldKind(Flag):
    ATTR = auto()
    COORD = auto()
    DATA = auto()
    NAME = auto()


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
