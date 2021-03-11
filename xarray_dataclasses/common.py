# standard library
from dataclasses import astuple, Field
from typing import Any, Callable, Dict


# third-party packages
from typing_extensions import Protocol


# submodules
from .typing import is_attr, is_coord, is_data, is_name


# type hints
DataClassDict = Dict[str, Any]  #: Type alias for dataclass fields.


class DataClass(Protocol):
    """Type hint for dataclass and its instance."""

    __dataclass_fields__: Dict[str, Field]


# helper functions (internal)
def asdict_attr(inst: DataClass) -> DataClassDict:
    """Return attr fields of a dataclass instance."""
    return _asdict_by_type(inst, is_attr)


def asdict_coord(inst: DataClass) -> DataClassDict:
    """Return coord fields of a dataclass instance."""
    return _asdict_by_type(inst, is_coord)


def asdict_data(inst: DataClass) -> DataClassDict:
    """Return data fields of a dataclass instance."""
    return _asdict_by_type(inst, is_data)


def asdict_name(inst: DataClass) -> DataClassDict:
    """Return name fields of a dataclass instance."""
    return _asdict_by_type(inst, is_name)


def _asdict_by_type(inst: DataClass, checker: Callable[[Any], bool]) -> DataClassDict:
    """Similar to asdict, but only has fields whose types are matched by checker."""
    fields = inst.__dataclass_fields__
    return {key: val for key, val in astuple(inst) if checker(fields[key].type)}
