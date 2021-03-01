__all__ = ["dataarrayclass"]


# standard library
from dataclasses import dataclass, Field, _DataclassParams
from typing import Callable, Dict, Optional, Union


# third-party packages
from typing_extensions import Protocol
from .field import set_fields


# type hints
class DataClass(Protocol):
    """Type hint for dataclasses."""

    __dataclass_fields__: Dict[str, Field]
    __dataclass_params__: _DataclassParams


DataClassDecorator = Callable[[type], DataClass]


# main features
def dataarrayclass(
    cls: Optional[type] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
) -> Union[DataClass, DataClassDecorator]:
    """Class decorator for creating DataArray class."""

    set_options = dataclass(
        init=init,
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
    )

    def to_dataclass(cls: type) -> type:
        set_fields(cls)
        set_options(cls)
        return cls

    if cls is None:
        return to_dataclass
    else:
        return to_dataclass(cls)
