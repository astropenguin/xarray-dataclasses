__all__ = ["asdataset", "datasetclass"]


# standard library
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, overload, Type, TypeVar, Union


# third-party packages
import xarray as xr
from typing_extensions import Protocol


# submodules
from .common import get_attrs, get_coords, get_data_vars
from .typing import DataClass
from .utils import copy_class, extend_class


# type hints (internal)
DS = TypeVar("DS", bound=xr.Dataset)


class DataClassWithFactory(DataClass, Protocol[DS]):
    __dataset_factory__: Callable[..., DS]


# runtime functions (public)
@overload
def asdataset(
    inst: DataClassWithFactory[DS],
    dataset_factory: Type[Any] = xr.Dataset,
) -> DS:
    ...


@overload
def asdataset(
    inst: DataClass,
    dataset_factory: Type[DS] = xr.Dataset,
) -> DS:
    ...


def asdataset(inst: Any, dataset_factory: Any = xr.Dataset) -> Any:
    """Convert a Dataset-class instance to Dataset one."""
    try:
        dataset_factory = inst.__dataset_factory__
    except AttributeError:
        pass

    dataset = dataset_factory(get_data_vars(inst))
    coords = get_coords(inst, dataset)

    dataset.coords.update(coords)
    dataset.attrs = get_attrs(inst)

    return dataset


def datasetclass(
    cls: Optional[Type[Any]] = None,
    *,
    init: bool = True,
    repr: bool = True,
    eq: bool = True,
    order: bool = False,
    unsafe_hash: bool = False,
    frozen: bool = False,
    shorthands: bool = True,
) -> Union[Type[DataClass], Callable[[type], Type[DataClass]]]:
    """Class decorator to create a Dataset class."""

    def to_dataclass(cls: Type[Any]) -> Type[DataClass]:
        if shorthands:
            cls = extend_class(cls, DatasetMixin)

        return dataclass(
            init=init,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
        )(cls)

    if cls is None:
        return to_dataclass
    else:
        return to_dataclass(cls)


# mix-in class (internal)
class DatasetMixin:
    """Mix-in class that provides shorthand methods."""

    __dataset_factory__ = xr.Dataset

    @classmethod
    def new(
        cls: Type[DataClassWithFactory[DS]],
        *args: Any,
        **kwargs: Any,
    ) -> DS:
        """Create a Dataset instance."""
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Update new() based on the dataclass definition."""
        super().__init_subclass__(**kwargs)

        # temporary class only for getting dataclass __init__
        try:
            Temp = dataclass(copy_class(cls))
        except RuntimeError:
            return

        init = Temp.__init__
        init.__annotations__["return"] = DS

        # create a concrete new method and bind
        @classmethod
        @wraps(init)
        def new(
            cls: Type[DataClassWithFactory[DS]],
            *args: Any,
            **kwargs: Any,
        ) -> DS:
            return asdataset(cls(*args, **kwargs))

        cls.new = new  # type: ignore
