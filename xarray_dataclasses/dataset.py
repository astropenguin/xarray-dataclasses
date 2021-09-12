__all__ = ["asdataset", "AsDataset"]


# standard library
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, overload, Type


# third-party packages
import xarray as xr
from typing_extensions import Protocol


# submodules
from .typing import DataClass, TDataset
from .parser import parse
from .utils import copy_class


# type hints
class DataClassWithFactory(DataClass, Protocol[TDataset]):
    __dataset_factory__: Callable[..., TDataset]


# runtime functions
@overload
def asdataset(
    inst: DataClassWithFactory[TDataset],
    dataset_factory: Type[Any] = xr.Dataset,
) -> TDataset:
    ...


@overload
def asdataset(
    inst: DataClass,
    dataset_factory: Type[TDataset] = xr.Dataset,
) -> TDataset:
    ...


def asdataset(inst: Any, dataset_factory: Any = xr.Dataset) -> Any:
    """Convert a Dataset-class instance to Dataset one."""
    try:
        dataset_factory = inst.__dataset_factory__
    except AttributeError:
        pass

    return parse(inst).to_dataset(dataset_factory=dataset_factory)


# mix-in class
class AsDataset:
    """Mix-in class that provides shorthand methods."""

    __dataset_factory__ = xr.Dataset

    @classmethod
    def new(
        cls: Type[DataClassWithFactory[TDataset]],
        *args: Any,
        **kwargs: Any,
    ) -> TDataset:
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
        init.__annotations__["return"] = TDataset

        # create a concrete new method and bind
        @classmethod
        @wraps(init)
        def new(
            cls: Type[DataClassWithFactory[TDataset]],
            *args: Any,
            **kwargs: Any,
        ) -> TDataset:
            return asdataset(cls(*args, **kwargs))

        cls.new = new  # type: ignore
