__all__ = ["asdataset", "AsDataset"]


# standard library
from dataclasses import dataclass, Field
from functools import wraps
from typing import Any, Callable, Dict, overload, Type, TypeVar


# third-party packages
import xarray as xr
from typing_extensions import ParamSpec, Protocol


# submodules
from .parser import DataModel
from .utils import copy_class


# type hints
P = ParamSpec("P")
R = TypeVar("R", bound=xr.Dataset)


class DataClass(Protocol[P]):
    """Type hint for a dataclass object."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]


class DataClassWithFactory(Protocol[P, R]):
    """Type hint for a dataclass object with a Dataset factory."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]
    __dataset_factory__: Callable[..., R]


# runtime functions and classes
@overload
def asdataset(
    dataclass: DataClassWithFactory[Any, R],
    dataset_factory: Any = xr.Dataset,
) -> R:
    ...


@overload
def asdataset(
    dataclass: DataClass[Any],
    dataset_factory: Callable[..., R] = xr.Dataset,
) -> R:
    ...


def asdataset(
    dataclass: Any,
    dataset_factory: Any = xr.Dataset,
) -> Any:
    """Create a Dataset object from a dataclass object.

    Args:
        dataclass: Dataclass object that defines typed Dataset.
        dataset_factory: Factory function of Dataset.

    Returns:
        Dataset object created from the dataclass object.

    """
    model = DataModel.from_dataclass(dataclass)

    try:
        return model.to_dataset(None, dataclass.__dataset_factory__)
    except AttributeError:
        return model.to_dataset(None, dataset_factory)


class AsDataset:
    """Mix-in class that provides shorthand methods."""

    def __dataset_factory__(self, data_vars: Any) -> xr.Dataset:
        """Default Dataset factory (xarray.Dataset)."""
        return xr.Dataset(data_vars)

    @classmethod
    def new(
        cls: Type[DataClassWithFactory[P, R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Create a Dataset object."""
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
        init.__annotations__["return"] = R

        # create a concrete new method and bind
        @classmethod
        @wraps(init)
        def new(
            cls: Type[DataClassWithFactory[P, R]],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            return asdataset(cls(*args, **kwargs))

        cls.new = new  # type: ignore
