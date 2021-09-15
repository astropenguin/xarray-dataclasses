__all__ = ["asdataset", "AsDataset"]


# standard library
from dataclasses import Field
from functools import wraps
from typing import Any, Callable, Dict, overload, Type, TypeVar, Union


# third-party packages
import xarray as xr
from typing_extensions import ParamSpec, Protocol


# submodules
from .datamodel import DataModel
from .utils import copy_function


# type hints
P = ParamSpec("P")
R = TypeVar("R", bound=xr.Dataset)
Reference = Union[xr.DataArray, xr.Dataset, None]


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
    reference: Reference = None,
    dataset_factory: Any = xr.Dataset,
) -> R:
    ...


@overload
def asdataset(
    dataclass: DataClass[Any],
    reference: Reference = None,
    dataset_factory: Callable[..., R] = xr.Dataset,
) -> R:
    ...


def asdataset(
    dataclass: Any,
    reference: Any = None,
    dataset_factory: Any = xr.Dataset,
) -> Any:
    """Create a Dataset object from a dataclass object.

    Args:
        dataclass: Dataclass object that defines typed Dataset.
        reference: DataArray or Dataset object as a reference of shape.
        dataset_factory: Factory function of Dataset.

    Returns:
        Dataset object created from the dataclass object.

    """
    try:
        dataset_factory = dataclass.__dataset_factory__
    except AttributeError:
        pass

    model = DataModel.from_dataclass(dataclass)
    dataset = dataset_factory()

    for data in model.data:
        dataset.update({data.name: data(reference)})

    for coord in model.coord:
        dataset.coords.update({coord.name: coord(dataset)})

    for attr in model.attr:
        dataset.attrs.update({attr.name: attr()})

    return dataset


class AsDatasetMeta(type):
    """Metaclass of the AsDataset class."""

    @property
    def new(cls: Type[DataClassWithFactory[P, R]]) -> Callable[P, R]:
        """Create a Dataset object from dataclass parameters."""
        init = copy_function(cls.__init__)  # type: ignore
        init.__annotations__["return"] = R
        init.__doc__ = cls.__doc__

        @wraps(init)
        def wrapper(
            cls: Type[DataClassWithFactory[P, R]],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            return asdataset(cls(*args, **kwargs))

        return wrapper.__get__(cls)  # type: ignore


class AsDataset(metaclass=AsDatasetMeta):
    """Mix-in class that provides shorthand methods."""

    def __dataset_factory__(self, data_vars: Any) -> xr.Dataset:
        """Default Dataset factory (xarray.Dataset)."""
        return xr.Dataset(data_vars)
