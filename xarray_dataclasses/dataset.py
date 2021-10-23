__all__ = ["asdataset", "AsDataset"]


# standard library
from dataclasses import Field
from functools import wraps
from types import MethodType
from typing import Any, Callable, Dict, overload, Type, TypeVar


# dependencies
import xarray as xr
from morecopy import copy
from typing_extensions import ParamSpec, Protocol


# submodules
from .datamodel import DataModel
from .typing import Reference


# type hints
P = ParamSpec("P")
R = TypeVar("R", bound=xr.Dataset)


class DataClass(Protocol[P]):
    """Type hint for a dataclass object."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]


class DatasetClass(Protocol[P, R]):
    """Type hint for a dataclass object with a Dataset factory."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]
    __dataset_factory__: Callable[..., R]


# runtime functions and classes
@overload
def asdataset(
    dataclass: DatasetClass[Any, R],
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


class classproperty:
    """Class property only for AsDataset.new().

    As a classmethod and a property can be chained together since Python 3.9,
    this class will be removed when the support for Python 3.7 and 3.8 ends.

    """

    def __init__(self, func: Callable[..., Callable[P, R]]) -> None:
        self.__func__ = func

    def __get__(self, obj: Any, cls: Type[DatasetClass[P, R]]) -> Callable[P, R]:
        return self.__func__(cls)


class AsDataset:
    """Mix-in class that provides shorthand methods."""

    def __dataset_factory__(self, data_vars: Any = None) -> xr.Dataset:
        """Default Dataset factory (xarray.Dataset)."""
        return xr.Dataset(data_vars)

    @classproperty
    def new(cls: Type[DatasetClass[P, R]]) -> Callable[P, R]:
        """Create a Dataset object from dataclass parameters."""

        init = copy(cls.__init__)
        init.__annotations__["return"] = R
        init.__doc__ = cls.__init__.__doc__

        @wraps(init)
        def new(
            cls: Type[DatasetClass[P, R]],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            return asdataset(cls(*args, **kwargs))

        return MethodType(new, cls)
