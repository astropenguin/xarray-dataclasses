__all__ = ["AsDataset", "asdataset"]


# standard library
from dataclasses import Field
from functools import wraps
from types import MethodType
from typing import Any, Callable, Dict, Type, TypeVar, overload


# dependencies
import xarray as xr
from morecopy import copy
from typing_extensions import ParamSpec, Protocol


# submodules
from .datamodel import DataModel, Reference


# type hints
P = ParamSpec("P")
TDataset = TypeVar("TDataset", bound=xr.Dataset)
TDataset_ = TypeVar("TDataset_", bound=xr.Dataset, contravariant=True)


class DataClass(Protocol[P]):
    """Type hint for a dataclass object."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]


class DatasetClass(Protocol[P, TDataset_]):
    """Type hint for a dataclass object with a Dataset factory."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]
    __dataset_factory__: Callable[..., TDataset_]


# custom classproperty
class classproperty:
    """Class property only for AsDataset.new().

    As a classmethod and a property can be chained together since Python 3.9,
    this class will be removed when the support for Python 3.7 and 3.8 ends.

    """

    def __init__(self, func: Callable[..., Callable[P, TDataset]]) -> None:
        self.__func__ = func

    def __get__(
        self,
        obj: Any,
        cls: Type[DatasetClass[P, TDataset]],
    ) -> Callable[P, TDataset]:
        return self.__func__(cls)


# runtime functions and classes
@overload
def asdataset(
    dataclass: DatasetClass[Any, TDataset],
    reference: Reference = None,
    dataset_factory: Any = xr.Dataset,
) -> TDataset:
    ...


@overload
def asdataset(
    dataclass: DataClass[Any],
    reference: Reference = None,
    dataset_factory: Callable[..., TDataset] = xr.Dataset,
) -> TDataset:
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


class AsDataset:
    """Mix-in class that provides shorthand methods."""

    def __dataset_factory__(self, data_vars: Any = None) -> xr.Dataset:
        """Default Dataset factory (xarray.Dataset)."""
        return xr.Dataset(data_vars)

    @classproperty
    def new(cls: Type[DatasetClass[P, TDataset]]) -> Callable[P, TDataset]:
        """Create a Dataset object from dataclass parameters."""

        init = copy(cls.__init__)
        init.__annotations__["return"] = TDataset
        init.__doc__ = cls.__init__.__doc__

        @wraps(init)
        def new(
            cls: Type[DatasetClass[P, TDataset]],
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> TDataset:
            return asdataset(cls(*args, **kwargs))

        return MethodType(new, cls)
