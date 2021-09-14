__all__ = ["asdataset", "AsDataset"]


# standard library
from dataclasses import dataclass, Field
from functools import wraps
from typing import Any, Callable, Dict, overload, Type, TypeVar


# third-party packages
import xarray as xr
from typing_extensions import ParamSpec, Protocol


# submodules
from .parser import parse
from .utils import copy_class


# type hints
P = ParamSpec("P")
R = TypeVar("R", bound=xr.Dataset)
DatasetFactory = Callable[..., R]


class DataClass(Protocol[P]):
    """Type hint for dataclass objects."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]


class DataClassWithFactory(Protocol[P, R]):
    """Type hint for dataclass objects."""

    __init__: Callable[P, None]
    __dataclass_fields__: Dict[str, Field[Any]]
    __dataset_factory__: DatasetFactory[R]


# runtime functions
@overload
def asdataset(
    inst: DataClassWithFactory[P, R],
    dataset_factory: DatasetFactory[Any] = xr.Dataset,
) -> R:
    ...


@overload
def asdataset(
    inst: DataClass[P],
    dataset_factory: DatasetFactory[R] = xr.Dataset,
) -> R:
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

    def __dataset_factory__(self, data_vars: Any) -> xr.Dataset:
        """Default Dataset factory (xarray.Dataset)."""
        return xr.Dataset(data_vars)

    @classmethod
    def new(
        cls: Type[DataClassWithFactory[P, R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
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
