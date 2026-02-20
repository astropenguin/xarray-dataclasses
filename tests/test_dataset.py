# standard library
from dataclasses import dataclass
from typing import Literal, Tuple


# dependencies
import numpy as np
import xarray as xr


# submodules
from xarray_dataclasses.dataarray import AsDataArray
from xarray_dataclasses.dataset import AsDataset
from xarray_dataclasses.dataoptions import DataOptions
from xarray_dataclasses.typing import Attr, Coord, Data

# constants
DIMS = "x", "y"
SHAPE = 10, 10


# type hints
X = Literal["x"]
Y = Literal["y"]


# dataclasses
class Custom(xr.Dataset):
    __slots__ = ()


@dataclass
class Image(AsDataArray):
    """2D image as DataArray."""

    data: Data[Tuple[X, Y], float]


@dataclass
class ColorImage(AsDataset):
    """2D color image as Dataset."""

    __dataoptions__ = DataOptions(Custom)

    red: Data[Tuple[X, Y], float]
    green: Data[Tuple[X, Y], float]
    blue: Data[Tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0
    units: Attr[str] = "cd / m^2"


# test datasets
created = ColorImage.new(
    Image.ones(SHAPE),
    Image.ones(SHAPE),
    Image.ones(SHAPE),
)
expected = Custom(
    data_vars={
        "red": xr.DataArray(np.ones(SHAPE), dims=DIMS),
        "green": xr.DataArray(np.ones(SHAPE), dims=DIMS),
        "blue": xr.DataArray(np.ones(SHAPE), dims=DIMS),
    },
    coords={
        "x": xr.DataArray(np.zeros(SHAPE[0]), dims="x"),
        "y": xr.DataArray(np.zeros(SHAPE[1]), dims="y"),
    },
    attrs={"units": "cd / m^2"},
)


# test functions
def test_type() -> None:
    assert type(created) is type(expected)


def test_data_vars() -> None:
    assert (created == expected).all()  # type: ignore


def test_dims() -> None:
    assert created.dims == expected.dims


def test_attrs() -> None:
    assert created.attrs == expected.attrs
