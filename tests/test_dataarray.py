# standard library
from dataclasses import dataclass
from typing import Literal, Tuple


# dependencies
import numpy as np
import xarray as xr


# submodules
from xarray_dataclasses.dataarray import AsDataArray
from xarray_dataclasses.dataoptions import DataOptions
from xarray_dataclasses.typing import Attr, Coord, Data, Name


# constants
DIMS = "x", "y"
SHAPE = 10, 10


# type hints
X = Literal["x"]
Y = Literal["y"]


# dataclasses
class Custom(xr.DataArray):
    """Custom DataArray."""

    __slots__ = ()


@dataclass
class Image(AsDataArray):
    """2D image as DataArray."""

    __dataoptions__ = DataOptions(Custom)

    data: Data[Tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0
    units: Attr[str] = "cd / m^2"
    name: Name[str] = "luminance"


# test datasets
created = Image.ones(SHAPE)
expected = Custom(
    np.ones(SHAPE, float),
    dims=DIMS,
    coords={
        "x": xr.DataArray(np.zeros(SHAPE[0]), dims="x"),
        "y": xr.DataArray(np.zeros(SHAPE[1]), dims="y"),
    },
    attrs={"units": "cd / m^2"},
    name="luminance",
)


# test functions
def test_type() -> None:
    assert type(created) is type(expected)


def test_data() -> None:
    assert (created == expected).all()  # type: ignore


def test_dtype() -> None:
    assert created.dtype == expected.dtype  # type: ignore


def test_dims() -> None:
    assert created.dims == expected.dims


def test_attrs() -> None:
    assert created.attrs == expected.attrs


def test_name() -> None:
    assert created.name == expected.name
