# standard library
from dataclasses import dataclass
from typing import Tuple


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Literal


# submodules
from xarray_dataclasses.dataarray import DataArrayMixin
from xarray_dataclasses.dataset import DatasetMixin
from xarray_dataclasses.typing import Attr, Coord, Data

# constants
DIMS = "x", "y"
SHAPE = 10, 10


# type hints
X = Literal[DIMS[0]]
Y = Literal[DIMS[1]]


# dataclasses
@dataclass
class Image(DataArrayMixin):
    data: Data[Tuple[X, Y], float]


@dataclass
class RGBImage(DatasetMixin):
    red: Data[Tuple[X, Y], float]
    green: Data[Tuple[X, Y], float]
    blue: Data[Tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0
    dpi: Attr[int] = 100


# test datasets
created = RGBImage.new(
    Image.ones(SHAPE),
    Image.ones(SHAPE),
    Image.ones(SHAPE),
)
expected = xr.Dataset(
    data_vars={
        "red": xr.DataArray(np.ones(SHAPE), dims=DIMS),
        "green": xr.DataArray(np.ones(SHAPE), dims=DIMS),
        "blue": xr.DataArray(np.ones(SHAPE), dims=DIMS),
    },
    coords={
        "x": ("x", np.zeros(SHAPE[0])),
        "y": ("y", np.zeros(SHAPE[1])),
    },
    attrs={"dpi": 100},
)


# test functions
def test_data_vars() -> None:
    assert (created == expected).all()  # type: ignore


def test_dims() -> None:
    assert created.dims == expected.dims


def test_attrs() -> None:
    assert created.attrs == expected.attrs
