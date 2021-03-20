# standard library
from dataclasses import dataclass
from typing import Tuple


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Literal


# submodules
from xarray_dataclasses.dataarray import DataArrayMixin
from xarray_dataclasses.typing import Attr, Coord, Data, Name


# constants
DIMS = "x", "y"
SHAPE = 10, 10


# type hints
X = Literal[DIMS[0]]
Y = Literal[DIMS[1]]


# dataclasses
@dataclass
class Image(DataArrayMixin):
    data: Data[float, Tuple[X, Y]]
    x: Coord[int, X] = 0
    y: Coord[int, Y] = 0
    dpi: Attr[int] = 100
    name: Name[str] = "image"


# test datasets
created = Image.ones(SHAPE)
expected = xr.DataArray(
    np.ones(SHAPE, float),
    dims=DIMS,
    coords={
        "x": ("x", np.zeros(SHAPE[0])),
        "y": ("y", np.zeros(SHAPE[1])),
    },
    attrs={"dpi": 100},
    name="image",
)


# test functions
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
