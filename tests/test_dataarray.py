# standard library
from dataclasses import dataclass
from typing import Any, Tuple


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Literal


# submodules
from xarray_dataclasses.dataarray import AsDataArray
from xarray_dataclasses.typing import Attr, Coord, Data, Name


# constants
DIMS = "x", "y"
SHAPE = 10, 10


# type hints
X = Literal[DIMS[0]]
Y = Literal[DIMS[1]]


# dataclasses
class Custom(xr.DataArray):
    __slots__ = ()


@dataclass
class Image(AsDataArray):
    data: Data[Tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0
    dpi: Attr[int] = 100
    name: Name[str] = "image"

    def __dataarray_factory__(self, data: Any = None) -> Custom:
        return Custom(data)


# test datasets
created = Image.ones(SHAPE)
expected = Custom(
    np.ones(SHAPE, float),
    dims=DIMS,
    coords={
        "x": xr.DataArray(np.zeros(SHAPE[0]), dims="x"),
        "y": xr.DataArray(np.zeros(SHAPE[1]), dims="y"),
    },
    attrs={"dpi": 100},
    name="image",
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
