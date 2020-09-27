# standard library
from dataclasses import dataclass


# dependencies
from pytest import mark
from xarray_dataclasses.bases import is_dataarrayclass
from xarray_dataclasses.typing import DataArray


# test datasets
@dataclass
class Valid:
    data: DataArray[("x", "y"), float]
    x: DataArray["x", int] = 0
    y: DataArray["y", int] = 0


@dataclass
class InvalidDataType:
    data: DataArray
    x: DataArray["x", int] = 0
    y: DataArray["y", int] = 0


@dataclass
class InvalidCoordsType:
    data: DataArray[("x", "y"), float]
    x: DataArray["z", int] = 0  # noqa
    y: DataArray["z", int] = 0  # noqa


testdata_subclass = [
    (Valid, True),
    (InvalidDataType, False),
    (InvalidCoordsType, False),
]


# test functions
@mark.parametrize("cls, expected", testdata_subclass)
def test_subclass(cls, expected):
    assert is_dataarrayclass(cls) == expected
