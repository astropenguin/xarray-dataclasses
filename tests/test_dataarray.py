# third-party packages
from pytest import mark
from xarray_dataclasses.dataarray import dataarrayclass, is_dataarrayclass
from xarray_dataclasses.typing import DataArray


# test datasets
DIMS = "x", "y", "z"


@dataarrayclass
class Base:
    data: DataArray[DIMS[:2], float]
    x: DataArray["x", int] = 0
    y: DataArray["y", int] = 0


@dataarrayclass
class Extended(Base):
    data: DataArray[DIMS, float]
    z: DataArray["z", int] = 0


class Invalid:
    w: DataArray["w", int] = 0


testdata = [
    (Base, True),
    (Extended, True),
    (Invalid, False),
]


# test functions
@mark.parametrize("cls, expected", testdata)
def test_isdataarrayclass(cls, expected):
    assert is_dataarrayclass(cls) == expected
