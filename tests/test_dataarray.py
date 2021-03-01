# dependencies
from pytest import mark
from xarray_dataclasses.core import dataarrayclass, is_dataarrayclass
from xarray_dataclasses.typing import DataArray


# test datasets
@dataarrayclass(("x", "y"), float)
class Base:
    x: DataArray["x", int] = 0
    y: DataArray["y", int] = 0


@dataarrayclass(("x", "y", "z"), float)
class Extended(Base):
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
