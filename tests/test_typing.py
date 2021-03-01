# third-party packages
import numpy as np
import xarray as xr
from pytest import mark
from xarray_dataclasses.typing import DataArray


# test datasets
dataarray = xr.DataArray([1, 2, 3], dims="x")

testdata_dims = [
    (None, None),
    ((), ()),
    ("x", ("x",)),
    (("x",), ("x",)),
    (("x", "y"), ("x", "y")),
]

testdata_dtype = [
    (None, None),
    (int, np.int64),
    (float, np.float64),
    ("int", np.int64),
    ("float", np.float64),
]

testdata_isinstance = [
    (dataarray, DataArray, True),
    (dataarray, DataArray[None, None], True),
    (dataarray, DataArray[None, int], True),
    (dataarray, DataArray[None, float], False),
    (dataarray, DataArray["x", None], True),
    (dataarray, DataArray["y", None], False),
    (dataarray, DataArray["x", int], True),
    (dataarray, DataArray["y", float], False),
]

testdata_issubclass = [
    (DataArray, xr.DataArray, True),
    (DataArray["x", None], xr.DataArray, True),
    (DataArray["x", None], DataArray, True),
    (xr.DataArray, DataArray, False),
    (xr.DataArray, DataArray["x", None], False),
    (DataArray, DataArray["x", None], False),
]


# test functions
@mark.parametrize("dims, expected", testdata_dims)
def test_dims(dims, expected):
    assert DataArray[dims, None].dims == expected


@mark.parametrize("dtype, expected", testdata_dtype)
def test_dtype(dtype, expected):
    assert DataArray[None, dtype].dtype == expected


@mark.parametrize("dataarray, type_, expected", testdata_isinstance)
def test_isinstance(dataarray, type_, expected):
    assert isinstance(dataarray, type_) == expected


@mark.parametrize("subcls, cls, expected", testdata_issubclass)
def test_issubclass(subcls, cls, expected):
    assert issubclass(subcls, cls) == expected
