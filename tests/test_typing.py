# dependencies
import numpy as np
from pytest import mark
from xarray_dataclasses.typing import DataArray


# test datasets
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


# test functions
@mark.parametrize("dims, expected", testdata_dims)
def test_dims(dims, expected):
    assert DataArray[dims, None].dims == expected


@mark.parametrize("dtype, expected", testdata_dtype)
def test_dtype(dtype, expected):
    assert DataArray[None, dtype].dtype == expected
