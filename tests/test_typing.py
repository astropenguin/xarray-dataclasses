# standard library
from typing import Any, Tuple


# third-party packages
import numpy as np
from pytest import mark
from typing_extensions import Literal


# submodules
from xarray_dataclasses.typing import DataArrayLike, get_dims, get_dtype


# type hints
X = Literal["x"]
Y = Literal["y"]


# test datasets
testdata_dims = [
    (None, ()),
    (Tuple[()], ()),
    (X, ("x",)),
    ("x", ("x",)),
    (Tuple[X], ("x",)),
    (Tuple["x"], ("x",)),  # noqa
    (Tuple[X, Y], ("x", "y")),
    (Tuple["x", "y"], ("x", "y")),  # noqa
]

testdata_dtype = [
    (Any, None),
    (bool, np.bool_),
    (bytes, np.bytes_),
    (str, np.str_),
    (int, np.int64),
    (float, np.float64),
    (complex, np.complex128),
    ("bool", np.bool_),
    ("bytes", np.bytes_),
    ("str", np.str_),
    ("int", np.int64),
    ("float", np.float64),
    ("complex", np.complex128),
    ("datetime64", np.datetime64),
]


# test functions
@mark.parametrize("test, expected", testdata_dims)
def test_dims(dims: Any, expected: Any) -> None:
    assert get_dims(DataArrayLike[Any, dims]) == expected  # type: ignore


@mark.parametrize("dtype, expected", testdata_dtype)
def test_dtype(dtype: Any, expected: Any) -> None:
    assert get_dtype(DataArrayLike[dtype, Any]) == expected  # type: ignore
