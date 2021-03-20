# standard library
from typing import Any, ForwardRef, Tuple


# third-party packages
from pytest import mark
from typing_extensions import Literal


# submodules
from xarray_dataclasses.typing import DataArrayLike, get_dims, get_dtype


# type hints
x = ForwardRef("x")
y = ForwardRef("y")
X = Literal["x"]
Y = Literal["y"]


# test datasets
testdata_dims = [
    (None, ()),
    (Tuple[()], ()),
    (X, ("x",)),
    (x, ("x",)),
    (Tuple[X], ("x",)),
    (Tuple[x], ("x",)),  # type: ignore
    (Tuple[X, Y], ("x", "y")),
    (Tuple[x, y], ("x", "y")),  # type: ignore
]

testdata_dtype = [
    (Any, None),
    (int, int),
    ("int", "int"),
    (Literal["int"], "int"),
]


# test functions
@mark.parametrize("dims, expected", testdata_dims)
def test_dims(dims: Any, expected: Any) -> None:
    assert get_dims(DataArrayLike[Any, dims]) == expected


@mark.parametrize("dtype, expected", testdata_dtype)
def test_dtype(dtype: Any, expected: Any) -> None:
    assert get_dtype(DataArrayLike[dtype, Any]) == expected
