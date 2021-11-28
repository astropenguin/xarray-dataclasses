# standard library
from typing import Any, Tuple


# third-party packages
from pytest import mark
from typing_extensions import Literal


# submodules
from xarray_dataclasses.typing import Data, get_dims, get_dtype, unannotate


# type hints
Int = Literal["int"]
X = Literal["x"]
Y = Literal["y"]


# test datasets
testdata_dims = [
    (Data[X, Any], ("x",)),
    (Data[Tuple[()], Any], ()),
    (Data[Tuple[X], Any], ("x",)),
    (Data[Tuple[X, Y], Any], ("x", "y")),
]

testdata_dtype = [
    (Data[X, Any], None),
    (Data[X, None], None),
    (Data[X, int], "int"),
    (Data[X, Int], "int"),
]


# test functions
@mark.parametrize("hint, dims", testdata_dims)
def test_get_dims(hint: Any, dims: Any) -> None:
    assert get_dims(unannotate(hint)) == dims


@mark.parametrize("hint, dtype", testdata_dtype)
def test_get_dtype(hint: Any, dtype: Any) -> None:
    assert get_dtype(unannotate(hint)) == dtype
