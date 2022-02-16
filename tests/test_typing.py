# standard library
from typing import Any, Tuple


# third-party packages
from pytest import mark
from typing_extensions import Literal


# submodules
from xarray_dataclasses.typing import ArrayLike, get_dims, get_dtype


# type hints
Int64 = Literal["int64"]
NoneType = type(None)
X = Literal["x"]
Y = Literal["y"]


# test datasets
testdata_dims = [
    (X, ("x",)),
    (Tuple[()], ()),
    (Tuple[X], ("x",)),
    (Tuple[X, Y], ("x", "y")),
    (ArrayLike[X, Any], ("x",)),
    (ArrayLike[Tuple[()], Any], ()),
    (ArrayLike[Tuple[X], Any], ("x",)),
    (ArrayLike[Tuple[X, Y], Any], ("x", "y")),
]

testdata_dtype = [
    (Any, None),
    (NoneType, None),
    (Int64, "int64"),
    (int, "int"),
    (ArrayLike[Any, Any], None),
    (ArrayLike[Any, NoneType], None),
    (ArrayLike[Any, Int64], "int64"),
    (ArrayLike[Any, int], "int"),
]


# test functions
@mark.parametrize("type_, dims", testdata_dims)
def test_get_dims(type_: Any, dims: Any) -> None:
    assert get_dims(type_) == dims


@mark.parametrize("type_, dtype", testdata_dtype)
def test_get_dtype(type_: Any, dtype: Any) -> None:
    assert get_dtype(type_) == dtype
