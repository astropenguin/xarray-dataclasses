# standard library
from typing import Any, Tuple


# third-party packages
from pytest import mark
from typing_extensions import Literal


# submodules
from xarray_dataclasses.typing import (
    ArrayLike,
    Attr,
    Coord,
    Data,
    Name,
    get_dims,
    get_dtype,
    get_field_type,
)


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

testdata_field_type = [
    (Attr[Any], "attr"),
    (Coord[Any, Any], "coord"),
    (Data[Any, Any], "data"),
    (Name[Any], "name"),
]


# test functions
@mark.parametrize("type_, dims", testdata_dims)
def test_get_dims(type_: Any, dims: Any) -> None:
    assert get_dims(type_) == dims


@mark.parametrize("type_, dtype", testdata_dtype)
def test_get_dtype(type_: Any, dtype: Any) -> None:
    assert get_dtype(type_) == dtype


@mark.parametrize("type_, field_type", testdata_field_type)
def test_get_field_type(type_: Any, field_type: Any) -> None:
    assert get_field_type(type_).value == field_type
