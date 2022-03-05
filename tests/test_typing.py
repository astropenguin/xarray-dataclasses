# standard library
from typing import Any, Optional, Tuple, Union


# third-party packages
from pytest import mark
from typing_extensions import Annotated, Literal


# submodules
from xarray_dataclasses.typing import (
    Attr,
    Collection,
    Coord,
    Data,
    Name,
    get_dims,
    get_dtype,
    get_field_type,
    get_repr_type,
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
    (Collection[X, Any], ("x",)),
    (Collection[Tuple[()], Any], ()),
    (Collection[Tuple[X], Any], ("x",)),
    (Collection[Tuple[X, Y], Any], ("x", "y")),
]

testdata_dtype = [
    (Any, None),
    (NoneType, None),
    (Int64, "int64"),
    (int, "int"),
    (Collection[Any, Any], None),
    (Collection[Any, NoneType], None),
    (Collection[Any, Int64], "int64"),
    (Collection[Any, int], "int"),
]

testdata_field_type = [
    (Attr[Any], "attr"),
    (Coord[Any, Any], "coord"),
    (Data[Any, Any], "data"),
    (Name[Any], "name"),
]

testdata_repr_type = [
    (int, int),
    (Annotated[int, "annotation"], int),
    (Union[int, float], int),
    (Optional[int], int),
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


@mark.parametrize("type_, repr_type", testdata_repr_type)
def test_get_repr_type(type_: Any, repr_type: Any) -> None:
    assert get_repr_type(type_) == repr_type
