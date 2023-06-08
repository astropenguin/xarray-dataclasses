# standard library
from typing import Any, Literal as L, Tuple, Union


# dependencies
import numpy as np
from pytest import mark
from typing_extensions import Annotated as Ann


# submodules
from xarray_dataclasses.typing import (
    Attr,
    Coord,
    Data,
    Name,
    Role,
    get_dims,
    get_dtype,
    get_name,
    get_role,
)


# test datasets
testdata_dims = [
    (Coord[Tuple[()], Any], ()),
    (Coord[L["x"], Any], ("x",)),
    (Coord[Tuple[L["x"]], Any], ("x",)),
    (Coord[Tuple[L["x"], L["y"]], Any], ("x", "y")),
    (Data[Tuple[()], Any], ()),
    (Data[L["x"], Any], ("x",)),
    (Data[Tuple[L["x"]], Any], ("x",)),
    (Data[Tuple[L["x"], L["y"]], Any], ("x", "y")),
    (Ann[Coord[L["x"], Any], "coord"], ("x",)),
    (Ann[Data[L["x"], Any], "data"], ("x",)),
    (Union[Ann[Coord[L["x"], Any], "coord"], Ann[Any, "any"]], ("x",)),
    (Union[Ann[Data[L["x"], Any], "data"], Ann[Any, "any"]], ("x",)),
]

testdata_dtype = [
    (Coord[Any, Any], None),
    (Coord[Any, None], None),
    (Coord[Any, int], np.dtype("i8")),
    (Coord[Any, L["i8"]], np.dtype("i8")),
    (Data[Any, Any], None),
    (Data[Any, None], None),
    (Data[Any, int], np.dtype("i8")),
    (Data[Any, L["i8"]], np.dtype("i8")),
    (Ann[Coord[Any, float], "coord"], np.dtype("f8")),
    (Ann[Data[Any, float], "data"], np.dtype("f8")),
    (Union[Ann[Coord[Any, float], "coord"], Ann[Any, "any"]], np.dtype("f8")),
    (Union[Ann[Data[Any, float], "data"], Ann[Any, "any"]], np.dtype("f8")),
]

testdata_name = [
    (Attr[Any], None),
    (Data[Any, Any], None),
    (Coord[Any, Any], None),
    (Name[Any], None),
    (Any, None),
    (Ann[Attr[Any], "attr"], "attr"),
    (Ann[Data[Any, Any], "data"], "data"),
    (Ann[Coord[Any, Any], "coord"], "coord"),
    (Ann[Name[Any], "name"], "name"),
    (Ann[Any, "other"], None),
    (Union[Ann[Attr[Any], "attr"], Ann[Any, "any"]], "attr"),
    (Union[Ann[Data[Any, Any], "data"], Ann[Any, "any"]], "data"),
    (Union[Ann[Coord[Any, Any], "coord"], Ann[Any, "any"]], "coord"),
    (Union[Ann[Name[Any], "name"], Ann[Any, "any"]], "name"),
    (Union[Ann[Any, "other"], Ann[Any, "any"]], None),
]

testdata_role = [
    (Attr[Any], Role.ATTR),
    (Data[Any, Any], Role.DATA),
    (Coord[Any, Any], Role.COORD),
    (Name[Any], Role.NAME),
    (Any, Role.OTHER),
    (Ann[Attr[Any], "attr"], Role.ATTR),
    (Ann[Data[Any, Any], "data"], Role.DATA),
    (Ann[Coord[Any, Any], "coord"], Role.COORD),
    (Ann[Name[Any], "name"], Role.NAME),
    (Ann[Any, "other"], Role.OTHER),
    (Union[Ann[Attr[Any], "attr"], Ann[Any, "any"]], Role.ATTR),
    (Union[Ann[Data[Any, Any], "data"], Ann[Any, "any"]], Role.DATA),
    (Union[Ann[Coord[Any, Any], "coord"], Ann[Any, "any"]], Role.COORD),
    (Union[Ann[Name[Any], "name"], Ann[Any, "any"]], Role.NAME),
    (Union[Ann[Any, "other"], Ann[Any, "any"]], Role.OTHER),
]


# test functions
@mark.parametrize("tp, dims", testdata_dims)
def test_get_dims(tp: Any, dims: Any) -> None:
    assert get_dims(tp) == dims


@mark.parametrize("tp, dtype", testdata_dtype)
def test_get_dtype(tp: Any, dtype: Any) -> None:
    assert get_dtype(tp) == dtype


@mark.parametrize("tp, name", testdata_name)
def test_get_name(tp: Any, name: Any) -> None:
    assert get_name(tp) == name


@mark.parametrize("tp, role", testdata_role)
def test_get_role(tp: Any, role: Any) -> None:
    assert get_role(tp) == role
