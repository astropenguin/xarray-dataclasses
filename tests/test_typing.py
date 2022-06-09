# standard library
from typing import Any, Tuple, Union


# dependencies
import numpy as np
from pytest import mark
from typing_extensions import Annotated as Ann
from typing_extensions import Literal as L


# submodules
from xarray_dataclasses.typing import (
    Attr,
    Coord,
    Data,
    FType,
    Name,
    get_dims,
    get_dtype,
    get_ftype,
    get_name,
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

testdata_ftype = [
    (Attr[Any], FType.ATTR),
    (Data[Any, Any], FType.DATA),
    (Coord[Any, Any], FType.COORD),
    (Name[Any], FType.NAME),
    (Any, FType.OTHER),
    (Ann[Attr[Any], "attr"], FType.ATTR),
    (Ann[Data[Any, Any], "data"], FType.DATA),
    (Ann[Coord[Any, Any], "coord"], FType.COORD),
    (Ann[Name[Any], "name"], FType.NAME),
    (Ann[Any, "other"], FType.OTHER),
    (Union[Ann[Attr[Any], "attr"], Ann[Any, "any"]], FType.ATTR),
    (Union[Ann[Data[Any, Any], "data"], Ann[Any, "any"]], FType.DATA),
    (Union[Ann[Coord[Any, Any], "coord"], Ann[Any, "any"]], FType.COORD),
    (Union[Ann[Name[Any], "name"], Ann[Any, "any"]], FType.NAME),
    (Union[Ann[Any, "other"], Ann[Any, "any"]], FType.OTHER),
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


# test functions
@mark.parametrize("tp, dims", testdata_dims)
def test_get_dims(tp: Any, dims: Any) -> None:
    assert get_dims(tp) == dims


@mark.parametrize("tp, dtype", testdata_dtype)
def test_get_dtype(tp: Any, dtype: Any) -> None:
    assert get_dtype(tp) == dtype


@mark.parametrize("tp, ftype", testdata_ftype)
def test_get_ftype(tp: Any, ftype: Any) -> None:
    assert get_ftype(tp) == ftype


@mark.parametrize("tp, name", testdata_name)
def test_get_name(tp: Any, name: Any) -> None:
    assert get_name(tp) == name
