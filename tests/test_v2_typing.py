# standard library
from dataclasses import dataclass
from typing import Any, Tuple, Union


# dependencies
import numpy as np
from xarray_dataclasses.v2.typing import (
    Attr,
    Coord,
    Coordof,
    Data,
    Dataof,
    Tag,
    get_dims,
    get_dtype,
    get_name,
    get_tag,
)
from pytest import mark
from typing_extensions import Annotated as Ann, Literal as L


# test data
@dataclass
class DataClass:
    data: Any


testdata_dims = [
    (Coord[Tuple[()], Any], ()),
    (Coord[L["x"], Any], ("x",)),
    (Coord[Tuple[L["x"]], Any], ("x",)),
    (Coord[Tuple[L["x"], L["y"]], Any], ("x", "y")),
    (Coordof[DataClass], None),
    (Data[Tuple[()], Any], ()),
    (Data[L["x"], Any], ("x",)),
    (Data[Tuple[L["x"]], Any], ("x",)),
    (Data[Tuple[L["x"], L["y"]], Any], ("x", "y")),
    (Dataof[DataClass], None),
    (Ann[Coord[L["x"], Any], "coord"], ("x",)),
    (Ann[Coordof[DataClass], "coord"], None),
    (Ann[Data[L["x"], Any], "data"], ("x",)),
    (Ann[Dataof[DataClass], "data"], None),
    (Union[Ann[Coord[L["x"], Any], "coord"], Ann[Any, "any"]], ("x",)),
    (Union[Ann[Coordof[DataClass], "coord"], Ann[Any, "any"]], None),
    (Union[Ann[Data[L["x"], Any], "data"], Ann[Any, "any"]], ("x",)),
    (Union[Ann[Dataof[DataClass], "data"], Ann[Any, "any"]], None),
]

testdata_dtype = [
    (Coord[Any, Any], None),
    (Coord[Any, None], None),
    (Coord[Any, int], np.dtype("i8")),
    (Coord[Any, Union[int, None]], np.dtype("i8")),
    (Coord[Any, L["i8"]], np.dtype("i8")),
    (Coordof[DataClass], None),
    (Data[Any, Any], None),
    (Data[Any, None], None),
    (Data[Any, int], np.dtype("i8")),
    (Data[Any, Union[int, None]], np.dtype("i8")),
    (Data[Any, L["i8"]], np.dtype("i8")),
    (Dataof[DataClass], None),
    (Ann[Coord[Any, float], "coord"], np.dtype("f8")),
    (Ann[Coordof[DataClass], "coord"], None),
    (Ann[Data[Any, float], "data"], np.dtype("f8")),
    (Ann[Dataof[DataClass], "data"], None),
    (Union[Ann[Coord[Any, float], "coord"], Ann[Any, "any"]], np.dtype("f8")),
    (Union[Ann[Coordof[DataClass], "coord"], Ann[Any, "any"]], None),
    (Union[Ann[Data[Any, float], "data"], Ann[Any, "any"]], np.dtype("f8")),
    (Union[Ann[Dataof[DataClass], "data"], Ann[Any, "any"]], None),
]

testdata_name = [
    (Attr[Any], None),
    (Coord[Any, Any], None),
    (Coordof[DataClass], None),
    (Data[Any, Any], None),
    (Dataof[DataClass], None),
    (Any, None),
    (Ann[Attr[Any], "attr"], "attr"),
    (Ann[Coord[Any, Any], "coord"], "coord"),
    (Ann[Coordof[DataClass], "coord"], "coord"),
    (Ann[Data[Any, Any], "data"], "data"),
    (Ann[Dataof[DataClass], "data"], "data"),
    (Ann[Any, "other"], None),
    (Ann[Attr[Any], ..., "attr"], None),
    (Ann[Coord[Any, Any], ..., "coord"], None),
    (Ann[Coordof[DataClass], ..., "coord"], None),
    (Ann[Data[Any, Any], ..., "data"], None),
    (Ann[Dataof[DataClass], ..., "data"], None),
    (Ann[Any, ..., "other"], None),
    (Union[Ann[Attr[Any], "attr"], Ann[Any, "any"]], "attr"),
    (Union[Ann[Coord[Any, Any], "coord"], Ann[Any, "any"]], "coord"),
    (Union[Ann[Coordof[DataClass], "coord"], Ann[Any, "any"]], "coord"),
    (Union[Ann[Data[Any, Any], "data"], Ann[Any, "any"]], "data"),
    (Union[Ann[Dataof[DataClass], "data"], Ann[Any, "any"]], "data"),
    (Union[Ann[Any, "other"], Ann[Any, "any"]], None),
]

testdata_tag = [
    (Attr[Any], Tag.ATTR),
    (Coord[Any, Any], Tag.COORD),
    (Coordof[DataClass], Tag.COORD),
    (Data[Any, Any], Tag.DATA),
    (Dataof[DataClass], Tag.DATA),
    (Any, Tag.OTHER),
    (Ann[Attr[Any], "attr"], Tag.ATTR),
    (Ann[Coord[Any, Any], "coord"], Tag.COORD),
    (Ann[Coordof[DataClass], "coord"], Tag.COORD),
    (Ann[Data[Any, Any], "data"], Tag.DATA),
    (Ann[Dataof[DataClass], "data"], Tag.DATA),
    (Ann[Any, "other"], Tag.OTHER),
    (Union[Ann[Attr[Any], "attr"], Ann[Any, "any"]], Tag.ATTR),
    (Union[Ann[Coord[Any, Any], "coord"], Ann[Any, "any"]], Tag.COORD),
    (Union[Ann[Coordof[DataClass], "coord"], Ann[Any, "any"]], Tag.COORD),
    (Union[Ann[Data[Any, Any], "data"], Ann[Any, "any"]], Tag.DATA),
    (Union[Ann[Dataof[DataClass], "data"], Ann[Any, "any"]], Tag.DATA),
    (Union[Ann[Any, "other"], Ann[Any, "any"]], Tag.OTHER),
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


@mark.parametrize("tp, tag", testdata_tag)
def test_get_tag(tp: Any, tag: Any) -> None:
    assert get_tag(tp) is tag
