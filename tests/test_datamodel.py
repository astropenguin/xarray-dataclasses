# standard library
from dataclasses import dataclass
from typing import Literal, Tuple


# submodules
from xarray_dataclasses.datamodel import DataModel
from xarray_dataclasses.typing import Attr, Coord, Coordof, Data, Dataof


# type hints
X = Literal["x"]
Y = Literal["y"]


# test datasets
@dataclass
class XAxis:
    data: Data[X, int]
    units: Attr[str] = "pixel"


@dataclass
class YAxis:
    data: Data[Y, int]
    units: Attr[str] = "pixel"


@dataclass
class Image:
    data: Data[Tuple[X, Y], float]
    mask: Coord[Tuple[X, Y], bool] = False
    x: Coordof[XAxis] = 0
    y: Coordof[YAxis] = 0


@dataclass
class ColorImage:
    red: Dataof[Image]
    green: Dataof[Image]
    blue: Dataof[Image]


xaxis_model = DataModel.from_dataclass(XAxis)
yaxis_model = DataModel.from_dataclass(YAxis)
image_model = DataModel.from_dataclass(Image)
color_model = DataModel.from_dataclass(ColorImage)


# test functions
def test_xaxis_attr() -> None:
    units = xaxis_model.attrs[0]
    assert units.name == "units"
    assert units.tag == "attr"
    assert units.type is str
    assert units.value == "pixel"
    assert units.cast == False


def test_xaxis_data() -> None:
    data = xaxis_model.data_vars[0]
    assert data.name == "data"
    assert data.tag == "data"
    assert data.dims == ("x",)
    assert data.dtype == "int"
    assert data.base is None
    assert data.cast == True


def test_yaxis_attr() -> None:
    units = yaxis_model.attrs[0]
    assert units.name == "units"
    assert units.tag == "attr"
    assert units.type is str
    assert units.value == "pixel"
    assert units.cast == False


def test_yaxis_data() -> None:
    data = yaxis_model.data_vars[0]
    assert data.name == "data"
    assert data.tag == "data"
    assert data.dims == ("y",)
    assert data.dtype == "int"
    assert data.base is None
    assert data.cast == True


def test_image_coord() -> None:
    mask = image_model.coords[0]
    assert mask.name == "mask"
    assert mask.tag == "coord"
    assert mask.dims == ("x", "y")
    assert mask.dtype == "bool"
    assert mask.base is None
    assert mask.cast == True

    x = image_model.coords[1]
    assert x.name == "x"
    assert x.tag == "coord"
    assert x.dims == ("x",)
    assert x.dtype == "int"
    assert x.base is XAxis
    assert x.cast == True

    y = image_model.coords[2]
    assert y.name == "y"
    assert y.tag == "coord"
    assert y.dims == ("y",)
    assert y.dtype == "int"
    assert y.base is YAxis
    assert y.cast == True


def test_image_data() -> None:
    data = image_model.data_vars[0]
    assert data.name == "data"
    assert data.tag == "data"
    assert data.dims == ("x", "y")
    assert data.dtype == "float"
    assert data.base is None
    assert data.cast == True


def test_color_data() -> None:
    red = color_model.data_vars[0]
    assert red.name == "red"
    assert red.tag == "data"
    assert red.dims == ("x", "y")
    assert red.dtype == "float"
    assert red.base is Image
    assert red.cast == True

    green = color_model.data_vars[1]
    assert green.name == "green"
    assert green.tag == "data"
    assert green.dims == ("x", "y")
    assert green.dtype == "float"
    assert green.base is Image
    assert green.cast == True

    blue = color_model.data_vars[2]
    assert blue.name == "blue"
    assert blue.tag == "data"
    assert blue.dims == ("x", "y")
    assert blue.dtype == "float"
    assert blue.base is Image
    assert blue.cast == True
