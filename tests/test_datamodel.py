# standard library
from dataclasses import dataclass
from typing import Tuple


# third-party packages
from typing_extensions import Literal


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
    assert xaxis_model.attr[0].name == "units"
    assert xaxis_model.attr[0].value == "pixel"
    assert xaxis_model.attr[0].type == "builtins.str"


def test_xaxis_data() -> None:
    assert xaxis_model.data[0].name == "data"
    assert xaxis_model.data[0].type == {"dims": ("x",), "dtype": "int"}
    assert xaxis_model.data[0].factory is None


def test_yaxis_attr() -> None:
    assert yaxis_model.attr[0].name == "units"
    assert yaxis_model.attr[0].value == "pixel"
    assert yaxis_model.attr[0].type == "builtins.str"


def test_yaxis_data() -> None:
    assert yaxis_model.data[0].name == "data"
    assert yaxis_model.data[0].type == {"dims": ("y",), "dtype": "int"}
    assert yaxis_model.data[0].factory is None


def test_matrix_coord() -> None:
    assert image_model.coord[0].name == "mask"
    assert image_model.coord[0].type == {"dims": ("x", "y"), "dtype": "bool"}
    assert image_model.coord[0].factory is None

    assert image_model.coord[1].name == "x"
    assert image_model.coord[1].type == {"dims": ("x",), "dtype": "int"}
    assert image_model.coord[1].factory is XAxis

    assert image_model.coord[2].name == "y"
    assert image_model.coord[2].type == {"dims": ("y",), "dtype": "int"}
    assert image_model.coord[2].factory is YAxis


def test_matrix_data() -> None:
    assert image_model.data[0].name == "data"
    assert image_model.data[0].type == {"dims": ("x", "y"), "dtype": "float"}


def test_image_data() -> None:
    assert color_model.data[0].name == "red"
    assert color_model.data[0].type == {"dims": ("x", "y"), "dtype": "float"}
    assert color_model.data[0].factory is Image

    assert color_model.data[1].name == "green"
    assert color_model.data[1].type == {"dims": ("x", "y"), "dtype": "float"}
    assert color_model.data[1].factory is Image

    assert color_model.data[2].name == "blue"
    assert color_model.data[2].type == {"dims": ("x", "y"), "dtype": "float"}
    assert color_model.data[2].factory is Image
