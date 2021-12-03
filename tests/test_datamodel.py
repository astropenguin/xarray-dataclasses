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
    item = next(iter(xaxis_model.attr.values()))
    assert item.key == "units"
    assert item.value == "pixel"
    assert item.type == "builtins.str"


def test_xaxis_data() -> None:
    item = next(iter(xaxis_model.data.values()))
    assert item.key == "data"
    assert item.type == {"dims": ("x",), "dtype": "int"}
    assert item.factory is None


def test_yaxis_attr() -> None:
    item = next(iter(yaxis_model.attr.values()))
    assert item.key == "units"
    assert item.value == "pixel"
    assert item.type == "builtins.str"


def test_yaxis_data() -> None:
    item = next(iter(yaxis_model.data.values()))
    assert item.key == "data"
    assert item.type == {"dims": ("y",), "dtype": "int"}
    assert item.factory is None


def test_image_coord() -> None:
    items = iter(image_model.coord.values())

    item = next(items)
    assert item.key == "mask"
    assert item.type == {"dims": ("x", "y"), "dtype": "bool"}
    assert item.factory is None

    item = next(items)
    assert item.key == "x"
    assert item.type == {"dims": ("x",), "dtype": "int"}
    assert item.factory is XAxis

    item = next(items)
    assert item.key == "y"
    assert item.type == {"dims": ("y",), "dtype": "int"}
    assert item.factory is YAxis


def test_image_data() -> None:
    item = next(iter(image_model.data.values()))
    assert item.key == "data"
    assert item.type == {"dims": ("x", "y"), "dtype": "float"}
    assert item.factory is None


def test_color_data() -> None:
    items = iter(color_model.data.values())

    item = next(items)
    assert item.key == "red"
    assert item.type == {"dims": ("x", "y"), "dtype": "float"}
    assert item.factory is Image

    item = next(items)
    assert item.key == "green"
    assert item.type == {"dims": ("x", "y"), "dtype": "float"}
    assert item.factory is Image

    item = next(items)
    assert item.key == "blue"
    assert item.type == {"dims": ("x", "y"), "dtype": "float"}
    assert item.factory is Image
