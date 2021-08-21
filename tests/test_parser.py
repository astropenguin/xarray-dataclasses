# standard library
from dataclasses import dataclass
from typing import Tuple


# third-party packages
from typing_extensions import Literal


# submodules
from xarray_dataclasses.parser import parse
from xarray_dataclasses.typing import Attr, Coord, Coordof, Data, Dataof, Name


# type hints
X = Literal["x"]
Y = Literal["y"]


# test datasets
@dataclass
class XAxis:
    data: Data[X, int]
    units: Attr[str] = "pixel"
    name: Name[str] = "x axis"


@dataclass
class YAxis:
    data: Data[Y, int]
    units: Attr[str] = "pixel"
    name: Name[str] = "y axis"


@dataclass
class Matrix:
    data: Data[Tuple[X, Y], float]
    mask: Coord[Tuple[X, Y], bool] = False
    x: Coordof[XAxis] = 0
    y: Coordof[YAxis] = 0


@dataclass
class Image:
    red: Dataof[Matrix]
    green: Dataof[Matrix]
    blue: Dataof[Matrix]


xaxis_structure = parse(XAxis)
yaxis_structure = parse(YAxis)
matrix_structure = parse(Matrix)
image_structure = parse(Image)


# test functions
def test_xaxis_attr() -> None:
    assert xaxis_structure.attr[0].name == "units"
    assert xaxis_structure.attr[0].value == "pixel"
    assert xaxis_structure.attr[0].type == "str"


def test_xaxis_data() -> None:
    assert xaxis_structure.data[0].name == "data"
    assert xaxis_structure.data[0].type == {"dims": ("x",), "dtype": "int"}


def test_xaxis_name() -> None:
    assert xaxis_structure.name[0].name == "name"
    assert xaxis_structure.name[0].value == "x axis"
    assert xaxis_structure.name[0].type == "str"


def test_yaxis_attr() -> None:
    assert yaxis_structure.attr[0].name == "units"
    assert yaxis_structure.attr[0].value == "pixel"
    assert yaxis_structure.attr[0].type == "str"


def test_yaxis_data() -> None:
    assert yaxis_structure.data[0].name == "data"
    assert yaxis_structure.data[0].type == {"dims": ("y",), "dtype": "int"}


def test_yaxis_name() -> None:
    assert yaxis_structure.name[0].name == "name"
    assert yaxis_structure.name[0].value == "y axis"
    assert yaxis_structure.name[0].type == "str"


def test_matrix_coord() -> None:
    assert matrix_structure.coord[0].name == "mask"
    assert matrix_structure.coord[0].type == {"dims": ("x", "y"), "dtype": "bool"}
    assert matrix_structure.coord[1].name == "x"
    assert matrix_structure.coord[1].type == "tests.test_parser.XAxis"
    assert matrix_structure.coord[2].name == "y"
    assert matrix_structure.coord[2].type == "tests.test_parser.YAxis"


def test_matrix_data() -> None:
    assert matrix_structure.data[0].name == "data"
    assert matrix_structure.data[0].type == {"dims": ("x", "y"), "dtype": "float"}


def test_image_data() -> None:
    assert image_structure.data[0].name == "red"
    assert image_structure.data[0].type == "tests.test_parser.Matrix"
    assert image_structure.data[1].name == "green"
    assert image_structure.data[1].type == "tests.test_parser.Matrix"
    assert image_structure.data[2].name == "blue"
    assert image_structure.data[2].type == "tests.test_parser.Matrix"
