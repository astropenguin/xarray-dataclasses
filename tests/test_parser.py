# standard library
from dataclasses import dataclass
from typing import Tuple


# third-party packages
from typing_extensions import Literal


# submodules
from xarray_dataclasses.parser import parse
from xarray_dataclasses.typing import Attr, Coord, Data, Name


# type hints
X = Literal["x"]
Y = Literal["y"]


# test datasets
@dataclass
class Image:
    data: Data[Tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0
    dpi: Attr[int] = 100
    name: Name[str] = "image"


structure = parse(Image)


# test functions
def test_attr() -> None:
    assert structure.attr[0].name == "dpi"
    assert structure.attr[0].value == 100
    assert structure.attr[0].type == int


def test_coord() -> None:
    assert structure.coord[0].name == "x"
    assert structure.coord[0].dims == ("x",)
    assert structure.coord[0].dtype == "int"
    assert structure.coord[1].name == "y"
    assert structure.coord[1].dims == ("y",)
    assert structure.coord[1].dtype == "int"


def test_data() -> None:
    assert structure.data[0].name == "data"
    assert structure.data[0].dims == ("x", "y")
    assert structure.data[0].dtype == "float"


def test_name() -> None:
    assert structure.name[0].name == "name"
    assert structure.name[0].value == "image"
    assert structure.name[0].type == str
