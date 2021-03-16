# standard library
from typing import Tuple


# third-party packages
import numpy as np
import xarray as xr
from typing_extensions import Literal


# submodules
from xarray_dataclasses.dataarray import dataarrayclass
from xarray_dataclasses.typing import Attr, Coord, Data, Name


# type hints
X = Literal["x"]
Y = Literal["y"]


# test datasets
@dataarrayclass
class Image:
    data: Data[float, Tuple[X, Y]]
    x: Coord[int, X] = 0
    y: Coord[int, Y] = 0
    dpi: Attr[int] = 100
    name: Name[str] = "image"


expected = xr.DataArray(
    np.ones([10, 10], float),
    dims=("x", "y"),
    coords={
        "x": ("x", np.zeros(10)),
        "y": ("y", np.zeros(10)),
    },
    attrs={"dpi": 100},
    name="image",
)


# test functions
def test_dataarrayclass():
    dataarray = Image.ones([10, 10])

    assert (dataarray == expected).all()
    assert dataarray.dtype == expected.dtype
    assert dataarray.dims == expected.dims
    assert dataarray.attrs == expected.attrs
    assert dataarray.name == expected.name
