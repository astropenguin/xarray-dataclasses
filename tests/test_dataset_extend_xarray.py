"""
Test `extend_xarray` option.

With this option, dataset dataclass `new` should return
an instance which is derived from dataclass as well as xarray.
"""
from typing import Literal, Tuple
import pytest

import xarray as xr
from xarray_dataclasses import datasetclass, Data
from xarray_dataclasses.typing import DataClassX, WithClass
from xarray_dataclasses.dataset import WithNewX

# constants
DIMS = ("x", "y")

# type hints
X = Literal["x"]
Y = Literal["y"]


class ImageDataset(xr.Dataset):
    __slots__ = ()
    data: xr.DataArray


@datasetclass
class ImageU:
    data: Data[Tuple[X, Y], float]


@datasetclass
class ImageC(WithClass[ImageDataset]):
    data: Data[Tuple[X, Y], float]


@datasetclass
class ImageX(DataClassX[ImageDataset]):
    data: Data[Tuple[X, Y], float]


@datasetclass
class ImageW(WithNewX[ImageDataset]):
    data: Data[Tuple[X, Y], float]


def test_dataset_extend_base():
    image = ImageU.new([[1, 2], [3, 4]])
    assert ImageU.new.__annotations__["return"] == xr.Dataset
    assert not isinstance(image, ImageDataset)


def test_dataset_extend_wc():
    image = ImageU.new([[1, 2], [3, 4]])
    assert ImageU.new.__annotations__["return"] == xr.Dataset
    assert not isinstance(image, ImageDataset)
    foo(image)  # static typing doesn't work?


def test_dataset_extend_dataclass_base():
    image = ImageX.new([[1, 2], [3, 4]])
    assert ImageX.new.__annotations__["return"] == ImageDataset
    assert isinstance(image, ImageDataset)
    foo(image)  # static typing works


def test_dataset_extend_with_new():
    image = ImageW.new([[1, 2], [3, 4]])
    assert ImageW.new.__annotations__["return"] == ImageDataset
    assert isinstance(image, ImageDataset)
    foo(image)


def foo(imageDataset: ImageDataset):
    pass
