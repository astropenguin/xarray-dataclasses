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


class MaskedImageDataset(ImageDataset):
    __slots__ = ()
    mask: xr.DataArray


class ColoredImageDataset(ImageDataset):
    __slots__ = ()
    tint: xr.DataArray


class MaskedColoredImageDataset(
    MaskedImageDataset, ColoredImageDataset
):
    __slots__ = ()


@datasetclass
class MaskedImageW(ImageW, WithNewX[MaskedImageDataset]):
    mask: Data[Tuple[X, Y], bool]


@datasetclass
class ColoredImageW(ImageW, WithNewX[ColoredImageDataset]):
    tint: Data[Tuple[X, Y], float]


@datasetclass
class MaskedColoredImageW(
    MaskedImageW, ColoredImageW, WithNewX[MaskedColoredImageDataset]
):
    pass


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


def test_dataset_extend_derived():
    masked = MaskedImageW.new([[1]], [[True]])
    assert (
        MaskedImageW.new.__annotations__["return"] == MaskedImageDataset
    )
    assert isinstance(masked, MaskedImageDataset)
    foo_masked(masked)


def test_dataset_extend_diamond():
    mc = MaskedColoredImageW.new(
        data=[[1]], mask=[[True]], tint=[[0.8]]
    )
    assert (
        MaskedColoredImageW.new.__annotations__["return"]
        == MaskedColoredImageDataset
    )
    assert isinstance(mc, MaskedColoredImageDataset)


def foo(imageDataset: ImageDataset):
    pass


def foo_masked(maskedDataset: MaskedImageDataset):
    pass
