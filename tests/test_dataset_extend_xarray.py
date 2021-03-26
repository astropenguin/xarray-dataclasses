"""
Test `extend_xarray` option.

With this option, dataset dataclass `new` should return
an instance which is derived from dataclass as well as xarray.
"""
from functools import singledispatch
from typing import Literal, Tuple
import pytest

import xarray as xr
from xarray_dataclasses import datasetclass, Data

# constants
DIMS = "x", "y"

# type hints
X = Literal[DIMS[0]]
Y = Literal[DIMS[1]]


class ImageDataset:
    __slots__ = tuple()


@datasetclass(xarray_base=ImageDataset)
class Image:
    data: Data[Tuple[X, Y], float]


class ImageMaskedDataset:
    __slots__ = tuple()


@datasetclass(xarray_base=ImageMaskedDataset)
class ImageMasked:
    data: Data[Tuple[X, Y], float]
    mask: Data[Tuple[X, Y], bool]


def test_dataset_extend_xarray():
    image = Image.new([[1, 2], [3, 4]])
    assert isinstance(image, xr.Dataset)
    assert isinstance(image, ImageDataset)


@singledispatch
def check_image_masked(x: xr.Dataset) -> str:
    raise TypeError("Not image?")


@check_image_masked.register
def _(x: ImageDataset):
    return "unmasked"


@check_image_masked.register
def _(x: ImageMaskedDataset):
    return "masked"


def test_dataset_singledispatch():
    generic = xr.Dataset(data_vars=dict())
    with pytest.raises(TypeError):
        check_image_masked(generic)
    assert check_image_masked(Image.new([[1]]) == "unmasked")
    assert (
        check_image_masked(ImageMasked.new([[1]], [[True]])) == "masked"
    )
