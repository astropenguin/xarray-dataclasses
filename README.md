# xarray-dataclasses

[![PyPI](https://img.shields.io/pypi/v/xarray-dataclasses.svg?label=PyPI&style=flat-square)](https://pypi.org/project/xarray-dataclasses/)
[![Python](https://img.shields.io/pypi/pyversions/xarray-dataclasses.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/project/xarray-dataclasses/)
[![Test](https://img.shields.io/github/workflow/status/astropenguin/xarray-dataclasses/Test?logo=github&label=Test&style=flat-square)](https://github.com/astropenguin/xarray-dataclasses/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=License&style=flat-square)](LICENSE)

xarray extension for DataArray and Dataset classes

## TL;DR

xarray-dataclasses is a Python package for creating DataArray and Dataset classes in the same manner as [the Python's native dataclass].
Here is an example code of what the package provides:

```python
from xarray_dataclasses import Coord, Data, dataarrayclass


@dataarrayclass
class Image:
    """DataArray that represents an image."""

    data: Data[tuple['x', 'y'], float]
    x: Coord['x', int] = 0
    y: Coord['y', int] = 0


# create a DataArray instance
image = Image.new([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])


# create a DataArray instance filled with ones
ones = Image.ones((2, 2), x=[0, 1], y=[0, 1])
```

### Features

- DataArray or Dataset instances with fixed dimensions, data type, and coordinates can easily be created.
- NumPy-like special functions such as ``ones()`` are provided as class methods.
- 100% compatible with [the Python's native dataclass].
- 100% compatible with static type check by [Pyright].

### Installation

```shell
$ pip install xarray-dataclasses
```

## Introduction

[xarray] is useful for handling labeled multi-dimensional data, but it is a bit troublesome to create a DataArray or Dataset instance with fixed dimensions, data type, or coordinates.
For example, let us think about the following specifications of DataArray instances:

- Dimensions of data must be `('x', 'y')`.
- Data type of data must be `float`.
- Data type of dimensions must be `int`.
- Default value of dimensions must be `0`.

Then a function to create a spec-compliant DataArray instance is something like this:

```python
import numpy as np
import xarray as xr


def spec_dataarray(data, x=None, y=None):
    """Create a spec-comliant DataArray instance."""
    data = np.array(data)

    if x is None:
        x = np.zeros(data.shape[0])
    else:
        x = np.array(x)

    if y is None:
        y = np.zeros(data.shape[1])
    else:
        y = np.array(y)

    return xr.DataArray(
        data=data.astype(float),
        dims=('x', 'y'),
        coords={
            'x': ('x', x.astype(int)),
            'y': ('y', y.astype(int)),
        },
    )


dataarray = spec_dataarray([[0, 1], [2, 3]])
```

The issues are (1) it is hard to figure out the specs from the code and (2) it is hard to reuse the code, for example, to add a new coordinate to the original specs.

[xarray-dataclasses](#xarray-dataclasses) resolves them by defining the specs as a dataclass with dedicated type hints:

```python
from xarray_dataclasses import Coord, Data, dataarrayclass


@dataarrayclass
class Specs:
    data: Data[tuple['x', 'y'], float]
    x: Coord['x', int] = 0
    y: Coord['y', int] = 0


dataarray = Specs.new([[0, 1], [2, 3]])
```

The specs are now much easier to read:
The type hints, `Data[<dims>, <dtype>]` and `Coord[<dims>, <dtype>]`, have complete information of DataArray creation.
The default values are given as class variables.

The class decorator, `@dataarrayclass`, converts a class to [the Python's native dataclass] and add class methods such as `new()` to it.
The extension of the specs is then easy by class inheritance.


<!-- References -->
[the Python's native dataclass]: https://docs.python.org/3/library/dataclasses.html
[Pyright]: https://github.com/microsoft/pyright
[xarray]: https://xarray.pydata.org/en/stable/index.html
