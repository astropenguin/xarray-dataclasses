# xarray-dataclasses

[![PyPI](https://img.shields.io/pypi/v/xarray-dataclasses.svg?label=PyPI&style=flat-square)](https://pypi.org/project/xarray-dataclasses/)
[![Python](https://img.shields.io/pypi/pyversions/xarray-dataclasses.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/project/xarray-dataclasses/)
[![Test](https://img.shields.io/github/workflow/status/astropenguin/xarray-dataclasses/Test?logo=github&label=Test&style=flat-square)](https://github.com/astropenguin/xarray-dataclasses/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=License&style=flat-square)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.4624819-blue?style=flat-square)](https://doi.org/10.5281/zenodo.4624819)

xarray extension for typed DataArray and Dataset creation


## TL;DR

xarray-dataclasses is a Python package for creating typed DataArray and Dataset objects of [xarray] using [the Python's dataclass].
Here is an example code of what the package provides:

```python
from dataclasses import dataclass
from xarray_dataclasses import AsDataArray, Coord, Data


@dataclass
class Image(AsDataArray):
    """DataArray that represents a 2D image."""

    data: Data[tuple['x', 'y'], float]
    x: Coord['x', int] = 0
    y: Coord['y', int] = 0


# create a DataArray object
image = Image.new([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])


# create a DataArray object filled with ones
ones = Image.ones((2, 2), x=[0, 1], y=[0, 1])
```

### Features

- DataArray or Dataset objects with fixed dimensions, data type, and coordinates can easily be created.
- NumPy-like special functions such as ``ones()`` are provided as class methods.
- Compatible with [the Python's dataclass].
- Compatible with static type check by [Pyright].

### Installation

```shell
$ pip install xarray-dataclasses
```


## Introduction

[xarray] is useful for handling labeled multi-dimensional data, but it is a bit troublesome to create a DataArray or Dataset object with fixed dimensions, data type, or coordinates (referred to as typed DataArray or typed Dataset, hereafter).
For example, let us think about the following specifications of DataArray objects:

- Dimensions of data must be `('x', 'y')`.
- Data type of data must be `float`.
- Data type of dimensions must be `int`.
- Default value of dimensions must be `0`.

Then a function to create a typed DataArray object is something like this:

```python
import numpy as np
import xarray as xr


def specs(data, x=None, y=None):
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


dataarray = specs([[0, 1], [2, 3]])
```

The issues are (1) it is not easy to figure out the specs from the code and (2) it is not easy to reuse the code, for example, to add a new coordinate to the original specs.

[xarray-dataclasses](#xarray-dataclasses) resolves them by defining the specs as a dataclass with dedicated type hints:

```python
from dataclasses import dataclass
from xarray_dataclasses import AsDataArray, Coord, Data


@dataclass
class Specs(AsDataArray):
    data: Data[tuple['x', 'y'], float]
    x: Coord['x', int] = 0
    y: Coord['y', int] = 0


dataarray = Specs.new([[0, 1], [2, 3]])
```

Now the specs become much easier to read:
The type hints, `Data[TDims, TDtype]` and `Coord[TDims, TDtype]`, have complete information of DataArray creation.
The default values are given as class variables.

`AsDataArray` is a mix-in class that provides class methods such as `new()`.
The extension of the specs is then easy by class inheritance.

## Basic usage

xarray-dataclasses uses [the Python's dataclass] (please learn how to use it before proceeding).
Data (or data variables), coordinates, attribute members, and name of a DataArray or Dataset object are defined as dataclass fields with the following dedicated type hints.

### `Data` type

`Data[TDims, TDtype]` specifies the field whose value will become the data of a DataArray object or a member of the data variables of a Dataset object.
It accepts two type variables, `TDims` and `TDtype`, for fixing dimensions and data type, respectively.
For example:

| Type hint | Inferred dims | Inferred dtype |
| --- | --- | --- |
| `Data['x', typing.Any]` | `('x',)` | `None` (no type casting) |
| `Data['x', int]` | `('x',)` | `numpy.dtype('int64')` |
| `Data['x', float]` | `('x',)` | `numpy.dtype('float64')` |
| `Data[tuple['x', 'y'], float]` | `('x', 'y')` | `numpy.dtype('float64')` |

Note: for Python 3.7 and 3.8, use `typing.Tuple` instead of `tuple`.

### `Coord` type

`Coord[TDims, TDtype]` specifies the field whose value will become a coordinate of a DataArray or Dataset object.
Similar to `Data`, it accepts two type variables, `TDims` and `TDtype`, for fixing dimensions and data type, respectively.

### `Attr` type

`Attr[T]` specifies the field whose value will become a member of the attributes (attrs) of a DataArray or Dataset object.
It accepts a type variable, `T`, for specifying the type of the value.
Note that the value will not be cast to the type.

### `Name` type

`Name[T]` specifies the field whose value will become the name of a DataArray.
It accepts a type variable, `T`, for specifying the type of the value.
Note that the value will not be cast to the type.

### DataArray class

DataArray class is a dataclass that defines typed DataArray creation.
For example:

```python
from dataclasses import dataclass
from xarray_dataclasses import AsDataArray, Attr, Coord, Data, Name


@dataclass
class Image(AsDataArray):
    """DataArray that represents a 2D image."""

    data: Data[tuple['x', 'y'], float]
    x: Coord['x', int] = 0
    y: Coord['y', int] = 0
    units: Attr[str] = 'dimensionless'
    name: Name[str] = 'default'
```

where exactly one `Data`-type field is allowed.
If more than two `Data`-type fields exist, the second and subsequent fields are ignored.
A typed DataArray object is created by a shorthand method, `new()`:

```python
Image.new([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])

<xarray.DataArray 'default' (x: 2, y: 2)>
array([[0., 1.],
       [2., 3.]])
Coordinates:
  * x        (x) int64 0 1
  * y        (y) int64 0 1
Attributes:
    units:    dimensionless
```

DataArray class has NumPy-like `empty()`, `zeros()`, `ones()`, `full()` methods:

```python
Image.ones((3, 3), name='flat')

<xarray.DataArray 'flat' (x: 3, y: 3)>
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
Coordinates:
  * x        (x) int64 0 0 0
  * y        (y) int64 0 0 0
Attributes:
    units:    dimensionless
```

### Dataset class

Dataset class is a dataclass that defines typed Dataset creation.
For example:

```python
from dataclasses import dataclass
from xarray_dataclasses import AsDataset, Attr, Coord, Data


@dataclass
class ColorImage(AsDataset):
    """Dataset that represents a 2D color image."""

    red: Data[tuple['x', 'y'], float]
    green: Data[tuple['x', 'y'], float]
    blue: Data[tuple['x', 'y'], float]
    x: Coord['x', int] = 0
    y: Coord['y', int] = 0
    units: Attr[str] = 'dimensionless'
```

where multiple `Data`-type fields are allowed.
A typed Dataset object is created by a shorthand method, `new()`:

```python
ColorImage.new(
    [[0, 0], [0, 0]],  # red
    [[1, 1], [1, 1]],  # green
    [[2, 2], [2, 2]],  # blue
)

<xarray.Dataset>
Dimensions:  (x: 2, y: 2)
Coordinates:
  * x        (x) int64 0 0
  * y        (y) int64 0 0
Data variables:
    red      (x, y) float64 0.0 0.0 0.0 0.0
    green    (x, y) float64 1.0 1.0 1.0 1.0
    blue     (x, y) float64 2.0 2.0 2.0 2.0
Attributes:
    units:    dimensionless
```

## Advanced usage

### `Dataof` and `Coordof` types

xarray-dataclasses provides advanced type hints, `Dataof[T]` and `Coordof[T]`.
Unlike `Data` and `Coord`, they accept a dataclass that defines a DataArray class.
This is useful, for example, when users want to add metadata to dimensions for [plotting].
For example:

```python
from dataclasses import dataclass
from xarray_dataclasses import AsDataset, Attr, Coordof, Data, Dataof


@dataclass
class XAxis:
    data: Data['x', int]
    long_name: Attr[str] = 'x axis'
    units: Attr[str] = 'pixel'


@dataclass
class YAxis:
    data: Data['y', int]
    long_name: Attr[str] = 'y axis'
    units: Attr[str] = 'pixel'


@dataclass
class Image:
    data: Data[tuple['x', 'y'], float]
    x: Coordof[XAxis] = 0
    y: Coordof[YAxis] = 0


@dataclass
class ColorImage(AsDataset):
    red: Dataof[Image]
    green: Dataof[Image]
    blue: Dataof[Image]
```

### Custom DataArray or Dataset factory

Users can use a custom DataArray or Dataset factory by defining a special class attribute, `__dataarray_factory__`, or `__dataset_factory__`.
For example:

```python
import xarray as xr
from dataclasses import dataclass
from xarray_dataclasses import AsDataArray, Coord, Data


class Custom(xr.DataArray):
    __slots__ = ()

    def custom_method(self) -> None:
        print('Custom method!')


@dataclass
class Image(AsDataArray):
    data: Data[tuple['x', 'y'], float]
    x: Coord['x', int] = 0
    y: Coord['y', int] = 0
    __dataarray_factory__ = Custom


image = Image.ones([3, 3])
image.custom_method() # Custom method!
```

### DataArray or Dataset creation without shorthands

xarray-dataclasses provides functions, `asdataarray` and `asdataset`, for DataArray or Dataset creation without shorthand methods.
This is useful, for example, users do not want to inherit the mix-in class (`AsDataArray` or `AsDataset`) in their codes.
For example:

```python
from dataclasses import dataclass
from xarray_dataclasses import asdataarray, Coord, Data


@dataclass
class Image:
    data: Data[tuple['x', 'y'], float]
    x: Coord['x', int] = 0
    y: Coord['y', int] = 0


image = asdataarray(Image([[0, 1], [2, 3]], x=[0, 1], y=[0, 1]))
```

### Static type check by [Pyright]

If users want to make your code compatible with [Pyright], please use the literal type for defining dimensions.
For example:

```python
from dataclasses import dataclass
from typing import Literal
from xarray_dataclasses import AsDataArray, Coord, Data


X = Literal['x']
Y = Literal['y']


@dataclass
class Image(AsDataArray):
    data: Data[tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0
```

Note: for Python 3.7, use `typing_extension.Literal` instead of `typing.Literal`.

<!-- References -->
[Pyright]: https://github.com/microsoft/pyright
[the Python's dataclass]: https://docs.python.org/3/library/dataclasses.html
[xarray]: https://xarray.pydata.org/en/stable/index.html
[plotting]: https://xarray.pydata.org/en/stable/user-guide/plotting.html#simple-example
