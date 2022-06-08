# xarray-dataclasses

[![Release](https://img.shields.io/pypi/v/xarray-dataclasses?label=Release&color=cornflowerblue&style=flat-square)](https://pypi.org/project/xarray-dataclasses/)
[![Python](https://img.shields.io/pypi/pyversions/xarray-dataclasses?label=Python&color=cornflowerblue&style=flat-square)](https://pypi.org/project/xarray-dataclasses/)
![Downloads](https://img.shields.io/pypi/dm/xarray-dataclasses?label=Downloads&color=cornflowerblue&style=flat-square)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.4624819-cornflowerblue?style=flat-square)](https://doi.org/10.5281/zenodo.4624819)
[![Tests](https://img.shields.io/github/workflow/status/astropenguin/xarray-dataclasses/Tests?label=Tests&style=flat-square)](https://github.com/astropenguin/xarray-dataclasses/actions)

xarray extension for typed DataArray and Dataset creation


## Overview

xarray-dataclasses is a Python package that makes it easy to create [xarray]'s DataArray and Dataset objects that are "typed" (i.e. fixed dimensions, data type, coordinates, attributes, and name) using [the Python's dataclass]:

```python
from dataclasses import dataclass
from typing import Literal
from xarray_dataclasses import AsDataArray, Coord, Data


X = Literal["x"]
Y = Literal["y"]


@dataclass
class Image(AsDataArray):
    """2D image as DataArray."""

    data: Data[tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0
```

### Features

- Typed DataArray or Dataset objects can easily be created:
  ```python
  image = Image.new([[0, 1], [2, 3]], [0, 1], [0, 1])
  ```
- NumPy-like filled-data creation is also available:
  ```python
  image = Image.zeros([2, 2], x=[0, 1], y=[0, 1])
  ```
- Support for features by [the Python's dataclass] (`field`, `__post_init__`, ...).
- Support for static type check by [Pyright].

### Installation

```shell
pip install xarray-dataclasses
```

## Basic usage

xarray-dataclasses uses [the Python's dataclass].
Data (or data variables), coordinates, attributes, and a name of DataArray or Dataset objects will be defined as dataclass fields by special type hints (`Data`, `Coord`, `Attr`, `Name`), respectively.
Note that the following code is supposed in the examples below.

```python
from dataclasses import dataclass
from typing import Literal
from xarray_dataclasses import AsDataArray, AsDataset
from xarray_dataclasses import Attr, Coord, Data, Name


X = Literal["x"]
Y = Literal["y"]
```

### Data field

Data field is a field whose value will become the data of a DataArray object or a data variable of a Dataset object.
The type hint `Data[TDims, TDtype]` fixes the dimensions and the data type of the object.
Here are some examples of how to specify them.

Type hint | Inferred dimensions
--- | ---
`Data[tuple[()], ...]` | `()`
`Data[Literal["x"], ...]` | `("x",)`
`Data[tuple[Literal["x"]], ...]` | `("x",)`
`Data[tuple[Literal["x"], Literal["y"]], ...]` | `("x", "y")`

Type hint | Inferred data type
--- | ---
`Data[..., Any]` | `None`
`Data[..., None]` | `None`
`Data[..., float]` | `numpy.dtype("float64")`
`Data[..., numpy.float128]` | `numpy.dtype("float128")`
`Data[..., Literal["datetime64[ns]"]]` | `numpy.dtype("<M8[ns]")`

### Coordinate field

Coordinate field is a field whose value will become a coordinate of a DataArray or a Dataset object.
The type hint `Coord[TDims, TDtype]` fixes the dimensions and the data type of the object.

### Attribute field

Attribute field is a field whose value will become an attribute of a DataArray or a Dataset object.
The type hint `Attr[TAttr]` specifies the type of the value, which is used only for static type check.

### Name field

Name field is a field whose value will become the name of a DataArray object.
The type hint `Name[TName]` specifies the type of the value, which is used only for static type check.

### DataArray class

DataArray class is a dataclass that defines typed DataArray specifications.
Exactly one data field is allowed in a DataArray class.
The second and subsequent data fields are just ignored in DataArray creation.

```python
@dataclass
class Image(AsDataArray):
    """2D image as DataArray."""

    data: Data[tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0
    units: Attr[str] = "cd / m^2"
    name: Name[str] = "luminance"
```

A DataArray object will be created by a class method `new()`:

```python
Image.new([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])

<xarray.DataArray "luminance" (x: 2, y: 2)>
array([[0., 1.],
       [2., 3.]])
Coordinates:
  * x        (x) int64 0 1
  * y        (y) int64 0 1
Attributes:
    units:    cd / m^2
```

NumPy-like class methods (`zeros()`, `ones()`, ...) are also available:

```python
Image.ones((3, 3))

<xarray.DataArray "luminance" (x: 3, y: 3)>
array([[1., 1., 1.],
       [1., 1., 1.],
       [1., 1., 1.]])
Coordinates:
  * x        (x) int64 0 0 0
  * y        (y) int64 0 0 0
Attributes:
    units:    cd / m^2
```

### Dataset class

Dataset class is a dataclass that defines typed Dataset specifications.
Multiple data fields are allowed to define the data variables of the object.

```python
@dataclass
class ColorImage(AsDataset):
    """2D color image as Dataset."""

    red: Data[tuple[X, Y], float]
    green: Data[tuple[X, Y], float]
    blue: Data[tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0
    units: Attr[str] = "cd / m^2"
```

A Dataset object will be created by a class method `new()`:

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
    units:    cd / m^2
```

## Advanced usage

### Coordof and Dataof type hints

xarray-dataclasses provides advanced type hints, `Coordof` and `Dataof`.
Unlike `Data` and `Coord`, they specify a dataclass that defines a DataArray class.
This is useful when users want to add metadata to dimensions for [plotting].
For example:

```python
from xarray_dataclasses import Coordof


@dataclass
class XAxis:
    data: Data[X, int]
    long_name: Attr[str] = "x axis"
    units: Attr[str] = "pixel"


@dataclass
class YAxis:
    data: Data[Y, int]
    long_name: Attr[str] = "y axis"
    units: Attr[str] = "pixel"


@dataclass
class Image(AsDataArray):
    """2D image as DataArray."""

    data: Data[tuple[X, Y], float]
    x: Coordof[XAxis] = 0
    y: Coordof[YAxis] = 0
```

### General data variable names in Dataset creation

Due to the limitation of Python's parameter names, it is not possible to define data variable names that contain white spaces, for example.
In such cases, please define DataArray classes of each data variable so that they have name fields and specify them by `Dataof` in a Dataset class.
Then the values of the name fields will be used as data variable names.
For example:

```python
@dataclass
class Red:
    data: Data[tuple[X, Y], float]
    name: Name[str] = "Red image"


@dataclass
class Green:
    data: Data[tuple[X, Y], float]
    name: Name[str] = "Green image"


@dataclass
class Blue:
    data: Data[tuple[X, Y], float]
    name: Name[str] = "Blue image"


@dataclass
class ColorImage(AsDataset):
    """2D color image as Dataset."""

    red: Dataof[Red]
    green: Dataof[Green]
    blue: Dataof[Blue]
```

```python
ColorImage.new(
    [[0, 0], [0, 0]],
    [[1, 1], [1, 1]],
    [[2, 2], [2, 2]],
)

<xarray.Dataset>
Dimensions:      (x: 2, y: 2)
Dimensions without coordinates: x, y
Data variables:
    Red image    (x, y) float64 0.0 0.0 0.0 0.0
    Green image  (x, y) float64 1.0 1.0 1.0 1.0
    Blue image   (x, y) float64 2.0 2.0 2.0 2.0
```

### Customization of DataArray or Dataset creation

For customization, users can add a special class attribute, `__dataoptions__`, to a DataArray or Dataset class.
A custom factory for DataArray or Dataset creation is only supported in the current implementation.


```python
import xarray as xr
from xarray_dataclasses import DataOptions


class Custom(xr.DataArray):
    """Custom DataArray."""

    __slots__ = ()

    def custom_method(self) -> bool:
        """Custom method."""
        return True


@dataclass
class Image(AsDataArray):
    """2D image as DataArray."""

    data: Data[tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0

    __dataoptions__ = DataOptions(Custom)


image = Image.ones([3, 3])
isinstance(image, Custom)  # True
image.custom_method()  # True
```

### DataArray and Dataset creation without shorthands

xarray-dataclasses provides functions, `asdataarray` and `asdataset`.
This is useful when users do not want to inherit the mix-in class (`AsDataArray` or `AsDataset`) in a DataArray or Dataset dataclass.
For example:

```python
from xarray_dataclasses import asdataarray


@dataclass
class Image:
    """2D image as DataArray."""

    data: Data[tuple[X, Y], float]
    x: Coord[X, int] = 0
    y: Coord[Y, int] = 0


image = asdataarray(Image([[0, 1], [2, 3]], [0, 1], [0, 1]))
```


<!-- References -->
[Pyright]: https://github.com/microsoft/pyright
[the Python's dataclass]: https://docs.python.org/3/library/dataclasses.html
[xarray]: https://xarray.pydata.org/en/stable/index.html
[plotting]: https://xarray.pydata.org/en/stable/user-guide/plotting.html#simple-example
