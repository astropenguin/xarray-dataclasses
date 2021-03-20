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

<!-- References -->
[the Python's native dataclass]: https://docs.python.org/3/library/dataclasses.html
[Pyright]: https://github.com/microsoft/pyright
