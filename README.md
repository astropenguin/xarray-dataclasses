# xarray-dataclasses

[![PyPI](https://img.shields.io/pypi/v/xarray-dataclasses.svg?label=PyPI&style=flat-square)](https://pypi.org/project/xarray-dataclasses/)
[![Python](https://img.shields.io/pypi/pyversions/xarray-dataclasses.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/project/xarray-dataclasses/)
[![Test](https://img.shields.io/github/workflow/status/astropenguin/xarray-dataclasses/Test?logo=github&label=Test&style=flat-square)](https://github.com/astropenguin/xarray-dataclasses/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?label=License&style=flat-square)](LICENSE)

xarray extension for dataarray classes

## TL;DR

xarray-dataclasses is a third-party Python package which helps to create DataArray classes in the same manner as [the Python's native dataclass].
Here is an introduction code of what the package provides:

```python
from xarray_dataclasses import DataArray, dataarrayclass


@dataarrayclass
class Image:
    """DataArray class to represent images."""

    data: DataArray[('x', 'y'), float]
    x: DataArray['x', int] = 0
    y: DataArray['y', int] = 0
```

The key features are:

```python
# create a DataArray instance
image = Image.new([[0, 1], [2, 3]], x=[0, 1], y=[0, 1])

# create a DataArray instance filled with ones
ones = Image.ones((2, 2), x=[0, 1], y=[0, 1])
```

- Custom DataArray instances with fixed dimensions, datatype, and coordinates can easily be created.
- NumPy-like special functions like ``ones()`` are provided as class methods.

## Requirements

- **Python:** 3.7, 3.8, or 3.9 (tested by the author)
- **Dependencies:** See [pyproject.toml](pyproject.toml)

## Installation

```shell
$ pip install xarray-dataclasses
```

<!-- References -->
[the Python's native dataclass]: https://docs.python.org/3/library/dataclasses.html
