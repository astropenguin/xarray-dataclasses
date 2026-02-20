# standard library
from dataclasses import dataclass
from typing import Collection, Literal as L, Tuple, Union


# dependencies
import numpy as np
from typing_extensions import Annotated as Ann
from xarray_dataclasses.typing import Attr, Coord, Coordof, Data


# type hints
X = L["x"]
Y = L["y"]
T = L["time"]
_ = Tuple[()]


# test data
@dataclass
class Longitude:
    """Specification of longitude."""

    data: Data[Tuple[X, Y], float]
    units: Attr[str] = "deg"


@dataclass
class Latitude:
    """Specification of latitude."""

    data: Data[Tuple[X, Y], float]
    units: Attr[str] = "deg"


@dataclass
class Weather:
    """Weather information."""

    temp: Ann[Data[Tuple[X, Y, T], float], "Temperature ({.temp_unit})"]
    """Measured temperature."""

    prec: Ann[Data[Tuple[X, Y, T], float], "Precipitation ({.prec_unit})"]
    """Measured precipitation."""

    lon: Union[Ann[Coordof[Longitude], "Longitude"], Collection[float]]
    """Longitude of the measured location."""

    lat: Union[Ann[Coordof[Latitude], "Latitude"], Collection[float]]
    """Latitude of the measured location."""

    time: Coord[T, L["M8[ns]"]]
    """Measured time."""

    reference_time: Union[Coord[_, L["M8[ns]"]], np.datetime64]
    """Reference time."""

    temp_unit: str = "deg C"
    """Unit of the temperature."""

    prec_unit: str = "mm"
    """Unit of the precipitation."""


weather = Weather(
    15 + 8 * np.random.randn(2, 2, 3),
    10 * np.random.rand(2, 2, 3),
    np.array([[-99.83, -99.32], [-99.79, -99.23]]),
    np.array([[42.25, 42.21], [42.63, 42.59]]),
    np.array(["2014-09-06", "2014-09-07", "2014-09-08"], "M8[ns]"),  # type: ignore
    np.datetime64("2014-09-05"),
)
