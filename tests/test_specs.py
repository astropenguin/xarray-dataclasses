# standard library
from dataclasses import MISSING, dataclass
from typing import Tuple


# dependencies
import numpy as np
import xarray as xr
from typing_extensions import Annotated as Ann
from typing_extensions import Literal as L
from xarray_dataclasses.specs import DataOptions, DataSpec
from xarray_dataclasses.typing import Attr, Coordof, Data, Name


# type hints
DataDims = Tuple[L["lon"], L["lat"], L["time"]]


# test datasets
@dataclass
class Lon:
    """Specification of relative longitude."""

    data: Data[L["lon"], float]
    units: Attr[str] = "deg"
    name: Name[str] = "Relative longitude"


@dataclass
class Lat:
    """Specification of relative latitude."""

    data: Data[L["lat"], float]
    units: Attr[str] = "m"
    name: Name[str] = "Relative latitude"


@dataclass
class Time:
    """Specification of time."""

    data: Data[L["time"], L["datetime64[ns]"]]
    name: Name[str] = "Time in UTC"


@dataclass
class Weather:
    """Time-series spatial weather information at a location."""

    temperature: Ann[Data[DataDims, float], "Temperature"]
    humidity: Ann[Data[DataDims, float], "Humidity"]
    wind_speed: Ann[Data[DataDims, float], "Wind speed"]
    wind_direction: Ann[Data[DataDims, float], "Wind direction"]
    lon: Coordof[Lon]
    lat: Coordof[Lat]
    time: Coordof[Time]
    location: Attr[str] = "Tokyo"
    longitude: Attr[float] = 139.69167
    latitude: Attr[float] = 35.68944
    name: Name[str] = "weather"


# test functions
def test_temperature() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_data["temperature"]

    assert spec.name == "Temperature"
    assert spec.role == "data"
    assert spec.dims == ("lon", "lat", "time")
    assert spec.dtype == np.dtype("f8")
    assert spec.default is MISSING
    assert spec.origin is None


def test_humidity() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_data["humidity"]

    assert spec.name == "Humidity"
    assert spec.role == "data"
    assert spec.dims == ("lon", "lat", "time")
    assert spec.dtype == np.dtype("f8")
    assert spec.default is MISSING
    assert spec.origin is None


def test_wind_speed() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_data["wind_speed"]

    assert spec.name == "Wind speed"
    assert spec.role == "data"
    assert spec.dims == ("lon", "lat", "time")
    assert spec.dtype == np.dtype("f8")
    assert spec.default is MISSING
    assert spec.origin is None


def test_wind_direction() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_data["wind_direction"]

    assert spec.name == "Wind direction"
    assert spec.role == "data"
    assert spec.dims == ("lon", "lat", "time")
    assert spec.dtype == np.dtype("f8")
    assert spec.default is MISSING
    assert spec.origin is None


def test_lon() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_coord["lon"]

    assert spec.name == "Relative longitude"
    assert spec.role == "coord"
    assert spec.dims == ("lon",)
    assert spec.dtype == np.dtype("f8")
    assert spec.default is MISSING
    assert spec.origin is Lon


def test_lat() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_coord["lat"]

    assert spec.name == "Relative latitude"
    assert spec.role == "coord"
    assert spec.dims == ("lat",)
    assert spec.dtype == np.dtype("f8")
    assert spec.default is MISSING
    assert spec.origin is Lat


def test_time() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_coord["time"]

    assert spec.name == "Time in UTC"
    assert spec.role == "coord"
    assert spec.dims == ("time",)
    assert spec.dtype == np.dtype("M8[ns]")
    assert spec.default is MISSING
    assert spec.origin is Time


def test_location() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_attr["location"]

    assert spec.name == "location"
    assert spec.role == "attr"
    assert spec.type is str
    assert spec.default == "Tokyo"


def test_longitude() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_attr["longitude"]

    assert spec.name == "longitude"
    assert spec.role == "attr"
    assert spec.type is float
    assert spec.default == 139.69167


def test_latitude() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_attr["latitude"]

    assert spec.name == "latitude"
    assert spec.role == "attr"
    assert spec.type is float
    assert spec.default == 35.68944


def test_name() -> None:
    spec = DataSpec.from_dataclass(Weather).specs.of_name["name"]

    assert spec.name == "name"
    assert spec.role == "name"
    assert spec.type is str
    assert spec.default == "weather"


def test_dataoptions() -> None:
    options = DataOptions(xr.DataArray)

    assert DataSpec().options.factory is type(None)
    assert DataSpec(options=options).options.factory is xr.DataArray
