# standard library
from dataclasses import MISSING


# dependencies
from test_data import Latitude, Longitude, Weather, weather
from xarray_dataclasses.v2.specs import Spec


# test data
spec = Spec.from_dataclass(Weather)
spec_updated = spec @ weather


# test functions
def test_temp() -> None:
    field = spec.fields.of_data[0]

    assert field.id == "temp"
    assert field.name == "Temperature ({.temp_unit})"
    assert field.role == "data"
    assert field.default is MISSING
    assert field.type is None
    assert field.dims == ("x", "y", "time")
    assert field.dtype == "float64"


def test_temp_updated() -> None:
    field = spec_updated.fields.of_data[0]

    assert field.id == "temp"
    assert field.name == "Temperature (deg C)"
    assert field.role == "data"
    assert (field.default == weather.temp).all()
    assert field.type is None
    assert field.dims == ("x", "y", "time")
    assert field.dtype == "float64"


def test_prec() -> None:
    field = spec.fields.of_data[1]

    assert field.id == "prec"
    assert field.name == "Precipitation ({.prec_unit})"
    assert field.role == "data"
    assert field.default is MISSING
    assert field.type is None
    assert field.dims == ("x", "y", "time")
    assert field.dtype == "float64"


def test_prec_updated() -> None:
    field = spec_updated.fields.of_data[1]

    assert field.id == "prec"
    assert field.name == "Precipitation (mm)"
    assert field.role == "data"
    assert (field.default == weather.prec).all()
    assert field.type is None
    assert field.dims == ("x", "y", "time")
    assert field.dtype == "float64"


def test_lon() -> None:
    field = spec.fields.of_coord[0]

    assert field.id == "lon"
    assert field.name == "Longitude"
    assert field.role == "coord"
    assert field.default is MISSING
    assert field.type is Longitude
    assert field.dims == ("x", "y")
    assert field.dtype == "float64"


def test_lon_updated() -> None:
    field = spec_updated.fields.of_coord[0]

    assert field.id == "lon"
    assert field.name == "Longitude"
    assert field.role == "coord"
    assert (field.default == weather.lon).all()
    assert field.type is Longitude
    assert field.dims == ("x", "y")
    assert field.dtype == "float64"


def test_lat() -> None:
    field = spec.fields.of_coord[1]

    assert field.id == "lat"
    assert field.name == "Latitude"
    assert field.role == "coord"
    assert field.default is MISSING
    assert field.type is Latitude
    assert field.dims == ("x", "y")
    assert field.dtype == "float64"


def test_lat_updated() -> None:
    field = spec_updated.fields.of_coord[1]

    assert field.id == "lat"
    assert field.name == "Latitude"
    assert field.role == "coord"
    assert (field.default == weather.lat).all()
    assert field.type is Latitude
    assert field.dims == ("x", "y")
    assert field.dtype == "float64"


def test_time() -> None:
    field = spec.fields.of_coord[2]

    assert field.id == "time"
    assert field.name == "time"
    assert field.role == "coord"
    assert field.default is MISSING
    assert field.type is None
    assert field.dims == ("time",)
    assert field.dtype == "datetime64[ns]"


def test_time_updated() -> None:
    field = spec_updated.fields.of_coord[2]

    assert field.id == "time"
    assert field.name == "time"
    assert field.role == "coord"
    assert (field.default == weather.time).all()
    assert field.type is None
    assert field.dims == ("time",)
    assert field.dtype == "datetime64[ns]"


def test_reference_time() -> None:
    field = spec.fields.of_coord[3]

    assert field.id == "reference_time"
    assert field.name == "reference_time"
    assert field.role == "coord"
    assert field.default is MISSING
    assert field.type is None
    assert field.dims == ()
    assert field.dtype == "datetime64[ns]"


def test_reference_time_updated() -> None:
    field = spec_updated.fields.of_coord[3]

    assert field.id == "reference_time"
    assert field.name == "reference_time"
    assert field.role == "coord"
    assert field.default == weather.reference_time
    assert field.type is None
    assert field.dims == ()
    assert field.dtype == "datetime64[ns]"
