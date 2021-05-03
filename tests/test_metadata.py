# third-party packages
import xarray_dataclasses


# test functions
def test_author():
    assert xarray_dataclasses.__author__ == "Akio Taniguchi"


def test_version():
    assert xarray_dataclasses.__version__ == "0.3.1"
