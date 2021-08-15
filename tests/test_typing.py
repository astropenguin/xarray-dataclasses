# standard library
from typing import Any, ForwardRef, Tuple, Union


# third-party packages
from pytest import mark
from typing_extensions import Annotated, Literal


# submodules
from xarray_dataclasses.typing import get_dims, get_dtype, unannotate


# type hints
x = ForwardRef("x")
y = ForwardRef("y")
X = Literal["x"]
Y = Literal["y"]


# test datasets
testdata_dims = [
    (Tuple[()], ()),
    (X, ("x",)),
    (x, ("x",)),
    (Tuple[X], ("x",)),
    (Tuple[x], ("x",)),  # type: ignore
    (Tuple[X, Y], ("x", "y")),
    (Tuple[x, y], ("x", "y")),  # type: ignore
]

testdata_dtype = [
    (Any, None),
    (int, "int"),
    ("int", "int"),
    (Literal["int"], "int"),
]

testdata_unannotate = [
    (x, x),
    (X, X),
    (Annotated[X, 0], X),
    (Union[Annotated[X, 0], Y], Union[X, Y]),
]


# test functions
@mark.parametrize("t_dims, dims", testdata_dims)
def test_get_dims(t_dims: Any, dims: Any) -> None:
    assert get_dims(t_dims) == dims


@mark.parametrize("t_dtype, dtype", testdata_dtype)
def test_get_dtype(t_dtype: Any, dtype: Any) -> None:
    assert get_dtype(t_dtype) == dtype


@mark.parametrize("t_before, t_after", testdata_unannotate)
def test_unannotate(t_before: Any, t_after: Any) -> None:
    assert unannotate(t_before) == t_after
