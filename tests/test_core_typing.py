# standard library
from sys import version_info
from typing import Any, Union


# dependencies
from pytest import mark
from xarray_dataclasses.core.typing import is_union


if version_info.minor >= 10:
    data_is_union = [
        (Any, False),
        (Union[int, str], True),
        (int | str, True),  # type: ignore
    ]
else:
    data_is_union = [
        (Any, False),
        (Union[int, str], True),
    ]


@mark.parametrize("tp, expected", data_is_union)
def test_get_tags(tp: Any, expected: bool) -> None:
    assert is_union(tp) == expected
