# standard library
from collections.abc import Iterable
from typing import Annotated as Ann, Any, Union


# dependencies
from pytest import mark
from xarray_dataclasses.core.tagging import Tag, get_tags


testdata_annotates = [
    (Tag.ANY, Ann[Any, Tag.ATTR], True),
    (Tag.ANY, Ann[Any, Tag.COORD], True),
    (Tag.ANY, Ann[Any, Tag.DIMS], True),
    (Tag.ANY, Ann[Any, Tag.DTYPE], True),
    (Tag.ANY, Ann[Any, Tag.MULTIPLE], True),
    (Tag.ANY, Ann[Any, Tag.NAME], True),
    (Tag.ANY, Ann[Any, Tag.VAR], True),
    (Tag.ANY, Ann[Any, Tag.FIELD], True),
    (Tag.ANY, Ann[Any, Tag.OPTION], True),
    (Tag.ANY, Ann[Any, Tag.ANY], True),
    (Tag.FIELD, Ann[Any, Tag.OPTION], False),
    (Tag.OPTION, Ann[Any, Tag.FIELD], False),
    (Tag.ANY, Any, False),
]

testdata_creates = [
    (Tag.ATTR, True),
    (Tag.COORD, True),
    (Tag.DIMS, True),
    (Tag.DTYPE, True),
    (Tag.MULTIPLE, True),
    (Tag.NAME, True),
    (Tag.VAR, True),
    (Tag.FIELD, True),
    (Tag.OPTION, True),
    (Tag.ANY, True),
    (object(), False),
]

testdata_union = [
    ([Tag.ATTR, Tag.COORD, Tag.NAME, Tag.VAR], Tag.FIELD),
    ([Tag.DIMS, Tag.DTYPE, Tag.MULTIPLE], Tag.OPTION),
    ([Tag.FIELD, Tag.OPTION], Tag.ANY),
]

testdata_get_tags = [
    (Any, Tag.ANY, ()),
    (Any, Tag.FIELD, ()),
    (Any, Tag.OPTION, ()),
    (Ann[Any, Tag.ATTR], Tag.ANY, (Tag.ATTR,)),
    (Ann[Any, Tag.ATTR], Tag.FIELD, (Tag.ATTR,)),
    (Ann[Any, Tag.ATTR], Tag.OPTION, ()),
    (Ann[Any, Tag.COORD], Tag.ANY, (Tag.COORD,)),
    (Ann[Any, Tag.COORD], Tag.FIELD, (Tag.COORD,)),
    (Ann[Any, Tag.COORD], Tag.OPTION, ()),
    (Ann[Any, Tag.DIMS], Tag.ANY, (Tag.DIMS,)),
    (Ann[Any, Tag.DIMS], Tag.FIELD, ()),
    (Ann[Any, Tag.DIMS], Tag.OPTION, (Tag.DIMS,)),
    (Ann[Any, Tag.DTYPE], Tag.ANY, (Tag.DTYPE,)),
    (Ann[Any, Tag.DTYPE], Tag.FIELD, ()),
    (Ann[Any, Tag.DTYPE], Tag.OPTION, (Tag.DTYPE,)),
    (Ann[Any, Tag.MULTIPLE], Tag.ANY, (Tag.MULTIPLE,)),
    (Ann[Any, Tag.MULTIPLE], Tag.FIELD, ()),
    (Ann[Any, Tag.MULTIPLE], Tag.OPTION, (Tag.MULTIPLE,)),
    (Ann[Any, Tag.NAME], Tag.ANY, (Tag.NAME,)),
    (Ann[Any, Tag.NAME], Tag.FIELD, (Tag.NAME,)),
    (Ann[Any, Tag.NAME], Tag.OPTION, ()),
    (Ann[Any, Tag.VAR], Tag.ANY, (Tag.VAR,)),
    (Ann[Any, Tag.VAR], Tag.FIELD, (Tag.VAR,)),
    (Ann[Any, Tag.VAR], Tag.OPTION, ()),
    (Ann[Any, Tag.VAR, object()], Tag.ANY, (Tag.VAR,)),
    (Ann[Any, Tag.VAR, object()], Tag.FIELD, (Tag.VAR,)),
    (Ann[Any, Tag.VAR, object()], Tag.OPTION, ()),
    (Ann[Any, Tag.VAR, Tag.MULTIPLE], Tag.ANY, (Tag.VAR, Tag.MULTIPLE)),
    (Ann[Any, Tag.VAR, Tag.MULTIPLE], Tag.FIELD, (Tag.VAR, Tag.MULTIPLE)),
    (Ann[Any, Tag.VAR, Tag.MULTIPLE], Tag.OPTION, (Tag.VAR, Tag.MULTIPLE)),
    (dict[str, Ann[Any, Tag.VAR]], Tag.ANY, (Tag.VAR,)),
    (dict[str, Ann[Any, Tag.VAR]], Tag.FIELD, (Tag.VAR,)),
    (dict[str, Ann[Any, Tag.VAR]], Tag.OPTION, ()),
    (Union[Ann[Any, Tag.VAR], Ann[Any, Tag.COORD]], Tag.ANY, (Tag.VAR,)),
    (Union[Ann[Any, Tag.VAR], Ann[Any, Tag.COORD]], Tag.FIELD, (Tag.VAR,)),
    (Union[Ann[Any, Tag.VAR], Ann[Any, Tag.COORD]], Tag.OPTION, ()),
]


@mark.parametrize("tag, tp, expected", testdata_annotates)
def test_annotates(tag: Tag, tp: Any, expected: bool) -> None:
    assert tag.annotates(tp) == expected


@mark.parametrize("obj, expected", testdata_creates)
def test_creates(obj: Any, expected: bool) -> None:
    assert Tag.creates(obj) == expected


@mark.parametrize("tags, expected", testdata_union)
def test_union(tags: Iterable[Tag], expected: Tag) -> None:
    assert Tag.union(tags) is expected


@mark.parametrize("tp, bound, expected", testdata_get_tags)
def test_get_tags(tp: Any, bound: Tag, expected: tuple[Tag, ...]) -> None:
    assert get_tags(tp, bound) == expected
