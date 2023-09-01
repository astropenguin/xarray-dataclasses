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
    (Tag.ANY, Ann[Any, Tag.INTENT], True),
    (Tag.ANY, Ann[Any, Tag.OPTION], True),
    (Tag.ANY, Ann[Any, Tag.TYPE], True),
    (Tag.ANY, Ann[Any, Tag.ANY], True),
    (Tag.INTENT, Ann[Any, Tag.OPTION], False),
    (Tag.OPTION, Ann[Any, Tag.TYPE], False),
    (Tag.TYPE, Ann[Any, Tag.INTENT], False),
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
    (Tag.INTENT, True),
    (Tag.OPTION, True),
    (Tag.TYPE, True),
    (Tag.ANY, True),
    (object(), False),
]

testdata_union = [
    ([Tag.ATTR, Tag.COORD, Tag.NAME, Tag.VAR], Tag.INTENT),
    ([Tag.DIMS, Tag.DTYPE], Tag.TYPE),
    ([Tag.MULTIPLE], Tag.OPTION),
    ([Tag.INTENT, Tag.OPTION, Tag.TYPE], Tag.ANY),
]

testdata_get_tags = [
    (Any, Tag.ANY, ()),
    (Any, Tag.INTENT, ()),
    (Any, Tag.OPTION, ()),
    (Ann[Any, Tag.ATTR], Tag.ANY, (Tag.ATTR,)),
    (Ann[Any, Tag.ATTR], Tag.INTENT, (Tag.ATTR,)),
    (Ann[Any, Tag.ATTR], Tag.OPTION, ()),
    (Ann[Any, Tag.ATTR], Tag.TYPE, ()),
    (Ann[Any, Tag.COORD], Tag.ANY, (Tag.COORD,)),
    (Ann[Any, Tag.COORD], Tag.INTENT, (Tag.COORD,)),
    (Ann[Any, Tag.COORD], Tag.OPTION, ()),
    (Ann[Any, Tag.COORD], Tag.TYPE, ()),
    (Ann[Any, Tag.DIMS], Tag.ANY, (Tag.DIMS,)),
    (Ann[Any, Tag.DIMS], Tag.INTENT, ()),
    (Ann[Any, Tag.DIMS], Tag.OPTION, ()),
    (Ann[Any, Tag.DIMS], Tag.TYPE, (Tag.DIMS,)),
    (Ann[Any, Tag.DTYPE], Tag.ANY, (Tag.DTYPE,)),
    (Ann[Any, Tag.DTYPE], Tag.INTENT, ()),
    (Ann[Any, Tag.DTYPE], Tag.OPTION, ()),
    (Ann[Any, Tag.DTYPE], Tag.TYPE, (Tag.DTYPE,)),
    (Ann[Any, Tag.MULTIPLE], Tag.ANY, (Tag.MULTIPLE,)),
    (Ann[Any, Tag.MULTIPLE], Tag.INTENT, ()),
    (Ann[Any, Tag.MULTIPLE], Tag.OPTION, (Tag.MULTIPLE,)),
    (Ann[Any, Tag.MULTIPLE], Tag.TYPE, ()),
    (Ann[Any, Tag.NAME], Tag.ANY, (Tag.NAME,)),
    (Ann[Any, Tag.NAME], Tag.INTENT, (Tag.NAME,)),
    (Ann[Any, Tag.NAME], Tag.OPTION, ()),
    (Ann[Any, Tag.NAME], Tag.TYPE, ()),
    (Ann[Any, Tag.VAR], Tag.ANY, (Tag.VAR,)),
    (Ann[Any, Tag.VAR], Tag.INTENT, (Tag.VAR,)),
    (Ann[Any, Tag.VAR], Tag.OPTION, ()),
    (Ann[Any, Tag.VAR], Tag.TYPE, ()),
    (Ann[Any, Tag.VAR, object()], Tag.ANY, (Tag.VAR,)),
    (Ann[Any, Tag.VAR, object()], Tag.INTENT, (Tag.VAR,)),
    (Ann[Any, Tag.VAR, object()], Tag.OPTION, ()),
    (Ann[Any, Tag.VAR, object()], Tag.TYPE, ()),
    (Ann[Any, Tag.VAR, Tag.MULTIPLE], Tag.ANY, (Tag.VAR, Tag.MULTIPLE)),
    (Ann[Any, Tag.VAR, Tag.MULTIPLE], Tag.INTENT, (Tag.VAR, Tag.MULTIPLE)),
    (Ann[Any, Tag.VAR, Tag.MULTIPLE], Tag.OPTION, (Tag.VAR, Tag.MULTIPLE)),
    (Ann[Any, Tag.VAR, Tag.MULTIPLE], Tag.TYPE, ()),
    (dict[str, Ann[Any, Tag.VAR]], Tag.ANY, (Tag.VAR,)),
    (dict[str, Ann[Any, Tag.VAR]], Tag.INTENT, (Tag.VAR,)),
    (dict[str, Ann[Any, Tag.VAR]], Tag.OPTION, ()),
    (dict[str, Ann[Any, Tag.VAR]], Tag.TYPE, ()),
    (Union[Ann[Any, Tag.VAR], Ann[Any, Tag.COORD]], Tag.ANY, (Tag.VAR,)),
    (Union[Ann[Any, Tag.VAR], Ann[Any, Tag.COORD]], Tag.INTENT, (Tag.VAR,)),
    (Union[Ann[Any, Tag.VAR], Ann[Any, Tag.COORD]], Tag.OPTION, ()),
    (Union[Ann[Any, Tag.VAR], Ann[Any, Tag.COORD]], Tag.TYPE, ()),
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
