# standard library
from typing import Any, Dict, Iterable, Tuple, Union


# dependencies
from pytest import mark
from typing_extensions import Annotated as Ann
from xarray_dataclasses.core.tagging import Tag, get_tags


data_annotates = [
    (Tag.ANY, Ann[Any, Tag.ATTR], True),
    (Tag.ANY, Ann[Any, Tag.COORD], True),
    (Tag.ANY, Ann[Any, Tag.DATA], True),
    (Tag.ANY, Ann[Any, Tag.DIMS], True),
    (Tag.ANY, Ann[Any, Tag.DTYPE], True),
    (Tag.ANY, Ann[Any, Tag.MULTIPLE], True),
    (Tag.ANY, Ann[Any, Tag.ORIGIN], True),
    (Tag.ANY, Ann[Any, Tag.FIELD], True),
    (Tag.ANY, Ann[Any, Tag.OPTION], True),
    (Tag.ANY, Ann[Any, Tag.ANY], True),
    (Tag.FIELD, Ann[Any, Tag.OPTION], False),
    (Tag.OPTION, Ann[Any, Tag.FIELD], False),
    (Tag.ANY, Any, False),
]

data_creates = [
    (Tag.ATTR, True),
    (Tag.COORD, True),
    (Tag.DATA, True),
    (Tag.DIMS, True),
    (Tag.DTYPE, True),
    (Tag.MULTIPLE, True),
    (Tag.ORIGIN, True),
    (Tag.FIELD, True),
    (Tag.OPTION, True),
    (Tag.ANY, True),
    (object(), False),
]

data_union = [
    ([Tag.ATTR, Tag.COORD, Tag.DATA], Tag.FIELD),
    ([Tag.DIMS, Tag.DTYPE, Tag.MULTIPLE, Tag.ORIGIN], Tag.OPTION),
    ([Tag.FIELD, Tag.OPTION], Tag.ANY),
]

data_get_tags = [
    (Any, Tag.ANY, ()),
    (Any, Tag.FIELD, ()),
    (Any, Tag.OPTION, ()),
    (Ann[Any, Tag.ATTR], Tag.ANY, (Tag.ATTR,)),
    (Ann[Any, Tag.ATTR], Tag.FIELD, (Tag.ATTR,)),
    (Ann[Any, Tag.ATTR], Tag.OPTION, ()),
    (Ann[Any, Tag.COORD], Tag.ANY, (Tag.COORD,)),
    (Ann[Any, Tag.COORD], Tag.FIELD, (Tag.COORD,)),
    (Ann[Any, Tag.COORD], Tag.OPTION, ()),
    (Ann[Any, Tag.DATA], Tag.ANY, (Tag.DATA,)),
    (Ann[Any, Tag.DATA], Tag.FIELD, (Tag.DATA,)),
    (Ann[Any, Tag.DATA], Tag.OPTION, ()),
    (Ann[Any, Tag.DIMS], Tag.ANY, (Tag.DIMS,)),
    (Ann[Any, Tag.DIMS], Tag.FIELD, ()),
    (Ann[Any, Tag.DIMS], Tag.OPTION, (Tag.DIMS,)),
    (Ann[Any, Tag.DTYPE], Tag.ANY, (Tag.DTYPE,)),
    (Ann[Any, Tag.DTYPE], Tag.FIELD, ()),
    (Ann[Any, Tag.DTYPE], Tag.OPTION, (Tag.DTYPE,)),
    (Ann[Any, Tag.MULTIPLE], Tag.ANY, (Tag.MULTIPLE,)),
    (Ann[Any, Tag.MULTIPLE], Tag.FIELD, ()),
    (Ann[Any, Tag.MULTIPLE], Tag.OPTION, (Tag.MULTIPLE,)),
    (Ann[Any, Tag.ORIGIN], Tag.ANY, (Tag.ORIGIN,)),
    (Ann[Any, Tag.ORIGIN], Tag.OPTION, (Tag.ORIGIN,)),
    (Ann[Any, Tag.ORIGIN], Tag.FIELD, ()),
    (Ann[Any, Tag.DATA, object()], Tag.ANY, (Tag.DATA,)),
    (Ann[Any, Tag.DATA, object()], Tag.FIELD, (Tag.DATA,)),
    (Ann[Any, Tag.DATA, object()], Tag.OPTION, ()),
    (Dict[str, Ann[Any, Tag.DATA]], Tag.ANY, (Tag.DATA,)),
    (Dict[str, Ann[Any, Tag.DATA]], Tag.FIELD, (Tag.DATA,)),
    (Dict[str, Ann[Any, Tag.DATA]], Tag.OPTION, ()),
    (Ann[Any, Tag.DATA, Tag.MULTIPLE], Tag.ANY, (Tag.DATA, Tag.MULTIPLE)),
    (Ann[Any, Tag.DATA, Tag.MULTIPLE], Tag.FIELD, (Tag.DATA, Tag.MULTIPLE)),
    (Ann[Any, Tag.DATA, Tag.MULTIPLE], Tag.OPTION, (Tag.DATA, Tag.MULTIPLE)),
    (Union[Ann[Any, Tag.DATA], Ann[Any, Tag.ORIGIN]], Tag.ANY, (Tag.DATA,)),
    (Union[Ann[Any, Tag.DATA], Ann[Any, Tag.ORIGIN]], Tag.FIELD, (Tag.DATA,)),
    (Union[Ann[Any, Tag.DATA], Ann[Any, Tag.ORIGIN]], Tag.OPTION, (Tag.ORIGIN,)),
]


@mark.parametrize("tag, tp, expected", data_annotates)
def test_annotates(tag: Tag, tp: Any, expected: bool) -> None:
    assert tag.annotates(tp) == expected


@mark.parametrize("obj, expected", data_creates)
def test_creates(obj: Any, expected: bool) -> None:
    assert Tag.creates(obj) == expected


@mark.parametrize("tags, expected", data_union)
def test_union(tags: Iterable[Tag], expected: Tag) -> None:
    assert Tag.union(tags) is expected


@mark.parametrize("tp, bound, expected", data_get_tags)
def test_get_tags(tp: Any, bound: Tag, expected: Tuple[Tag, ...]) -> None:
    assert get_tags(tp, bound) == expected
