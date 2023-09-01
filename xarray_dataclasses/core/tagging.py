__all__ = ["Tag"]


# standard library
from collections.abc import Iterable
from enum import Flag, auto
from functools import reduce
from itertools import chain, filterfalse
from operator import or_
from typing import Annotated, Any, Optional


# dependencies
from typing_extensions import Self, TypeGuard, get_args, get_origin


class Tag(Flag):
    """Collection of tags for annotating types."""

    ATTR = auto()
    """Tag for an attribute-intended type."""

    COORD = auto()
    """Tag for a coordinate-intended type."""

    DIMS = auto()
    """Tag for data dimensions."""

    DTYPE = auto()
    """Tag for a data type."""

    MULTIPLE = auto()
    """Tag for multiplicity."""

    NAME = auto()
    """Tag for a name-intended type."""

    VAR = auto()
    """Tag for a variable-intended type."""

    INTENT = ATTR | COORD | NAME | VAR
    """Union of intent-related tags."""

    OPTION = MULTIPLE
    """Union of option-related tags."""

    TYPE = DIMS | DTYPE
    """Union of type-related tags."""

    ANY = INTENT | OPTION | TYPE
    """Union of all tags."""

    def annotates(self, tp: Any) -> bool:
        """Check if the tag annotates a type hint."""
        tags = filter(type(self).creates, get_args(tp))
        return bool(self & type(self).union(tags))

    @classmethod
    def creates(cls, obj: Any) -> TypeGuard[Self]:
        """Check if Tag is the type of an object."""
        return isinstance(obj, cls)

    @classmethod
    def union(cls, tags: Iterable[Self]) -> Self:
        """Create a tag as an union of tags."""
        return reduce(or_, tags, cls(0))

    def __repr__(self) -> str:
        """Return the bracket-style string of the tag."""
        return str(self)

    def __str__(self) -> str:
        """Return the bracket-style string of the tag."""
        return f"<{str(self.name).lower()}>"


def gen_annotated(tp: Any) -> Iterable[Any]:
    """Generate all annotated types in a type hint."""
    if get_origin(tp) is Annotated:
        yield tp
        yield from gen_annotated(get_args(tp)[0])
    else:
        yield from chain(*map(gen_annotated, get_args(tp)))


def get_tagged(
    tp: Any,
    bound: Tag = Tag.ANY,
    keep_annotations: bool = False,
) -> Optional[Any]:
    """Extract the first tagged type from a type hint."""
    for tagged in filter(bound.annotates, gen_annotated(tp)):
        return tagged if keep_annotations else get_args(tagged)[0]


def get_tags(tp: Any, bound: Tag = Tag.ANY) -> tuple[Tag, ...]:
    """Extract all tags from the first tagged type."""
    tagged = get_tagged(tp, bound, True)
    return tuple(filter(Tag.creates, get_args(tagged)[1:]))


def get_nontags(tp: Any, bound: Tag = Tag.ANY) -> tuple[Any, ...]:
    """Extract all except tags from the first tagged type."""
    tagged = get_tagged(tp, bound, True)
    return tuple(filterfalse(Tag.creates, get_args(tagged)[1:]))
