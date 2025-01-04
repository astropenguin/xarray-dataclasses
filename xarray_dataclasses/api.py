__all__ = ["asdataarray", "asdataset", "asxarray"]


# standard library
from dataclasses import replace
from typing import Any, ForwardRef, Literal, Optional, overload

# dependencies
from dataspecs import ID, ROOT, Spec, Specs
from numpy import asarray, array
from typing_extensions import get_args, get_origin
from xarray import DataArray, Dataset, Variable
from .typing import (
    DataClass,
    DataClassOf,
    Factory,
    HashDict,
    PAny,
    TAny,
    TDataArray,
    TDataset,
    TXarray,
    Tag,
)


# type hints
Attrs = HashDict[Any]
Vars = HashDict[Variable]


@overload
def asdataarray(
    obj: DataClassOf[PAny, TDataArray],
    /,
    *,
    factory: None = None,
) -> TDataArray: ...


@overload
def asdataarray(
    obj: DataClass[PAny],
    /,
    *,
    factory: Factory[TDataArray],
) -> TDataArray: ...


@overload
def asdataarray(
    obj: DataClass[PAny],
    /,
    *,
    factory: None = None,
) -> DataArray: ...


def asdataarray(obj: Any, /, *, factory: Any = None) -> Any:
    """Create a DataArray object from a dataclass object."""
    ...


@overload
def asdataset(
    obj: DataClassOf[PAny, TDataset],
    /,
    *,
    factory: None = None,
) -> TDataset: ...


@overload
def asdataset(
    obj: DataClass[PAny],
    /,
    *,
    factory: Factory[TDataset],
) -> TDataset: ...


@overload
def asdataset(
    obj: DataClass[PAny],
    /,
    *,
    factory: None = None,
) -> Dataset: ...


def asdataset(obj: Any, /, *, factory: Any = None) -> Any:
    """Create a Dataset object from a dataclass object."""
    ...


@overload
def asxarray(
    obj: DataClassOf[PAny, TXarray],
    /,
    *,
    factory: None = None,
) -> TXarray: ...


@overload
def asxarray(
    obj: DataClass[PAny],
    /,
    *,
    factory: Factory[TXarray],
) -> TXarray: ...


def asxarray(obj: Any, /, *, factory: Any = None) -> Any:
    """Create a DataArray/set object from a dataclass object."""
    ...


def get_attrs(specs: Specs[Spec[Any]], /, *, at: ID = ROOT) -> Attrs:
    """Create attributes from data specs."""
    attrs: Attrs = {}

    for spec in specs[at.children][Tag.ATTR]:
        options = specs[spec.id.children]
        factory = maybe(options[Tag.FACTORY].unique).data or identity
        name = maybe(options[Tag.NAME].unique).data or spec.id.name

        if Tag.MULTIPLE not in spec.tags:
            spec = replace(spec, data={name: spec.data})

        for name, data in spec[HashDict[Any]].data.items():
            attrs[name] = factory(data)

    return attrs


def get_vars(specs: Specs[Spec[Any]], of: Tag, /, *, at: ID = ROOT) -> Vars:
    """Create variables of given tag from data specs."""
    vars: Vars = {}

    for spec in specs[at.children][of]:
        options = specs[spec.id.children]
        attrs = get_attrs(specs, at=spec.id)
        factory = maybe(options[Tag.FACTORY].unique).data or Variable
        name = maybe(options[Tag.NAME].unique).data or spec.id.name

        if (type_ := maybe(options[Tag.DIMS].unique).type) is None:
            raise RuntimeError("Could not find any data spec for dims.")
        elif get_origin(type_) is tuple:
            dims = tuple(str(unwrap(arg)) for arg in get_args(type_))
        else:
            dims = (str(unwrap(type_)),)

        if (type_ := maybe(options[Tag.DTYPE].unique).type) is None:
            raise RuntimeError("Could not find any data spec for dims.")
        elif type_ is type(None) or type_ is Any:
            dtype = None
        else:
            dtype = unwrap(type_)

        if Tag.MULTIPLE not in spec.tags:
            spec = replace(spec, data={name: spec.data})

        for name, data in spec[HashDict[Any]].data.items():
            if not (data := asarray(data, dtype)).ndim:
                data = array(data, ndmin=len(dims))

            vars[name] = factory(attrs=attrs, data=data, dims=dims)

    return vars


def identity(obj: TAny, /) -> TAny:
    """Identity function used for the default factory."""
    return obj


def maybe(obj: Optional[Spec[Any]], /) -> Spec[Any]:
    """Return a dummy (``None``-filled) data spec if an object is not one."""
    return Spec(ROOT, (), None, None) if obj is None else obj


def unwrap(obj: Any, /) -> Any:
    """Unwrap if an object is a literal or a forward reference."""
    if get_origin(obj) is Literal:
        return args[0] if len(args := get_args(obj)) == 1 else obj
    elif isinstance(obj, ForwardRef):
        return obj.__forward_arg__
    else:
        return obj
