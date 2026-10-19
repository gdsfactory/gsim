"""Base model for the concern-grouped configuration of a simulation.

``gsim.fdtd``, ``gsim.tcad`` and ``gsim.thermal`` configure a simulation
through a few concern objects (geometry, domain, materials, solver,
...). Each is a validated pydantic model that can be set one attribute
at a time or several at once by calling it.

A concern can be mutated in place at any time, so a Simulation that
keeps a mesh or a solve compares snapshots of its inputs
(:func:`fingerprint`) rather than watching assignments.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, Self, overload

import numpy as np
from pydantic import BaseModel, ConfigDict

__all__ = [
    "Concern",
    "MutableModel",
    "NamedModels",
    "Same",
    "component_digest",
    "fingerprint",
]


class MutableModel(BaseModel):
    """Validated model that supports concise batch updates."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    def __call__(self, **updates: Any) -> Self:
        """Validate and apply several settings atomically.

        The current values are passed back as they are, not dumped, so a
        field holding an arbitrary object (a component, a stack) survives.
        """
        updated = type(self).model_validate({**dict(self), **updates})
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, getattr(updated, field_name))
        return self


class Concern[T]:
    """A Simulation attribute that validates whatever replaces it.

    A model concern takes a model, a mapping of its fields, or None for
    its defaults; a collection takes the collection or a list of items.
    """

    def __init__(self, kind: type[T]) -> None:
        """Remember the concern's type."""
        self._kind = kind
        self._attribute = ""

    def __set_name__(self, owner: type, name: str) -> None:
        """Store the value under a private name."""
        self._attribute = f"_{name}"

    @overload
    def __get__(self, instance: None, owner: type) -> Self: ...

    @overload
    def __get__(self, instance: object, owner: type) -> T: ...

    def __get__(self, instance: object | None, owner: type) -> T | Self:
        """Return the concern."""
        if instance is None:
            return self
        return getattr(instance, self._attribute)

    def __set__(self, instance: object, value: Any) -> None:
        """Validate and store a replacement."""
        kind: Any = self._kind
        if isinstance(value, kind):
            pass
        elif issubclass(kind, BaseModel):
            value = kind() if value is None else kind.model_validate(value)
        else:
            value = kind(value)
        setattr(instance, self._attribute, value)


class NamedModels[T: BaseModel]:
    """Insertion-ordered collection of models with a unique ``name``."""

    _kind: str
    _model: type[T]

    def __init__(self, items: Iterable[T | Mapping[str, Any]] | None = None) -> None:
        """Validate an optional initial list."""
        self._items: dict[str, T] = {}
        for item in items or []:
            self._insert(self._model.model_validate(item))

    def _insert(self, item: T) -> T:
        """Store an item, requiring a unique name."""
        name = item.name  # ty: ignore[unresolved-attribute]
        if name in self._items:
            raise ValueError(f"{self._kind} {name!r} already exists")
        self._items[name] = item
        return item

    def remove(self, name: str) -> T:
        """Remove and return one item by name."""
        return self._items.pop(name)

    def clear(self) -> None:
        """Remove every item."""
        self._items.clear()

    def __getitem__(self, name: str) -> T:
        """Return one item by name."""
        return self._items[name]

    def __contains__(self, name: object) -> bool:
        """Whether an item of that name is declared."""
        return name in self._items

    def __iter__(self) -> Iterator[T]:
        """Iterate over the items in insertion order."""
        return iter(list(self._items.values()))

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._items)

    def __repr__(self) -> str:
        """Show the items."""
        return f"{type(self).__name__}({list(self._items.values())!r})"


class Same:
    """Snapshot of an object compared by identity.

    Holds the object, so its address cannot be reused by another one that
    would then compare equal.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        """Hold the object."""
        self.value = value

    def __eq__(self, other: object) -> bool:
        """The same object, not an equal one."""
        return isinstance(other, Same) and other.value is self.value

    __hash__ = None  # ty: ignore[invalid-assignment]


def component_digest(component: Any) -> Any:
    """The drawn polygons of a component, so an edit in place is seen."""
    try:
        polygons = component.get_polygons_points(by="tuple")
    except Exception:
        return None
    digest = hashlib.sha256()
    for layer in sorted(polygons, key=repr):
        digest.update(repr(layer).encode())
        for points in polygons[layer]:
            digest.update(np.ascontiguousarray(points, dtype=np.float64).tobytes())
    return digest.hexdigest()


def fingerprint(value: Any) -> Any:
    """A comparable snapshot of a configuration value.

    Models and containers are taken apart, arrays by their bytes; any
    other object (a function) by identity.
    """
    if isinstance(value, Same):
        return value
    if isinstance(value, BaseModel):
        return (type(value).__name__, fingerprint(dict(value)))
    if isinstance(value, Mapping):
        return tuple((key, fingerprint(item)) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(fingerprint(item) for item in value)
    if isinstance(value, np.ndarray):
        return ("ndarray", value.shape, value.dtype.str, value.tobytes())
    if value is None or isinstance(value, (str, bytes, int, float, complex, Path)):
        return value
    return Same(value)
