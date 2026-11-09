"""Compatibility lookups for the device-wide XY field monitor name."""

from __future__ import annotations

from collections.abc import Collection
from typing import overload

_FIELD_MONITOR_ALIASES = {"overview_xy": "field_xy", "field_xy": "overview_xy"}


def resolve_monitor_name(name: str, names: Collection[str]) -> str:
    """Prefer an exact name, falling back to the other XY field-image name."""
    if name not in names:
        alternate = _FIELD_MONITOR_ALIASES.get(name)
        if alternate is not None and alternate in names:
            return alternate
    return name


class MonitorNameAliases[T](dict[str, T]):
    """Dictionary retaining stored keys while supporting field-image aliases."""

    def __missing__(self, name: str) -> T:
        """Resolve an XY alias without adding duplicate iteration keys."""
        alternate = resolve_monitor_name(name, self.keys())
        if alternate != name:
            return self[alternate]
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        """Include the compatibility name in membership checks."""
        if isinstance(name, str):
            name = resolve_monitor_name(name, self.keys())
        return dict.__contains__(self, name)

    @overload
    def get(self, key: object, default: None = None, /) -> T | None: ...

    @overload
    def get(self, key: object, default: T, /) -> T: ...

    @overload
    def get[D](self, key: object, default: D, /) -> T | D: ...

    def get(self, key: object, default: object = None, /) -> object:
        """Return an exact or aliased monitor, otherwise the supplied default."""
        if isinstance(key, str):
            key = resolve_monitor_name(key, self.keys())
        return super().get(key, default)
