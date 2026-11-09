"""Apply PDK overlap priority to local and serialized mode geometry."""

from collections.abc import Iterable, Mapping
from typing import Any


def order_by_mesh_order(entries: Iterable[Any]) -> list[Any]:
    """Emit lower mesh_order last; preserve legacy and equal-priority order."""

    def priority(entry: Any) -> float:
        value = (
            entry.get("mesh_order")
            if isinstance(entry, Mapping)
            else getattr(entry, "mesh_order", None)
        )
        return float("inf") if value is None else float(value)

    return sorted(entries, key=priority, reverse=True)
