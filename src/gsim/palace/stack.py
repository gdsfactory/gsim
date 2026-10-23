"""Adapter from the canonical PDK stack to gsim's Palace ``LayerStack``.

The canonical resolver (:mod:`gsim.common.pdk.stack`) describes the stack in a
backend-neutral way.  Palace's mesher consumes a :class:`gsim.common.stack.LayerStack`
(``layers``/``dielectrics``/``materials``), so this adapter maps one to the
other without re-deriving or synthesizing anything.
"""

from __future__ import annotations

from gsim.common.pdk.adapters import build_layer_stack
from gsim.common.pdk.stack import ResolvedStack
from gsim.common.stack.extractor import LayerStack


def build_palace_layer_stack(resolved: ResolvedStack) -> LayerStack:
    """Map a canonical :class:`ResolvedStack` to a Palace ``LayerStack``.

    Every PDK layer is preserved (drawn or not) so downstream via/conductor
    lookups and port z-references keep working.  No synthetic dielectrics are
    added; the returned ``dielectrics`` list contains only regions the canonical
    resolver was explicitly given.
    """
    return build_layer_stack(resolved, backend="palace")


__all__ = ["build_palace_layer_stack"]
