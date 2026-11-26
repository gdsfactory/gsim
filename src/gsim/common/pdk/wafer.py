"""Blanket films declared on the PDK wafer layer.

gdsfactory PDKs such as the generic PDK declare their substrate, buried oxide
and cladding as ``LayerLevel`` entries on the ``WAFER`` layer, which no cell
draws. Their material fills the whole chip, so they are blanket films exactly
like levels marked ``background=True``. This module marks them as such on a
copy of the stack, leaving the PDK's own stack untouched.
"""

from __future__ import annotations

from typing import Any

from gsim.common.stack._layer_utils import (
    _tuple_from_layer_expression,
    get_gds_layer_tuple,
)

#: gdsfactory's wafer layer, used when a PDK's layer map has no ``WAFER`` entry.
GDSFACTORY_WAFER_LAYER = (999, 0)


def wafer_layer_tuple(pdk: Any | None) -> tuple[int, int]:
    """Return the GDS tuple of the PDK's wafer layer.

    The PDK's own ``WAFER`` layer is used when its layer map declares one;
    otherwise the gdsfactory convention ``(999, 0)`` applies.
    """
    wafer = getattr(getattr(pdk, "layers", None), "WAFER", None)
    parsed = _tuple_from_layer_expression(wafer) if wafer is not None else None
    return parsed if parsed is not None else GDSFACTORY_WAFER_LAYER


def is_wafer_level(level: Any, wafer_layer: tuple[int, int]) -> bool:
    """Return whether *level* is a plain, unsized level on the wafer layer."""
    from gdsfactory.technology import LogicalLayer

    layer = getattr(level, "layer", None)
    if not isinstance(layer, LogicalLayer):
        return False  # Boolean expressions are patterned masks.
    if any((*layer.sizings_xoffsets, *layer.sizings_yoffsets)):
        return False
    return get_gds_layer_tuple(level) == wafer_layer


def with_wafer_films(layer_stack: Any, pdk: Any | None) -> Any:
    """Return *layer_stack* with wafer-layer levels marked ``background=True``.

    The input stack is never modified: when a level changes, the result is a
    shallow copy holding copied levels. Stacks of gdsfactory versions without
    the ``background`` field are returned unchanged.
    """
    wafer_layer = wafer_layer_tuple(pdk)
    updated = {
        key: level.model_copy(update={"background": True})
        for key, level in layer_stack.layers.items()
        if "background" in type(level).model_fields
        and not level.background
        and is_wafer_level(level, wafer_layer)
    }
    if not updated:
        return layer_stack
    return layer_stack.model_copy(update={"layers": {**layer_stack.layers, **updated}})


__all__ = [
    "GDSFACTORY_WAFER_LAYER",
    "is_wafer_level",
    "wafer_layer_tuple",
    "with_wafer_films",
]
