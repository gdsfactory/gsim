"""Shared layer classification and GDS-layer extraction helpers.

Used by both ``extractor`` and ``visualization`` modules.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from gdsfactory.technology import LayerLevel

logger = logging.getLogger(__name__)


def get_gds_layer_tuple(layer_level: LayerLevel) -> tuple[int, int] | None:
    """Extract GDS layer tuple (layer, datatype) from a gdsfactory LayerLevel.

    Returns ``None`` when the layer cannot be parsed (callers decide fallback).
    """
    layer: Any = layer_level.layer

    if isinstance(layer, tuple):
        return (int(layer[0]), int(layer[1]))

    if isinstance(layer, int):
        return (int(layer), 0)

    if hasattr(layer, "layer"):
        inner = layer.layer
        if hasattr(inner, "layer") and hasattr(inner, "datatype"):
            return (int(inner.layer), int(inner.datatype))
        if isinstance(inner, int):
            datatype = getattr(layer, "datatype", 0)
            return (int(inner), int(datatype) if datatype else 0)
        if hasattr(inner, "layer"):
            innermost = inner.layer
            if isinstance(innermost, int):
                datatype = getattr(inner, "datatype", 0)
                return (int(innermost), int(datatype) if datatype else 0)

    if hasattr(layer, "layer") and hasattr(layer, "datatype"):
        return (int(layer.layer), int(layer.datatype))

    if hasattr(layer, "value"):
        if isinstance(layer.value, tuple):
            return (int(layer.value[0]), int(layer.value[1]))
        if isinstance(layer.value, int):
            return (int(layer.value), 0)

    if isinstance(layer, str):
        if "/" in layer:
            parts = layer.split("/")
            return (int(parts[0]), int(parts[1]))
        return None

    try:
        return (int(layer), 0)
    except (TypeError, ValueError):
        logger.warning("Could not parse layer %s", layer)
        return None


def classify_layer_type(
    layer_name: str,
    material: str | None = None,
) -> Literal["conductor", "via", "dielectric", "substrate"]:
    """Classify a layer as conductor, via, dielectric, or substrate.

    When *material* is provided, the materials DB is consulted for additional
    classification hints (conductor vs dielectric).  When ``None``, only
    name-based heuristics are used.
    """
    name_lower = layer_name.lower()

    if "via" in name_lower or "cont" in name_lower:
        return "via"

    if any(
        m in name_lower for m in ["metal", "topmetal", "m1", "m2", "m3", "m4", "m5"]
    ):
        return "conductor"

    if "substrate" in name_lower or name_lower == "sub":
        return "substrate"

    # Material-based lookup (only when caller supplies material)
    if material is not None:
        from gsim.common.stack.materials import get_material_properties

        props = get_material_properties(material)
        if props:
            if props.type == "conductor":
                return "conductor"
            if props.type == "semiconductor" and "substrate" in name_lower:
                return "substrate"
            if props.type == "dielectric":
                return "dielectric"

        material_lower = material.lower()
        if material_lower in [
            "aluminum",
            "copper",
            "tungsten",
            "gold",
            "al",
            "cu",
            "w",
        ]:
            return "conductor"

    if "poly" in name_lower or "active" in name_lower:
        return "conductor"

    return "dielectric"
