"""Shared vertical reference for mode cells, sampling grids and domain reports."""

from typing import Any


def material_extent(stack: Any) -> tuple[float, float] | None:
    """Return the non-air layer and dielectric extent, excluding air padding."""
    bounds = [
        (layer.zmin, layer.zmax)
        for layer in stack.layers.values()
        if layer.material not in {"Air", "air"}
    ]
    bounds.extend(
        (dielectric["zmin"], dielectric["zmax"])
        for dielectric in getattr(stack, "dielectrics", ())
        if dielectric.get("material") not in {"Air", "air"}
    )
    if not bounds:
        return None
    return min(lower for lower, _ in bounds), max(upper for _, upper in bounds)


def cell_extent(stack: Any) -> tuple[float, float]:
    """Use the reported material extent, with a fallback for air-only cells."""
    extent = material_extent(stack)
    if extent is not None:
        return extent
    bounds = [(layer.zmin, layer.zmax) for layer in stack.layers.values()]
    bounds.extend(
        (dielectric["zmin"], dielectric["zmax"])
        for dielectric in getattr(stack, "dielectrics", ())
    )
    if not bounds:
        raise ValueError("Cannot determine mode cell bounds from an empty stack.")
    return min(lower for lower, _ in bounds), max(upper for _, upper in bounds)
