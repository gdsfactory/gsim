"""Shared vertical reference for mode cells, sampling grids and domain reports."""

from typing import Any


def is_ideal_air(name: str, materials: Any = None) -> bool:
    """Identify unit-index Air, retaining custom responses with the same name."""
    if name not in {"Air", "air"}:
        return False
    material = (materials or {}).get(name)
    if material is None:
        return True
    data = material if isinstance(material, dict) else getattr(material, "__dict__", {})
    return (
        data.get("epsilon_diag") in (None, [1, 1, 1])
        and data.get("mu_diag") in (None, [1, 1, 1])
        and not any(data.get("epsilon_offdiag") or [])
        and not data.get("D_conductivity")
        and not any(data.get("D_conductivity_diag") or [])
        and not data.get("epsilon_susceptibilities")
    )


def material_extent(
    stack: Any, component: Any | None = None, materials: Any = None
) -> tuple[float, float] | None:
    """Include modeled media, omitting undrawn Air masks and synthetic padding.

    Without a component, layers describe a uniform slab and are all modeled.
    """
    from gsim.meep.mode_solver import _layer_has_any_polygon

    if stack is None:
        return None
    bounds = [
        (layer.zmin, layer.zmax)
        for layer in stack.layers.values()
        if not is_ideal_air(layer.material, materials)
        or component is None
        or _layer_has_any_polygon(component, layer)
    ]
    bounds.extend(
        (dielectric["zmin"], dielectric["zmax"])
        for dielectric in getattr(stack, "dielectrics", ())
        if not (
            is_ideal_air(dielectric.get("material"), materials)
            and dielectric.get("name") in {"air_box", "air_box_bottom"}
        )
    )
    if not bounds:
        return None
    return min(lower for lower, _ in bounds), max(upper for _, upper in bounds)


def cell_extent(
    stack: Any, component: Any | None = None, materials: Any = None
) -> tuple[float, float]:
    """Use the modeled extent, with a fallback for padding-only cells."""
    extent = material_extent(stack, component, materials)
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
