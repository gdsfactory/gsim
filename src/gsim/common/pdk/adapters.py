"""Adapters from the canonical PDK stack to simulation layer stacks.

The canonical resolver (:mod:`gsim.common.pdk.stack`) is backend-neutral.  The
Palace mesher and the Meep config builder both consume
:class:`gsim.common.stack.extractor.LayerStack`, so this module performs that
translation once, preserving every authoritative field and recording enough
provenance for configs to be audited.
"""

from __future__ import annotations

from typing import Any

from gsim.common.pdk.stack import ResolvedStack
from gsim.common.stack.extractor import Layer, LayerStack


def build_layer_stack(
    resolved: ResolvedStack,
    *,
    backend: str,
    extra_simulation: dict[str, Any] | None = None,
) -> LayerStack:
    """Map a canonical :class:`ResolvedStack` to a gsim ``LayerStack``.

    Every PDK layer is preserved (drawn or not) so downstream via/conductor
    lookups and port z-references keep working.  No synthetic dielectrics are
    added; the returned ``dielectrics`` list contains only regions the canonical
    resolver was explicitly given.

    Args:
        resolved: Canonical PDK-authoritative stack.
        backend: Backend name recorded in the stack provenance (``"palace"`` or
            ``"meep"``).
        extra_simulation: Additional backend-specific ``simulation`` metadata.

    Returns:
        A :class:`~gsim.common.stack.extractor.LayerStack` that faithfully
        mirrors *resolved*.
    """
    stack = LayerStack(pdk_name=resolved.pdk_name)

    for key, process_layer in resolved.layers.items():
        stack.layers[key] = Layer(
            name=key,
            gds_layer=process_layer.gds_layer,
            zmin=process_layer.zmin,
            zmax=process_layer.zmax,
            thickness=process_layer.thickness,
            material=process_layer.material,
            layer_type=process_layer.layer_type,  # ty: ignore[invalid-argument-type]
            sidewall_angle=process_layer.sidewall_angle,
            mesh_order=process_layer.mesh_order,
        )

    for name, properties in resolved.materials.items():
        stack.materials[name] = dict(properties)

    stack.dielectrics = [
        {
            "name": dielectric.name,
            "zmin": dielectric.zmin,
            "zmax": dielectric.zmax,
            "material": dielectric.material,
        }
        for dielectric in resolved.dielectrics
    ]

    provenance = resolved.provenance()
    stack.simulation = {
        "pdk_authoritative": True,
        "backend": backend,
        "wavelength_um": resolved.wavelength_um,
        "pdk_overlay": resolved.overlay,
        "drawn_z_range": (
            list(resolved.drawn_z_range) if resolved.drawn_z_range else None
        ),
        "undrawn_z_range": (
            list(resolved.undrawn_z_range) if resolved.undrawn_z_range else None
        ),
        "stack_hash": resolved.stack_hash,
        "stack_notes": list(resolved.notes),
        "pdk_provenance": provenance,
    }
    if extra_simulation:
        stack.simulation.update(extra_simulation)
    return stack


__all__ = ["build_layer_stack"]
