"""Adapter from the canonical PDK stack to gsim's Palace ``LayerStack``.

The canonical resolver (:mod:`gsim.common.pdk.stack`) describes the stack in a
backend-neutral way.  Palace's mesher consumes a :class:`gsim.common.stack.LayerStack`
(``layers``/``dielectrics``/``materials``), so this adapter maps one to the
other without re-deriving or synthesizing anything.
"""

from __future__ import annotations

from gsim.common.pdk.stack import ResolvedStack
from gsim.common.stack.extractor import Layer, LayerStack


def build_palace_layer_stack(resolved: ResolvedStack) -> LayerStack:
    """Map a canonical :class:`ResolvedStack` to a Palace ``LayerStack``.

    Every PDK layer is preserved (drawn or not) so downstream via/conductor
    lookups and port z-references keep working.  No synthetic dielectrics are
    added; the returned ``dielectrics`` list contains only regions the canonical
    resolver was explicitly given.
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

    stack.simulation = {
        "pdk_authoritative": True,
        "wavelength_um": resolved.wavelength_um,
        "pdk_overlay": resolved.overlay,
        "drawn_z_range": (
            list(resolved.drawn_z_range) if resolved.drawn_z_range else None
        ),
    }
    return stack


__all__ = ["build_palace_layer_stack"]
