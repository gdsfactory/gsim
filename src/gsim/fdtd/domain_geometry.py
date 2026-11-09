"""Resolve blanket films over the physical FDTD domain."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from gsim.common.pdk import ResolvedPassivePcell, resolve_passive_pcell
from gsim.fdtd.domain import background_bounds_nm
from gsim.fdtd.mesh_geometry import UM_TO_NM

if TYPE_CHECKING:
    from gsim.fdtd.api import Domain, Geometry


def resolve_domain_geometry(
    geometry: Geometry,
    *,
    pdk: Any,
    wavelength_um: float,
    domain: Domain,
    background_material: str,
) -> ResolvedPassivePcell:
    """Size the domain once, then fill its XY extent with native blanket films."""
    options = {
        "pdk": pdk,
        "settings": geometry.settings,
        "wavelength_um": wavelength_um,
        "port_layer_map": geometry.port_layer_map,
        "z_bounds": domain.z_bounds if domain.clip_geometry else None,
        "x_bounds": domain.x_bounds if domain.clip_geometry else None,
        "y_bounds": domain.y_bounds if domain.clip_geometry else None,
    }
    resolved = resolve_passive_pcell(geometry.component, **options)
    if not any(
        getattr(level, "background", False)
        for level in resolved.layer_stack.layers.values()
    ):
        return resolved
    bounds = background_bounds_nm(
        resolved,
        background_material,
        domain.padding,
        padding_x=domain.padding_x,
        padding_y=domain.padding_y,
        padding_z=domain.padding_z,
        x_bounds=domain.x_bounds,
        y_bounds=domain.y_bounds,
        z_bounds=domain.z_bounds,
        clip_geometry=domain.clip_geometry,
    )
    options.update(
        settings=None,
        x_bounds=(bounds[0] / UM_TO_NM, bounds[3] / UM_TO_NM),
        y_bounds=(bounds[1] / UM_TO_NM, bounds[4] / UM_TO_NM),
    )
    expanded = resolve_passive_pcell(resolved.component, **options)
    # Newly exposed blanket films can add material height, but XY padding must
    # remain anchored to the original footprint.
    layout_bounds = (
        (*resolved.bounds[0][:2], expanded.bounds[0][2]),
        (*resolved.bounds[1][:2], expanded.bounds[1][2]),
    )
    return replace(expanded, layout_bounds=layout_bounds)
