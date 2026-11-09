"""Resolve a passive PCell through authoritative PDK APIs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from math import isfinite, radians, tan
from typing import Any

import gdsfactory as gf
from pdk_schema import MaterialCard
from shapely import box

from gsim.common.materials import (
    MaterialResolutionError,
    canonical_material_name,
    get_project_material_cards,
    resolve_material_snapshot,
)
from gsim.common.pdk.clipping import material_clipping_axes
from gsim.common.pdk.models import (
    ComponentResolutionError,
    LayerResolutionError,
    ResolvedLayer,
    ResolvedPassivePcell,
)
from gsim.common.pdk.models import ResolvedPort as ResolvedPort
from gsim.common.pdk.models import UnsupportedPortError as UnsupportedPortError
from gsim.common.pdk.ports import _resolved_ports
from gsim.common.polygon import fuse_polygons


def _layer_mesh_order(layer_level: Any) -> int:
    """Return the authoritative mesh order for a LayerLevel.

    PDKs may keep the real order in ``info['mesh_order']`` (e.g. IHP), with the
    ``LayerLevel.mesh_order`` field left at its class default.
    """
    info = getattr(layer_level, "info", None) or {}
    if "mesh_order" in info:
        return int(info["mesh_order"])
    return int(layer_level.mesh_order)


def _resolve_pdk(pdk_or_module: Any | None) -> tuple[Any, Any | None]:
    """Return the explicit PDK object or the active PDK."""
    if pdk_or_module is None:
        return gf.get_active_pdk(), None
    return getattr(pdk_or_module, "PDK", pdk_or_module), pdk_or_module


def _resolve_component(
    pdk: Any,
    component_spec: Any,
    settings: Mapping[str, Any] | None,
) -> Any:
    """Instantiate a component through the selected PDK."""
    if isinstance(component_spec, gf.Component):
        if settings:
            raise ComponentResolutionError(
                "Settings cannot be supplied with an already-instantiated Component."
            )
        return component_spec
    try:
        previous_pdk = gf.get_active_pdk()
    except ValueError:
        previous_pdk = None
    switched_pdk = previous_pdk is not pdk and hasattr(pdk, "activate")
    try:
        if switched_pdk:
            pdk.activate()
        return pdk.get_component(component_spec, settings=settings)
    except Exception as error:
        raise ComponentResolutionError(
            f"Could not instantiate component {component_spec!r} through PDK "
            f"{getattr(pdk, 'name', type(pdk).__name__)!r}: {error}"
        ) from error
    finally:
        if switched_pdk and previous_pdk is not None:
            previous_pdk.activate()


def _resolve_layer_stack(pdk: Any) -> Any:
    """Return the authoritative LayerStack attached to a PDK."""
    get_layer_stack = getattr(pdk, "get_layer_stack", None)
    layer_stack = (
        get_layer_stack()
        if callable(get_layer_stack)
        else getattr(pdk, "layer_stack", None)
    )
    if layer_stack is None:
        raise LayerResolutionError(
            f"PDK {getattr(pdk, 'name', type(pdk).__name__)!r} has no LayerStack."
        )
    return layer_stack


def _resolve_project_cards(
    pdk: Any,
    pdk_or_module: Any | None,
) -> Mapping[str, MaterialCard]:
    """Return cards from a PDK object, module, or the active project."""
    return get_project_material_cards(
        pdk_or_module if pdk_or_module is not None else pdk
    )


def _inside_xy_window(geometry: Any, level: Any, x_bounds: Any, y_bounds: Any) -> bool:
    """Conservatively retain any footprint whose sidewalls can reach the window."""
    if x_bounds is None and y_bounds is None:
        return True
    if level.bias not in (None, 0, 0.0) or level.z_to_bias is not None:
        return True  # Leave unsupported profiles to the backend's diagnostics.
    slope = tan(radians(float(level.sidewall_angle))) * abs(float(level.thickness))
    offset = max(0, level.width_to_z * slope, (level.width_to_z - 1) * slope)
    footprint = geometry.buffer(offset, join_style=2) if offset else geometry
    xmin, ymin, xmax, ymax = footprint.bounds
    left, right = x_bounds if x_bounds is not None else (xmin, xmax)
    bottom, top = y_bounds if y_bounds is not None else (ymin, ymax)
    return footprint.intersection(box(left, bottom, right, top)).area > 0


def _resolved_layers(
    derived_component: Any,
    layer_stack: Any,
    z_bounds: tuple[float, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
    clipped_layers: dict[str, tuple[str, ...]] | None = None,
) -> dict[str, ResolvedLayer]:
    """Extract nonempty physical layers while preserving fabrication fields."""
    for axis, bounds in (("x", x_bounds), ("y", y_bounds), ("z", z_bounds)):
        if bounds is not None and (
            not all(isfinite(value) for value in bounds) or bounds[0] >= bounds[1]
        ):
            raise LayerResolutionError(
                f"{axis}_bounds must be finite and strictly increasing."
            )
    resolved: dict[str, ResolvedLayer] = {}
    for layer_key, layer_level in layer_stack.layers.items():
        z_start = float(layer_level.zmin)
        z_end = z_start + float(layer_level.thickness)
        geometry = fuse_polygons(
            derived_component,
            layer_level,
            round_tol=6,
            simplify_tol=0.0,
            background_bounds=(
                x_bounds[0] if x_bounds else derived_component.dxmin,
                y_bounds[0] if y_bounds else derived_component.dymin,
                x_bounds[1] if x_bounds else derived_component.dxmax,
                y_bounds[1] if y_bounds else derived_component.dymax,
            )
            if x_bounds is not None or y_bounds is not None
            else None,
            background_overhang_axes=(x_bounds is not None, y_bounds is not None),
        )
        if clipped_layers is not None:
            axes = material_clipping_axes(
                geometry, layer_level, x_bounds, y_bounds, z_bounds
            )
            if axes:
                clipped_layers[layer_key] = axes
        if z_bounds is not None and (
            max(z_start, z_end) <= z_bounds[0] or min(z_start, z_end) >= z_bounds[1]
        ):
            continue
        if geometry.is_empty or not _inside_xy_window(
            geometry, layer_level, x_bounds, y_bounds
        ):
            continue
        material = layer_level.material
        if not material:
            raise LayerResolutionError(
                f"LayerStack entry {layer_key!r} has geometry but no material."
            )
        thickness = float(layer_level.thickness)
        if thickness == 0:
            raise LayerResolutionError(
                f"LayerStack entry {layer_key!r} has zero thickness."
            )
        zmin = float(layer_level.zmin)
        resolved[layer_key] = ResolvedLayer(
            key=layer_key,
            declared_name=layer_level.name,
            layer=layer_level.layer,
            derived_layer=layer_level.derived_layer,
            geometry=geometry,
            material=material,
            zmin=zmin,
            thickness=thickness,
            zmax=zmin + thickness,
            sidewall_angle=float(layer_level.sidewall_angle),
            width_to_z=float(layer_level.width_to_z),
            bias=layer_level.bias,
            z_to_bias=layer_level.z_to_bias,
            mesh_order=_layer_mesh_order(layer_level),
        )
    if not resolved:
        raise LayerResolutionError(
            "The component has no geometry on any material-bearing LayerStack entry."
        )
    return resolved


def _bounds(
    layers: Mapping[str, ResolvedLayer],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return combined three-dimensional bounds for resolved layers."""
    xy_bounds = [layer.geometry.bounds for layer in layers.values()]
    z_bounds = [layer.z_bounds for layer in layers.values()]
    return (
        (
            min(bounds[0] for bounds in xy_bounds),
            min(bounds[1] for bounds in xy_bounds),
            min(bounds[0] for bounds in z_bounds),
        ),
        (
            max(bounds[2] for bounds in xy_bounds),
            max(bounds[3] for bounds in xy_bounds),
            max(bounds[1] for bounds in z_bounds),
        ),
    )


def resolve_passive_pcell(
    component: Any,
    *,
    pdk: Any | None = None,
    settings: Mapping[str, Any] | None = None,
    wavelength_um: float = 1.55,
    port_layer_map: Mapping[tuple[int, int], str | tuple[str, ...]] | None = None,
    z_bounds: tuple[float, float] | None = None,
    x_bounds: tuple[float, float] | None = None,
    y_bounds: tuple[float, float] | None = None,
) -> ResolvedPassivePcell:
    """Resolve a passive component, fabrication stack, materials, and ports.

    The PDK LayerStack is authoritative. Only stack entries with geometry in the
    resolved component are evaluated, so unrelated process materials are not
    required. MaterialCards attached to the project PDK take precedence over
    gsim's built-in fallbacks. ``port_layer_map`` associates a virtual port
    layer with one physical LayerStack key or a tuple of abutting same-material
    bands. Each selected entry must contain the port location; the first key
    remains its primary material/priority reference.

    Guided ports that do not match a physical layer can use their cross-section
    metadata or a matching registered PDK cross-section. Coincident drawing
    sections identify physical bands; ambiguous bands require an explicit map.
    Existing direct matches and explicit selections keep their precedence.

    Optional axis bounds omit external films and footprints before material
    lookup. Blanket films fill the requested XY bounds, retaining their exclusion
    masks. Intersecting layers retain their original extrusion profiles; the
    backend clips their solids. Contiguous bands of the same material at an
    unmapped port contribute to its complete cross-section.
    """
    pdk_object, pdk_or_module = _resolve_pdk(pdk)
    resolved_component = _resolve_component(pdk_object, component, settings)
    layer_stack = _resolve_layer_stack(pdk_object)
    try:
        derived_component = layer_stack.get_component_with_derived_layers(
            resolved_component
        )
    except Exception as error:
        raise LayerResolutionError(
            f"Could not evaluate derived layers: {error}"
        ) from error
    clipped_layers: dict[str, tuple[str, ...]] = {}
    layers = _resolved_layers(
        resolved_component, layer_stack, z_bounds, x_bounds, y_bounds, clipped_layers
    )
    project_cards = _resolve_project_cards(pdk_object, pdk_or_module)
    layers = {
        key: replace(
            layer,
            material=canonical_material_name(
                layer.material,
                pdk_or_module if pdk_or_module is not None else pdk_object,
            ),
        )
        for key, layer in layers.items()
    }
    materials = {}
    for material_name in dict.fromkeys(layer.material for layer in layers.values()):
        try:
            materials[material_name] = resolve_material_snapshot(
                material_name,
                wavelength_um,
                project_cards,
            )
        except MaterialResolutionError as error:
            layer_names = [
                layer.key
                for layer in layers.values()
                if layer.material == material_name
            ]
            raise LayerResolutionError(
                f"Could not resolve material {material_name!r} used by layers "
                f"{layer_names}: {error}"
            ) from error
    ports = _resolved_ports(
        resolved_component, layers, port_layer_map or {}, z_bounds, pdk=pdk_object
    )
    return ResolvedPassivePcell(
        component=resolved_component,
        derived_component=derived_component,
        pdk=pdk_object,
        layer_stack=layer_stack,
        layers=layers,
        materials=materials,
        ports=ports,
        bounds=_bounds(layers),
        wavelength_um=float(wavelength_um),
        pdk_module=pdk_or_module if pdk_or_module is not pdk_object else None,
        clipped_layers=clipped_layers,
    )


__all__ = ["resolve_passive_pcell"]
