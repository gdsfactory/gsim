"""Resolve a passive PCell through authoritative PDK APIs."""

from __future__ import annotations

from collections.abc import Mapping
from math import cos, radians, sin
from typing import Any

import gdsfactory as gf
from pdk_schema import MaterialCard
from shapely import Point

from gsim.common.materials import (
    MaterialResolutionError,
    get_project_material_cards,
    resolve_material_snapshot,
)
from gsim.common.pdk.models import (
    ComponentResolutionError,
    LayerResolutionError,
    ResolvedLayer,
    ResolvedPassivePcell,
    ResolvedPort,
    UnsupportedPortError,
)
from gsim.common.polygon import fuse_polygons

_ANGLE_TOLERANCE_DEGREES = 1e-6
_PORT_POSITION_TOLERANCE_UM = 1e-6


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
    attached_cards = getattr(pdk, "material_cards", None)
    if attached_cards is not None:
        return attached_cards
    if pdk_or_module is not None:
        return get_project_material_cards(pdk_or_module)
    return get_project_material_cards(pdk)


def _resolved_layers(
    derived_component: Any, layer_stack: Any
) -> dict[str, ResolvedLayer]:
    """Extract nonempty physical layers while preserving fabrication fields."""
    resolved: dict[str, ResolvedLayer] = {}
    for layer_key, layer_level in layer_stack.layers.items():
        geometry = fuse_polygons(
            derived_component,
            layer_level,
            round_tol=6,
            simplify_tol=0.0,
        )
        if geometry.is_empty:
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


def _layer_tuples(layer: Any) -> set[tuple[int, int]]:
    """Collect concrete GDS layer tuples from a logical layer expression."""
    if layer is None:
        return set()
    if isinstance(layer, tuple) and len(layer) == 2:
        return {(int(layer[0]), int(layer[1]))}
    layer_number = getattr(layer, "layer", None)
    datatype = getattr(layer, "datatype", None)
    if isinstance(layer_number, int) and isinstance(datatype, int):
        return {(layer_number, datatype)}
    nested_layer = getattr(layer, "layer", None)
    if nested_layer is not None:
        return _layer_tuples(nested_layer)
    layer_tuples: set[tuple[int, int]] = set()
    for attribute in ("layer1", "layer2"):
        nested = getattr(layer, attribute, None)
        if nested is not None:
            layer_tuples.update(_layer_tuples(nested))
    if layer_tuples:
        return layer_tuples
    try:
        values = tuple(layer)
    except (TypeError, ValueError):
        return set()
    if len(values) != 2:
        return set()
    return {(int(values[0]), int(values[1]))}


def _axis_aligned_orientation(
    port_name: str, orientation: Any
) -> tuple[float, tuple[int, int, int]]:
    """Normalize an axis-aligned angle and return its outward normal."""
    if orientation is None:
        raise UnsupportedPortError(f"Port {port_name!r} has no orientation.")
    normalized = float(orientation) % 360.0
    aligned = (round(normalized / 90.0) * 90.0) % 360.0
    angular_error = abs((normalized - aligned + 180.0) % 360.0 - 180.0)
    if angular_error > _ANGLE_TOLERANCE_DEGREES:
        raise UnsupportedPortError(
            f"Port {port_name!r} has orientation {orientation}; FDTD ports "
            "must be axis-aligned."
        )
    angle = radians(aligned)
    return aligned, (round(cos(angle)), round(sin(angle)), 0)


def _port_orientation_and_normal(
    port: Any,
) -> tuple[float, tuple[int, int, int]]:
    """Resolve guided-port normals while preserving vertical-port semantics."""
    port_type = str(getattr(port, "port_type", ""))
    if port_type.startswith("vertical_"):
        orientation = 0.0 if port.orientation is None else float(port.orientation)
        return orientation % 360.0, (0, 0, 1)
    return _axis_aligned_orientation(port.name, port.orientation)


def _port_layer(port: Any) -> tuple[int, int]:
    """Return a concrete GDS tuple for a component port."""
    if isinstance(port.layer, int):
        layer_info = port.kcl.get_info(port.layer)
        return int(layer_info.layer), int(layer_info.datatype)
    layers = _layer_tuples(port.layer)
    if len(layers) != 1:
        raise UnsupportedPortError(
            f"Port {port.name!r} does not have one concrete GDS layer."
        )
    return next(iter(layers))


def _port_candidates(
    port: Any, layers: Mapping[str, ResolvedLayer]
) -> list[ResolvedLayer]:
    """Find physical layers that own a port's position and GDS layer."""
    port_layer = _port_layer(port)
    center = Point(float(port.dcenter[0]), float(port.dcenter[1]))
    candidates = []
    for resolved_layer in layers.values():
        source_layers = _layer_tuples(resolved_layer.layer)
        source_layers.update(_layer_tuples(resolved_layer.derived_layer))
        if port_layer not in source_layers:
            continue
        if resolved_layer.geometry.buffer(_PORT_POSITION_TOLERANCE_UM).covers(center):
            candidates.append(resolved_layer)
    return sorted(candidates, key=lambda layer: (layer.mesh_order, layer.key))


def _resolved_ports(
    component: Any,
    layers: Mapping[str, ResolvedLayer],
) -> dict[str, ResolvedPort]:
    """Map every component port to one resolved physical layer."""
    resolved: dict[str, ResolvedPort] = {}
    for port in component.ports:
        orientation, normal = _port_orientation_and_normal(port)
        port_type = str(getattr(port, "port_type", ""))
        candidates = _port_candidates(port, layers)
        is_vertical = port_type.startswith("vertical_")
        if not candidates and not is_vertical:
            raise UnsupportedPortError(
                f"Port {port.name!r} on layer {_port_layer(port)} does not map to "
                "resolved LayerStack geometry."
            )
        layer = candidates[0] if candidates else None
        z_lower, z_upper = layer.z_bounds if layer is not None else (0.0, 0.0)
        resolved[port.name] = ResolvedPort(
            name=port.name,
            center=(
                float(port.dcenter[0]),
                float(port.dcenter[1]),
                (z_lower + z_upper) / 2,
            ),
            width=float(port.width),
            orientation=orientation,
            normal=normal,
            port_type=port_type,
            layer_key=layer.key if layer is not None else None,
            material=layer.material if layer is not None else None,
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
) -> ResolvedPassivePcell:
    """Resolve a passive component, fabrication stack, materials, and ports.

    The PDK LayerStack is authoritative. Only stack entries with geometry in the
    resolved component are evaluated, so unrelated process materials are not
    required. MaterialCards attached to the project PDK take precedence over
    gsim's built-in fallbacks.
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
    layers = _resolved_layers(resolved_component, layer_stack)
    project_cards = _resolve_project_cards(pdk_object, pdk_or_module)
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
    ports = _resolved_ports(resolved_component, layers)
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
    )


__all__ = ["resolve_passive_pcell"]
