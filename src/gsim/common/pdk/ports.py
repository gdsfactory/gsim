"""Resolve guided ports from physical layers and cross-section definitions."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from math import cos, isclose, radians, sin
from typing import Any

import gdsfactory as gf
from gdsfactory.cross_section import CrossSection
from shapely import Point

from gsim.common.pdk.models import ResolvedLayer, ResolvedPort, UnsupportedPortError

_ANGLE_TOLERANCE_DEGREES = 1e-6
_PORT_POSITION_TOLERANCE_UM = 1e-6


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
    port: Any,
    layers: Mapping[str, ResolvedLayer],
    port_layer_map: Mapping[tuple[int, int], str | tuple[str, ...]],
) -> list[ResolvedLayer]:
    """Find physical layers that own a port's position and GDS layer."""
    port_layer = _port_layer(port)
    center = Point(float(port.dcenter[0]), float(port.dcenter[1]))
    if port_layer in port_layer_map:
        selection = port_layer_map[port_layer]
        layer_keys = (selection,) if isinstance(selection, str) else selection
        if (
            not isinstance(layer_keys, tuple)
            or not layer_keys
            or any(not isinstance(key, str) or not key for key in layer_keys)
        ):
            raise UnsupportedPortError(
                f"Port {port.name!r} on layer {port_layer} requires a nonempty "
                "LayerStack key or tuple of nonempty keys."
            )
        if len(set(layer_keys)) != len(layer_keys):
            raise UnsupportedPortError(
                f"Port {port.name!r} on layer {port_layer} maps to duplicate "
                "LayerStack keys."
            )
        mapped = []
        for layer_key in layer_keys:
            mapped_layer = layers.get(layer_key)
            if mapped_layer is None:
                raise UnsupportedPortError(
                    f"Port {port.name!r} on layer {port_layer} maps to LayerStack "
                    f"entry {layer_key!r}, which has no resolved geometry."
                )
            if not mapped_layer.geometry.buffer(_PORT_POSITION_TOLERANCE_UM).covers(
                center
            ):
                raise UnsupportedPortError(
                    f"Port {port.name!r} on layer {port_layer} maps to LayerStack "
                    f"entry {layer_key!r}, but its center is outside resolved geometry."
                )
            mapped.append(mapped_layer)
        if len(_connected_port_layers(mapped)) != len(mapped):
            raise UnsupportedPortError(
                f"Port {port.name!r} on layer {port_layer} must map to abutting, "
                "non-overlapping bands of the same material."
            )
        return mapped
    return _geometry_candidates(port, layers, {port_layer})


def _geometry_candidates(
    port: Any,
    layers: Mapping[str, ResolvedLayer],
    drawn_layers: set[tuple[int, int]],
) -> list[ResolvedLayer]:
    """Find physical bands containing a port on any selected drawing layer."""
    center = Point(float(port.dcenter[0]), float(port.dcenter[1]))
    candidates = []
    for resolved_layer in layers.values():
        source_layers = _layer_tuples(resolved_layer.layer)
        source_layers.update(_layer_tuples(resolved_layer.derived_layer))
        if not drawn_layers.intersection(source_layers):
            continue
        if resolved_layer.geometry.buffer(_PORT_POSITION_TOLERANCE_UM).covers(center):
            candidates.append(resolved_layer)
    return sorted(candidates, key=lambda layer: (layer.mesh_order, layer.key))


def _connected_port_layers(candidates: list[ResolvedLayer]) -> list[ResolvedLayer]:
    """Join abutting bands in candidate order, excluding overlapping owners."""
    if not candidates:
        return []
    connected = [candidates[0]]
    remaining = candidates[1:]
    while remaining:
        lower = min(layer.z_bounds[0] for layer in connected)
        upper = max(layer.z_bounds[1] for layer in connected)
        for index, layer in enumerate(remaining):
            if layer.material != connected[0].material:
                continue
            start, end = layer.z_bounds
            if (
                abs(start - upper) <= _PORT_POSITION_TOLERANCE_UM
                or abs(end - lower) <= _PORT_POSITION_TOLERANCE_UM
            ):
                # Add one owner at a time: two candidates can abut this edge
                # while overlapping each other. Keep the higher-priority one.
                connected.append(remaining.pop(index))
                break
        else:
            break
    return connected


class _CrossSections:
    """Read port cross-section metadata and lazily inspect a PDK registry."""

    def __init__(self, pdk: Any) -> None:
        """Keep registry discovery local to one component resolution."""
        self.pdk = pdk
        self._registered: list[CrossSection] | None = None

    def for_port(self, port: Any) -> list[CrossSection]:
        """Prefer an actual port snapshot or named cross-section over defaults."""
        info = getattr(port, "info", None) or {}
        settings = info.get("cross_section_settings")
        name = info.get("cross_section")
        getter = getattr(self.pdk, "get_cross_section", None)
        if settings is not None or name is not None:
            if settings is not None and not isinstance(settings, Mapping):
                raise UnsupportedPortError(
                    f"Port {port.name!r} cross_section_settings must be a mapping."
                )
            if settings is not None and "sections" in settings:
                load = partial(CrossSection.model_validate, settings)
            elif callable(getter) and name is not None:
                load = partial(getter, name, **(settings or {}))
            else:
                raise UnsupportedPortError(
                    f"Port {port.name!r} requires a registered cross-section name."
                )
            try:
                cross_section = load()
            except Exception as error:
                raise UnsupportedPortError(
                    f"Could not resolve cross-section metadata for port {port.name!r}: "
                    f"{error}. Supply a valid cross-section or port_layer_map."
                ) from error
            if not isinstance(cross_section, CrossSection):
                raise UnsupportedPortError(
                    f"Port {port.name!r} requires a gdsfactory CrossSection."
                )
            return [cross_section]
        if self._registered is None:
            self._registered = []
            if callable(getter):
                for registered_name in getattr(self.pdk, "cross_sections", None) or {}:
                    try:
                        cross_section = getter(registered_name)
                    except Exception:
                        # Unrelated factories may require explicit settings.
                        continue
                    if isinstance(cross_section, CrossSection):
                        self._registered.append(cross_section)
        return self._registered


def _core_drawing_layers(
    cross_section: CrossSection, marker: tuple[int, int]
) -> set[tuple[int, int]]:
    """Select visible sections coincident with the primary marker section."""
    if not cross_section.sections:
        return set()
    primary = cross_section.sections[0]
    try:
        primary_layer = gf.get_layer_tuple(primary.layer)
    except (TypeError, ValueError):
        return set()
    if primary_layer != marker:
        return set()
    result = set()
    for section in cross_section.sections[1:]:
        if section.hidden or not (
            isclose(
                section.offset,
                primary.offset,
                abs_tol=_PORT_POSITION_TOLERANCE_UM,
                rel_tol=0,
            )
            and isclose(
                section.width,
                primary.width,
                abs_tol=_PORT_POSITION_TOLERANCE_UM,
                rel_tol=0,
            )
        ):
            continue
        drawn_layer = gf.get_layer_tuple(section.layer)
        if drawn_layer != marker:
            result.add(drawn_layer)
    return result


def _cross_section_candidates(
    port: Any, layers: Mapping[str, ResolvedLayer], catalog: _CrossSections
) -> list[ResolvedLayer]:
    """Infer a unique connected physical core without choosing a material by name."""
    try:
        previous_pdk = gf.get_active_pdk()
    except ValueError:
        previous_pdk = None
    switched = previous_pdk is not catalog.pdk and hasattr(catalog.pdk, "activate")
    try:
        if switched:
            _activate(catalog.pdk)
        matches = {}
        for cross_section in catalog.for_port(port):
            try:
                drawing_layers = _core_drawing_layers(cross_section, _port_layer(port))
            except (TypeError, ValueError) as error:
                raise UnsupportedPortError(
                    f"Could not resolve drawing layers for port {port.name!r}: "
                    f"{error}. "
                    "Supply port_layer_map to select physical bands."
                ) from error
            candidates = _geometry_candidates(port, layers, drawing_layers)
            if not candidates:
                continue
            if len(_connected_port_layers(candidates)) != len(candidates):
                raise UnsupportedPortError(
                    f"Cross-section for port {port.name!r} matches ambiguous "
                    "physical bands "
                    f"{[layer.key for layer in candidates]}. Supply port_layer_map."
                )
            matches[tuple(sorted(layer.key for layer in candidates))] = candidates
        if len(matches) > 1:
            raise UnsupportedPortError(
                f"Port {port.name!r} matches multiple physical cross-sections. "
                "Set its cross_section metadata or supply port_layer_map."
            )
        return next(iter(matches.values()), [])
    finally:
        if switched and previous_pdk is not None:
            _activate(previous_pdk)


def _activate(pdk: Any) -> None:
    """Select the actual PDK object even when another instance shares its name."""
    if isinstance(pdk, gf.Pdk):
        pdk.activate(force=True)
    else:
        pdk.activate()


def _resolved_ports(
    component: Any,
    layers: Mapping[str, ResolvedLayer],
    port_layer_map: Mapping[tuple[int, int], str | tuple[str, ...]],
    z_bounds: tuple[float, float] | None = None,
    *,
    pdk: Any | None = None,
) -> dict[str, ResolvedPort]:
    """Map every component port to one resolved physical layer."""
    resolved: dict[str, ResolvedPort] = {}
    cross_sections = _CrossSections(pdk)
    for port in component.ports:
        orientation, normal = _port_orientation_and_normal(port)
        port_type = str(getattr(port, "port_type", ""))
        candidates = _port_candidates(port, layers, port_layer_map)
        is_vertical = port_type.startswith("vertical_")
        if not candidates and not is_vertical:
            candidates = _cross_section_candidates(port, layers, cross_sections)
        if not candidates and not is_vertical:
            raise UnsupportedPortError(
                f"Port {port.name!r} on layer {_port_layer(port)} does not map to "
                "resolved LayerStack geometry."
            )
        owners = _connected_port_layers(candidates)
        layer = owners[0] if owners else None
        z_lower = min((owner.z_bounds[0] for owner in owners), default=0.0)
        z_upper = max((owner.z_bounds[1] for owner in owners), default=0.0)
        if owners and z_bounds is not None:
            z_lower = max(z_lower, z_bounds[0])
            z_upper = min(z_upper, z_bounds[1])
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
            layer_keys=tuple(owner.key for owner in owners) if len(owners) > 1 else (),
            z_bounds=(z_lower, z_upper) if owners else None,
        )
    return resolved
