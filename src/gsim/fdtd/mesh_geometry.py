"""Robust OCC geometry construction for coarse GDSFactory FDTD meshes."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import ceil, radians, tan
from typing import Any

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import explain_validity

from gsim.common.pdk import ResolvedLayer, ResolvedPort
from gsim.fdtd.mesh_prism import extrude_polygon
from gsim.fdtd.models import FDTDGeometryError

UM_TO_NM = 1000.0
GEOMETRY_TOLERANCE_NM = 1e-3


def _polygon_sort_key(polygon: Polygon) -> tuple[float, ...]:
    """Return a stable ordering key for disconnected polygon members."""
    centroid = polygon.centroid
    return (
        round(centroid.x, 12),
        round(centroid.y, 12),
        round(polygon.area, 12),
        *(round(value, 12) for value in polygon.bounds),
    )


def iter_polygons(geometry: BaseGeometry, *, layer_key: str) -> list[Polygon]:
    """Return every valid polygonal member while retaining interior rings."""
    if isinstance(geometry, Polygon):
        polygons = [geometry]
    elif isinstance(geometry, MultiPolygon):
        polygons = list(geometry.geoms)
    elif isinstance(geometry, GeometryCollection):
        polygons = [part for part in geometry.geoms if isinstance(part, Polygon)]
    else:
        polygons = []
    if not polygons:
        raise FDTDGeometryError(
            f"Layer {layer_key!r} has unsupported or empty "
            f"{type(geometry).__name__} geometry."
        )
    for polygon in polygons:
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
            raise FDTDGeometryError(
                f"Layer {layer_key!r} has invalid polygon geometry: "
                f"{explain_validity(polygon)}."
            )
    return sorted(polygons, key=_polygon_sort_key)


def _sidewall_offset_um(layer: ResolvedLayer, normalized_z: float) -> float:
    """Evaluate the PDK's linear lateral offset at normalized z."""
    return (
        (layer.width_to_z - normalized_z)
        * abs(layer.thickness)
        * tan(radians(layer.sidewall_angle))
    )


def sidewall_slice_count(
    layer: ResolvedLayer,
    geometry_tolerance_nm: float,
) -> int:
    """Choose midpoint slices that bound lateral geometry error."""
    if geometry_tolerance_nm <= 0:
        raise ValueError("geometry_tolerance_nm must be positive.")
    total_displacement_nm = (
        abs(layer.thickness) * abs(tan(radians(layer.sidewall_angle))) * UM_TO_NM
    )
    if total_displacement_nm == 0:
        return 1
    return max(1, ceil(total_displacement_nm / (2 * geometry_tolerance_nm)))


def _port_groups(
    ports: Iterable[ResolvedPort],
) -> dict[tuple[int, int, float], list[ResolvedPort]]:
    """Group ports that share one axis-aligned end plane."""
    groups: dict[tuple[int, int, float], list[ResolvedPort]] = defaultdict(list)
    for port in ports:
        normal_axis = next(index for index, value in enumerate(port.normal) if value)
        if normal_axis not in {0, 1}:
            raise FDTDGeometryError(
                f"Guided port {port.name!r} is not in the component plane."
            )
        groups[
            (normal_axis, port.normal[normal_axis], port.center[normal_axis])
        ].append(port)
    return groups


def _clip_to_port_plane(
    geometry: BaseGeometry,
    *,
    normal_axis: int,
    normal_sign: int,
    target: float,
    maximum_port_width: float,
) -> BaseGeometry:
    """Clip geometry at one outward-facing port plane."""
    xmin, ymin, xmax, ymax = geometry.bounds
    margin = max(xmax - xmin, ymax - ymin, maximum_port_width, 1.0)
    if normal_axis == 0 and normal_sign < 0:
        clip = box(target, ymin - margin, xmax + margin, ymax + margin)
    elif normal_axis == 0:
        clip = box(xmin - margin, ymin - margin, target, ymax + margin)
    elif normal_sign < 0:
        clip = box(xmin - margin, target, xmax + margin, ymax + margin)
    else:
        clip = box(xmin - margin, ymin - margin, xmax + margin, target)
    return geometry.intersection(clip)


def _condition_profile_at_ports(
    geometry: BaseGeometry,
    ports: list[ResolvedPort],
    offset_um: float,
) -> BaseGeometry:
    """Create all same-plane port stubs together, then clip once per plane."""
    conditioned = geometry
    extension_epsilon_um = GEOMETRY_TOLERANCE_NM / UM_TO_NM
    for (normal_axis, normal_sign, target), grouped_ports in _port_groups(
        ports
    ).items():
        transverse_axis = 1 - normal_axis
        extensions = []
        for port in grouped_ports:
            half_width = port.width / 2 + offset_um
            if half_width <= 0:
                raise FDTDGeometryError(
                    f"Layer {port.layer_key!r} sidewall closes port {port.name!r}."
                )
            transverse_lower = port.center[transverse_axis] - half_width
            transverse_upper = port.center[transverse_axis] + half_width
            inward = target - normal_sign * (abs(offset_um) + extension_epsilon_um)
            if normal_axis == 0:
                extension = box(
                    min(target, inward),
                    transverse_lower,
                    max(target, inward),
                    transverse_upper,
                )
            else:
                extension = box(
                    transverse_lower,
                    min(target, inward),
                    transverse_upper,
                    max(target, inward),
                )
            extensions.append(extension)
        conditioned = unary_union([conditioned, *extensions])
        conditioned = _clip_to_port_plane(
            conditioned,
            normal_axis=normal_axis,
            normal_sign=normal_sign,
            target=target,
            maximum_port_width=max(port.width for port in grouped_ports),
        )
    return conditioned


def _validate_layer(layer: ResolvedLayer) -> None:
    """Reject fabrication profiles the initial mesh writer cannot evaluate."""
    if layer.bias not in (None, 0, 0.0) or layer.z_to_bias is not None:
        raise FDTDGeometryError(
            f"Layer {layer.key!r} uses bias or z_to_bias, which is not supported "
            "by the initial GDSFactory FDTD mesh writer."
        )
    if not 0 <= layer.width_to_z <= 1:
        raise FDTDGeometryError(
            f"Layer {layer.key!r} width_to_z must be between 0 and 1."
        )
    if abs(layer.sidewall_angle) >= 80:
        raise FDTDGeometryError(
            f"Layer {layer.key!r} sidewall angle is too steep to mesh safely."
        )


def _scaled_ring(
    coordinates: Iterable[tuple[float, ...]],
) -> list[tuple[float, float]]:
    """Convert one closed Shapely ring from micrometers to nanometers."""
    points = [
        (float(point[0]) * UM_TO_NM, float(point[1]) * UM_TO_NM)
        for point in coordinates
    ]
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    return points


def _add_polygon_prism(
    kernel: Any,
    polygon: Polygon,
    *,
    z_lower_um: float,
    z_upper_um: float,
    layer_key: str,
) -> int:
    """Extrude one polygon while retaining every interior ring as a hole."""
    exterior = _scaled_ring(polygon.exterior.coords)
    holes = []
    for interior in polygon.interiors:
        points = _scaled_ring(interior.coords)
        holes.append(([point[0] for point in points], [point[1] for point in points]))
    volume_tag = extrude_polygon(
        kernel,
        [point[0] for point in exterior],
        [point[1] for point in exterior],
        z_lower_um * UM_TO_NM,
        (z_upper_um - z_lower_um) * UM_TO_NM,
        holes=holes,
    )
    if volume_tag is None:
        raise FDTDGeometryError(f"Could not extrude layer {layer_key!r}.")
    return volume_tag


def _add_stepped_layer_volumes(
    kernel: Any,
    layer: ResolvedLayer,
    ports: list[ResolvedPort],
    *,
    geometry_tolerance_nm: float,
) -> list[int]:
    """Create midpoint-slice prisms when exact loft correspondence is unsafe."""
    slice_count = sidewall_slice_count(layer, geometry_tolerance_nm)
    z_lower_um, z_upper_um = layer.z_bounds
    volume_tags = []
    for slice_index in range(slice_count):
        lower_fraction = slice_index / slice_count
        upper_fraction = (slice_index + 1) / slice_count
        midpoint_fraction = (lower_fraction + upper_fraction) / 2
        offset_um = _sidewall_offset_um(layer, midpoint_fraction)
        profile = layer.geometry.buffer(offset_um, join_style=2)
        profile = _condition_profile_at_ports(profile, ports, offset_um)
        slice_z_lower = z_lower_um + lower_fraction * (z_upper_um - z_lower_um)
        slice_z_upper = z_lower_um + upper_fraction * (z_upper_um - z_lower_um)
        volume_tags.extend(
            _add_polygon_prism(
                kernel,
                polygon,
                z_lower_um=slice_z_lower,
                z_upper_um=slice_z_upper,
                layer_key=layer.key,
            )
            for polygon in iter_polygons(profile, layer_key=layer.key)
        )
    if not volume_tags:
        raise FDTDGeometryError(f"Layer {layer.key!r} produced no volumes.")
    return volume_tags


__all__ = [
    "GEOMETRY_TOLERANCE_NM",
    "UM_TO_NM",
    "iter_polygons",
    "sidewall_slice_count",
]
