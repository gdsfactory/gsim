"""Continuous ruled sidewalls for GDSFactory FDTD transfer meshes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from math import hypot
from typing import Any

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from gsim.common.pdk import ResolvedLayer, ResolvedPort
from gsim.fdtd.mesh_geometry import (
    GEOMETRY_TOLERANCE_NM,
    UM_TO_NM,
    _add_stepped_layer_volumes,
    _sidewall_offset_um,
    _validate_layer,
    iter_polygons,
)
from gsim.fdtd.models import FDTDGeometryError

_EDGE_DIRECTION_TOLERANCE = 1e-7


class LoftIncompatibleError(ValueError):
    """Raised when exact bottom and top contours cannot be paired safely."""


def _ring_coordinates(polygon: Polygon) -> list[tuple[float, float]]:
    """Return an open counterclockwise exterior ring in micrometers."""
    oriented_polygon = orient(polygon, sign=1.0)
    return [
        (float(point[0]), float(point[1]))
        for point in list(oriented_polygon.exterior.coords)[:-1]
    ]


def _validate_edge_correspondence(
    base_coordinates: list[tuple[float, float]],
    offset_coordinates: list[tuple[float, float]],
) -> None:
    """Reject ring pairings whose corresponding edges are not parallel."""
    point_count = len(base_coordinates)
    for index in range(point_count):
        base_start = base_coordinates[index]
        base_end = base_coordinates[(index + 1) % point_count]
        offset_start = offset_coordinates[index]
        offset_end = offset_coordinates[(index + 1) % point_count]
        base_vector = (
            base_end[0] - base_start[0],
            base_end[1] - base_start[1],
        )
        offset_vector = (
            offset_end[0] - offset_start[0],
            offset_end[1] - offset_start[1],
        )
        base_length = hypot(*base_vector)
        offset_length = hypot(*offset_vector)
        if base_length == 0 or offset_length == 0:
            raise LoftIncompatibleError("A sidewall contour has a zero-length edge.")
        cross_product = (
            base_vector[0] * offset_vector[1] - base_vector[1] * offset_vector[0]
        )
        dot_product = (
            base_vector[0] * offset_vector[0] + base_vector[1] * offset_vector[1]
        )
        if (
            abs(cross_product) > _EDGE_DIRECTION_TOLERANCE * base_length * offset_length
            or dot_product <= 0
        ):
            raise LoftIncompatibleError(
                "Offsetting changed polygon edge correspondence."
            )


def _align_offset_ring(
    base_coordinates: list[tuple[float, float]],
    offset_coordinates: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Cyclically align a stable offset ring with its source vertices."""
    if len(base_coordinates) != len(offset_coordinates):
        raise LoftIncompatibleError("Offsetting changed the polygon vertex count.")
    if len(base_coordinates) < 3:
        raise LoftIncompatibleError("A sidewall contour has fewer than 3 vertices.")

    base_x, base_y = base_coordinates[0]
    best_shift = min(
        range(len(offset_coordinates)),
        key=lambda index: (
            (base_x - offset_coordinates[index][0]) ** 2
            + (base_y - offset_coordinates[index][1]) ** 2
        ),
    )
    aligned_coordinates = [
        offset_coordinates[(index + best_shift) % len(offset_coordinates)]
        for index in range(len(offset_coordinates))
    ]
    _validate_edge_correspondence(base_coordinates, aligned_coordinates)
    return aligned_coordinates


def _align_ring_to_ports(
    base_coordinates: list[tuple[float, float]],
    offset_coordinates: list[tuple[float, float]],
    ports: list[ResolvedPort],
    offset_um: float,
) -> list[tuple[float, float]]:
    """Keep each tapered port end on its fixed simulation-domain plane."""
    aligned_coordinates = [list(point) for point in offset_coordinates]
    tolerance_um = GEOMETRY_TOLERANCE_NM / UM_TO_NM
    matched_indices: dict[str, list[int]] = defaultdict(list)
    for index, base_point in enumerate(base_coordinates):
        for port in ports:
            normal_axis = next(
                axis for axis, component in enumerate(port.normal) if component
            )
            if normal_axis not in {0, 1}:
                raise FDTDGeometryError(
                    f"Guided port {port.name!r} is not in the component plane."
                )
            transverse_axis = 1 - normal_axis
            transverse_delta = (
                base_point[transverse_axis] - port.center[transverse_axis]
            )
            if (
                abs(base_point[normal_axis] - port.center[normal_axis]) > tolerance_um
                or abs(abs(transverse_delta) - port.width / 2) > tolerance_um
            ):
                continue
            half_width = port.width / 2 + offset_um
            if half_width <= 0:
                raise FDTDGeometryError(
                    f"Layer {port.layer_key!r} sidewall closes port {port.name!r}."
                )
            transverse_sign = 1.0 if transverse_delta > 0 else -1.0
            aligned_coordinates[index][normal_axis] = port.center[normal_axis]
            aligned_coordinates[index][transverse_axis] = (
                port.center[transverse_axis] + transverse_sign * half_width
            )
            matched_indices[port.name].append(index)
            break

    point_count = len(base_coordinates)
    for port in ports:
        indices = matched_indices[port.name]
        if len(indices) != 2 or (indices[0] - indices[1]) % point_count not in {
            1,
            point_count - 1,
        }:
            raise LoftIncompatibleError(
                f"Port {port.name!r} does not map to one polygon edge."
            )
    return [(point[0], point[1]) for point in aligned_coordinates]


def _ports_on_polygon(
    polygon: Polygon,
    ports: list[ResolvedPort],
) -> list[ResolvedPort]:
    """Return ports whose two edge endpoints are vertices of this polygon."""
    coordinates = _ring_coordinates(polygon)
    tolerance_um = GEOMETRY_TOLERANCE_NM / UM_TO_NM
    matched_ports = []
    for port in ports:
        normal_axis = next(
            axis for axis, component in enumerate(port.normal) if component
        )
        transverse_axis = 1 - normal_axis
        matching_vertices = [
            point
            for point in coordinates
            if abs(point[normal_axis] - port.center[normal_axis]) <= tolerance_um
            and abs(
                abs(point[transverse_axis] - port.center[transverse_axis])
                - port.width / 2
            )
            <= tolerance_um
        ]
        if matching_vertices:
            matched_ports.append(port)
    return matched_ports


def loft_section_polygons(
    layer: ResolvedLayer,
    ports: list[ResolvedPort],
) -> tuple[Polygon, Polygon]:
    """Prepare exact bottom and top contours with verified correspondence."""
    base_polygons = iter_polygons(layer.geometry, layer_key=layer.key)
    if len(base_polygons) != 1 or base_polygons[0].interiors:
        raise LoftIncompatibleError(
            "Lofting currently requires one connected polygon without holes."
        )
    base_coordinates = _ring_coordinates(base_polygons[0])
    section_polygons = []
    for normalized_z in (0.0, 1.0):
        offset_um = _sidewall_offset_um(layer, normalized_z)
        offset_geometry = layer.geometry.buffer(offset_um, join_style=2)
        try:
            offset_polygons = iter_polygons(offset_geometry, layer_key=layer.key)
        except FDTDGeometryError as error:
            raise LoftIncompatibleError(str(error)) from error
        if len(offset_polygons) != 1 or offset_polygons[0].interiors:
            raise LoftIncompatibleError(
                "Offsetting changed the polygon component or hole topology."
            )
        offset_coordinates = _ring_coordinates(offset_polygons[0])
        aligned_coordinates = _align_offset_ring(
            base_coordinates,
            offset_coordinates,
        )
        aligned_coordinates = _align_ring_to_ports(
            base_coordinates,
            aligned_coordinates,
            ports,
            offset_um,
        )
        section_polygon = Polygon(aligned_coordinates)
        if (
            section_polygon.is_empty
            or not section_polygon.is_valid
            or section_polygon.area <= 0
        ):
            raise LoftIncompatibleError("A port-aligned sidewall contour is invalid.")
        section_polygons.append(section_polygon)
    return section_polygons[0], section_polygons[1]


def _add_polygon_wire(kernel: Any, polygon: Polygon, z_nm: float) -> int:
    """Create one closed OCC wire from a preflighted polygon exterior."""
    point_tags = [
        kernel.addPoint(x_um * UM_TO_NM, y_um * UM_TO_NM, z_nm)
        for x_um, y_um in _ring_coordinates(polygon)
    ]
    line_tags = [
        kernel.addLine(point_tag, point_tags[(index + 1) % len(point_tags)])
        for index, point_tag in enumerate(point_tags)
    ]
    return kernel.addWire(line_tags, checkClosed=True)


def _add_lofted_layer_volumes(
    kernel: Any,
    layer: ResolvedLayer,
    section_polygons: tuple[Polygon, Polygon],
) -> list[int]:
    """Create continuous ruled solids from verified bottom and top contours."""
    bottom_polygon, top_polygon = section_polygons
    z_lower_um, z_upper_um = layer.z_bounds
    bottom_wire = _add_polygon_wire(kernel, bottom_polygon, z_lower_um * UM_TO_NM)
    top_wire = _add_polygon_wire(kernel, top_polygon, z_upper_um * UM_TO_NM)
    loft_entities = kernel.addThruSections(
        [bottom_wire, top_wire],
        makeSolid=True,
        makeRuled=True,
    )
    volume_tags = [
        entity_tag for dimension, entity_tag in loft_entities if dimension == 3
    ]
    if not volume_tags:
        raise FDTDGeometryError(f"Layer {layer.key!r} produced no lofted volume.")
    return volume_tags


def add_layer_volumes(
    kernel: Any,
    layer: ResolvedLayer,
    ports: list[ResolvedPort],
    *,
    geometry_tolerance_nm: float,
) -> list[int]:
    """Loft compatible polygon members, with a stepped fallback per member."""
    _validate_layer(layer)
    polygons = iter_polygons(layer.geometry, layer_key=layer.key)
    polygon_ports = [_ports_on_polygon(polygon, ports) for polygon in polygons]
    assigned_port_names = {
        port.name for matched_ports in polygon_ports for port in matched_ports
    }
    if assigned_port_names != {port.name for port in ports}:
        return _add_stepped_layer_volumes(
            kernel,
            layer,
            ports,
            geometry_tolerance_nm=geometry_tolerance_nm,
        )

    volume_tags = []
    for polygon, matched_ports in zip(polygons, polygon_ports, strict=True):
        polygon_layer = replace(layer, geometry=polygon)
        try:
            section_polygons = loft_section_polygons(polygon_layer, matched_ports)
        except LoftIncompatibleError:
            volume_tags.extend(
                _add_stepped_layer_volumes(
                    kernel,
                    polygon_layer,
                    matched_ports,
                    geometry_tolerance_nm=geometry_tolerance_nm,
                )
            )
        else:
            volume_tags.extend(
                _add_lofted_layer_volumes(
                    kernel,
                    polygon_layer,
                    section_polygons,
                )
            )
    return volume_tags


__all__ = ["add_layer_volumes", "loft_section_polygons"]
