"""Geometry extraction and creation for generic GMSH mesh generation.

This module handles extracting polygons from gdsfactory components
and creating 3D geometry in gmsh. Solver-agnostic — no Palace or Meep imports.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gsim.common.mesh import gmsh_utils

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)


@dataclass
class GeometryData:
    """Container for geometry data extracted from component."""

    polygons: list  # List of (layer_num, pts_x, pts_y, holes) tuples
    bbox: tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)
    layer_bboxes: dict  # layer_num -> (xmin, ymin, xmax, ymax)


def extract_geometry(component, stack: LayerStack) -> GeometryData:
    """Extract polygon geometry from a gdsfactory component.

    Args:
        component: gdsfactory Component
        stack: LayerStack for layer mapping

    Returns:
        GeometryData with polygons and bounding boxes
    """
    polygons = []
    global_bbox = [math.inf, math.inf, -math.inf, -math.inf]
    layer_bboxes = {}

    # Get polygons from component
    polygons_by_index = component.get_polygons()

    # Build layer_index -> GDS tuple mapping
    layout = component.kcl.layout
    index_to_gds = {}
    for layer_index in range(layout.layers()):
        if layout.is_valid_layer(layer_index):
            info = layout.get_info(layer_index)
            index_to_gds[layer_index] = (info.layer, info.datatype)

    # Build GDS tuple -> layer number mapping
    gds_to_layernum = {}
    for layer_data in stack.layers.values():
        gds_tuple = tuple(layer_data.gds_layer)
        gds_to_layernum[gds_tuple] = gds_tuple[0]

    # Convert polygons
    for layer_index, polys in polygons_by_index.items():
        gds_tuple = index_to_gds.get(layer_index)
        if gds_tuple is None:
            continue

        layernum = gds_to_layernum.get(gds_tuple)
        if layernum is None:
            continue

        for poly in polys:
            # Convert klayout polygon to lists (nm -> um)
            points = list(poly.each_point_hull())
            if len(points) < 3:
                continue

            pts_x = [pt.x / 1000.0 for pt in points]
            pts_y = [pt.y / 1000.0 for pt in points]

            # Extract holes from polygon
            holes = []
            for hole_idx in range(poly.holes()):
                hole_pts = list(poly.each_point_hole(hole_idx))
                if len(hole_pts) >= 3:
                    hx = [pt.x / 1000.0 for pt in hole_pts]
                    hy = [pt.y / 1000.0 for pt in hole_pts]
                    holes.append((hx, hy))

            polygons.append((layernum, pts_x, pts_y, holes))

            # Update bounding boxes
            xmin, xmax = min(pts_x), max(pts_x)
            ymin, ymax = min(pts_y), max(pts_y)

            global_bbox[0] = min(global_bbox[0], xmin)
            global_bbox[1] = min(global_bbox[1], ymin)
            global_bbox[2] = max(global_bbox[2], xmax)
            global_bbox[3] = max(global_bbox[3], ymax)

            if layernum not in layer_bboxes:
                layer_bboxes[layernum] = [xmin, ymin, xmax, ymax]
            else:
                bbox = layer_bboxes[layernum]
                bbox[0] = min(bbox[0], xmin)
                bbox[1] = min(bbox[1], ymin)
                bbox[2] = max(bbox[2], xmax)
                bbox[3] = max(bbox[3], ymax)

    return GeometryData(
        polygons=polygons,
        bbox=(global_bbox[0], global_bbox[1], global_bbox[2], global_bbox[3]),
        layer_bboxes=layer_bboxes,
    )


def get_layer_info(stack: LayerStack, gds_layer: int) -> dict | None:
    """Get layer info from stack by GDS layer number.

    Args:
        stack: LayerStack with layer definitions
        gds_layer: GDS layer number

    Returns:
        Dict with layer info or None if not found
    """
    for name, layer in stack.layers.items():
        if layer.gds_layer[0] == gds_layer:
            return {
                "name": name,
                "zmin": layer.zmin,
                "zmax": layer.zmax,
                "thickness": layer.zmax - layer.zmin,
                "material": layer.material,
                "type": layer.layer_type,
            }
    return None


def add_layer_volumes(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
) -> dict[str, list[int]]:
    """Extrude ALL layer polygons into 3D volumes.

    Unlike Palace's ``add_metals()`` which only handles conductor/via layers
    and creates shell surfaces for conductors, this function processes every
    layer type and always creates true volumes.

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with layer definitions

    Returns:
        Dict mapping layer_name to list of volume tags.
    """
    layer_volumes: dict[str, list[int]] = {}

    # Group polygons by GDS layer number
    polygons_by_layer: dict[int, list[tuple]] = {}
    for layernum, pts_x, pts_y, holes in geometry.polygons:
        if layernum not in polygons_by_layer:
            polygons_by_layer[layernum] = []
        polygons_by_layer[layernum].append((pts_x, pts_y, holes))

    # Process each layer
    for layernum, polys in polygons_by_layer.items():
        layer_info = get_layer_info(stack, layernum)
        if layer_info is None:
            continue

        layer_name = layer_info["name"]
        zmin = layer_info["zmin"]
        thickness = layer_info["thickness"]

        if thickness <= 0:
            continue

        if layer_name not in layer_volumes:
            layer_volumes[layer_name] = []

        for pts_x, pts_y, holes in polys:
            vol_tag = gmsh_utils.extrude_polygon(
                kernel, pts_x, pts_y, zmin, thickness, holes=holes
            )
            if vol_tag is not None:
                layer_volumes[layer_name].append(vol_tag)

    kernel.removeAllDuplicates()
    kernel.synchronize()

    return layer_volumes


def add_dielectrics(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    margin: float,
    air_margin: float,
    *,
    include_airbox: bool = True,
) -> dict:
    """Add dielectric boxes and optionally an airbox to gmsh.

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with dielectric definitions
        margin: XY margin around design (um)
        air_margin: Air box margin (um)
        include_airbox: Whether to add a surrounding airbox volume.
            Required for RF (Palace) simulations; typically not needed
            for photonic (Meep) simulations.

    Returns:
        Dict with material_name -> list of volume_tags
    """
    dielectric_tags: dict[str, list[int]] = {}

    # Get overall geometry bounds
    xmin, ymin, xmax, ymax = geometry.bbox
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin

    # Track overall z range
    z_min_all = math.inf
    z_max_all = -math.inf

    # Merge adjacent dielectrics of the same material into single boxes
    # to avoid coincident faces and sliver tets at shared z-boundaries.
    # E.g. box (SiO2, -3→0) + clad (SiO2, 0→1.8) → one box (SiO2, -3→1.8).
    merged: dict[str, list[tuple[float, float]]] = {}
    for dielectric in stack.dielectrics:
        material = dielectric["material"]
        d_zmin = dielectric["zmin"]
        d_zmax = dielectric["zmax"]
        if material not in merged:
            merged[material] = []
        merged[material].append((d_zmin, d_zmax))

    # For each material, sort ranges and merge overlapping/adjacent ones
    for material, ranges in merged.items():
        ranges.sort()
        combined: list[tuple[float, float]] = [ranges[0]]
        for lo, hi in ranges[1:]:
            prev_lo, prev_hi = combined[-1]
            if lo <= prev_hi + 1e-6:  # adjacent or overlapping
                combined[-1] = (prev_lo, max(prev_hi, hi))
            else:
                combined.append((lo, hi))
        merged[material] = combined

    # Create one GMSH box per merged z-range.
    # Adjacent boxes of different materials share a face at the same z;
    # removeAllDuplicates() after creation merges these shared faces so
    # fragment() works cleanly without sliver tets.
    for material, ranges in merged.items():
        dielectric_tags[material] = []
        for d_zmin, d_zmax in ranges:
            z_min_all = min(z_min_all, d_zmin)
            z_max_all = max(z_max_all, d_zmax)

            box_tag = gmsh_utils.create_box(
                kernel,
                xmin,
                ymin,
                d_zmin,
                xmax,
                ymax,
                d_zmax,
            )
            dielectric_tags[material].append(box_tag)

    kernel.removeAllDuplicates()
    kernel.synchronize()

    # Add surrounding airbox (needed for RF, not for photonics)
    if include_airbox:
        air_xmin = xmin - air_margin
        air_ymin = ymin - air_margin
        air_xmax = xmax + air_margin
        air_ymax = ymax + air_margin
        air_zmin = z_min_all - air_margin
        air_zmax = z_max_all + air_margin

        airbox_tag = kernel.addBox(
            air_xmin,
            air_ymin,
            air_zmin,
            air_xmax - air_xmin,
            air_ymax - air_ymin,
            air_zmax - air_zmin,
        )
        dielectric_tags["airbox"] = [airbox_tag]

    kernel.synchronize()

    return dielectric_tags


def _resolve_port_layer(
    port,
    component,
    stack: LayerStack,
) -> tuple[str, float, float] | None:
    """Resolve a gdsfactory port's layer to a stack layer name and z-range.

    Args:
        port: gdsfactory Port object
        component: gdsfactory Component (needed for klayout layer lookup)
        stack: LayerStack

    Returns:
        ``(layer_name, zmin, zmax)`` or ``None`` if no matching stack layer.
    """
    # Build GDS tuple → stack layer lookup
    gds_to_layer = {}
    for name, layer in stack.layers.items():
        gds_to_layer[tuple(layer.gds_layer)] = (name, layer)

    # port.layer is a klayout layer index (int); convert to GDS tuple
    port_layer_raw = port.layer
    if isinstance(port_layer_raw, int):
        layout = component.kcl.layout
        if not layout.is_valid_layer(port_layer_raw):
            return None
        info = layout.get_info(port_layer_raw)
        port_gds = (info.layer, info.datatype)
    else:
        port_gds = tuple(port_layer_raw)

    match = gds_to_layer.get(port_gds)
    if match is None:
        return None

    layer_name, layer = match
    return (layer_name, layer.zmin, layer.zmax)


__all__ = [
    "GeometryData",
    "add_dielectrics",
    "add_layer_volumes",
    "extract_geometry",
    "get_layer_info",
]
