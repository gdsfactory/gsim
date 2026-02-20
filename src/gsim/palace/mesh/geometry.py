"""Geometry extraction and creation for Palace mesh generation.

This module handles extracting polygons from gdsfactory components
and creating 3D geometry in gmsh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import gmsh_utils

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.ports.config import PalacePort


@dataclass
class GeometryData:
    """Container for geometry data extracted from component."""

    polygons: list  # List of (layer_num, pts_x, pts_y) tuples
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

            polygons.append((layernum, pts_x, pts_y))

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


def add_metals(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    planar_conductors: bool = False,
) -> dict:
    """Add metal and via geometries to gmsh.

    Creates extruded volumes for vias and shells (surfaces) for conductors.
    If planar_conductors is True, conductors are treated as 2D surfaces (PEC).

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with layer definitions
        planar_conductors: If True, treat conductors as 2D PEC surfaces

    Returns:
        Dict with layer_name -> list of (surface_tags_xy, surface_tags_z) for
        conductors, or volume_tags for vias.
    """
    # layer_name -> {"volumes": [], "surfaces_xy": [], "surfaces_z": []}
    metal_tags = {}

    # Group polygons by layer
    polygons_by_layer = {}
    for layernum, pts_x, pts_y in geometry.polygons:
        if layernum not in polygons_by_layer:
            polygons_by_layer[layernum] = []
        polygons_by_layer[layernum].append((pts_x, pts_y))

    # Process each layer
    for layernum, polys in polygons_by_layer.items():
        layer_info = get_layer_info(stack, layernum)
        if layer_info is None:
            continue

        layer_name = layer_info["name"]
        layer_type = layer_info["type"]
        zmin = layer_info["zmin"]
        thickness = layer_info["thickness"]

        if layer_type not in ("conductor", "via"):
            continue

        if layer_name not in metal_tags:
            metal_tags[layer_name] = {
                "volumes": [],
                "surfaces_xy": [],
                "surfaces_z": [],
            }

        for pts_x, pts_y in polys:
            # Create extruded polygon
            surfacetag = gmsh_utils.create_polygon_surface(kernel, pts_x, pts_y, zmin)
            if surfacetag is None:
                continue

            if planar_conductors and layer_type == "conductor":
                # For planar conductors, keep as 2D surface (PEC boundary)
                metal_tags[layer_name]["surfaces_xy"].append(surfacetag)
            elif thickness > 0:
                result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
                volumetag = result[1][1]

                if layer_type == "via":
                    # Keep vias as volumes
                    metal_tags[layer_name]["volumes"].append(volumetag)
                else:
                    # For conductors, get shell surfaces and remove volume
                    _, surfaceloops = kernel.getSurfaceLoops(volumetag)
                    if surfaceloops:
                        metal_tags[layer_name]["volumes"].append(
                            (volumetag, surfaceloops[0])
                        )
                    kernel.remove([(3, volumetag)])

    kernel.removeAllDuplicates()
    kernel.synchronize()

    return metal_tags


def add_dielectrics(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    margin: float,
    air_margin: float,
) -> dict:
    """Add dielectric boxes and airbox to gmsh.

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with dielectric definitions
        margin: XY margin around design (um)
        air_margin: Air box margin (um)

    Returns:
        Dict with material_name -> list of volume_tags
    """
    dielectric_tags = {}

    # Get overall geometry bounds
    xmin, ymin, xmax, ymax = geometry.bbox
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin

    # Track overall z range
    z_min_all = math.inf
    z_max_all = -math.inf

    # Sort dielectrics by z (top to bottom for correct layering)
    sorted_dielectrics = sorted(
        stack.dielectrics, key=lambda d: d["zmax"], reverse=True
    )

    # Add dielectric boxes
    offset = 0
    offset_delta = margin / 20

    for dielectric in sorted_dielectrics:
        material = dielectric["material"]
        d_zmin = dielectric["zmin"]
        d_zmax = dielectric["zmax"]

        z_min_all = min(z_min_all, d_zmin)
        z_max_all = max(z_max_all, d_zmax)

        if material not in dielectric_tags:
            dielectric_tags[material] = []

        # Create box with slight offset to avoid mesh issues
        box_tag = gmsh_utils.create_box(
            kernel,
            xmin - offset,
            ymin - offset,
            d_zmin,
            xmax + offset,
            ymax + offset,
            d_zmax,
        )
        dielectric_tags[material].append(box_tag)

        # Alternate offset to avoid coincident faces
        offset = offset_delta if offset == 0 else 0

    # Add surrounding airbox
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


def add_ports(
    kernel,
    ports: list[PalacePort],
    stack: LayerStack,
) -> tuple[dict, list]:
    """Add port surfaces to gmsh.

    Args:
        kernel: gmsh OCC kernel
        ports: List of PalacePort objects (single or multi-element)
        stack: Layer stack

    Returns:
        (port_tags dict, port_info list)

    For single-element ports: port_tags["P{num}"] = [surface_tag]
    For multi-element ports: port_tags["P{num}"] = [surface_tag, surface_tag, ...]
    """
    from gsim.palace.ports.config import PortGeometry

    port_tags = {}  # "P{num}" -> [surface_tag(s)]
    port_info = []
    port_num = 1

    for port in ports:
        if port.multi_element:
            # Multi-element port (CPW)
            if port.layer is None or port.centers is None or port.directions is None:
                continue
            target_layer = stack.layers.get(port.layer)
            if target_layer is None:
                continue

            z = target_layer.zmin
            hw = port.width / 2
            hl = (port.length or port.width) / 2

            # Determine axis from orientation
            angle = port.orientation % 360
            is_y_axis = 45 <= angle < 135 or 225 <= angle < 315

            surfaces = []
            for cx, cy in port.centers:
                if is_y_axis:
                    surf = gmsh_utils.create_port_rectangle(
                        kernel, cx - hw, cy - hl, z, cx + hw, cy + hl, z
                    )
                else:
                    surf = gmsh_utils.create_port_rectangle(
                        kernel, cx - hl, cy - hw, z, cx + hl, cy + hw, z
                    )
                surfaces.append(surf)

            port_tags[f"P{port_num}"] = surfaces

            port_info.append(
                {
                    "portnumber": port_num,
                    "Z0": port.impedance,
                    "type": "cpw",
                    "elements": [
                        {"surface_idx": i, "direction": port.directions[i]}
                        for i in range(len(port.centers))
                    ],
                    "width": port.width,
                    "length": port.length or port.width,
                    "zmin": z,
                    "zmax": z,
                }
            )

        elif port.geometry == PortGeometry.VIA:
            # Via port: vertical between two layers
            if port.from_layer is None or port.to_layer is None:
                continue
            from_layer = stack.layers.get(port.from_layer)
            to_layer = stack.layers.get(port.to_layer)
            if from_layer is None or to_layer is None:
                continue

            x, y = port.center
            hw = port.width / 2

            if from_layer.zmin < to_layer.zmin:
                zmin = from_layer.zmax
                zmax = to_layer.zmin
            else:
                zmin = to_layer.zmax
                zmax = from_layer.zmin

            # Create vertical port surface
            if port.direction in ("x", "-x"):
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x, y - hw, zmin, x, y + hw, zmax
                )
            else:
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x - hw, y, zmin, x + hw, y, zmax
                )

            port_tags[f"P{port_num}"] = [surfacetag]
            port_info.append(
                {
                    "portnumber": port_num,
                    "Z0": port.impedance,
                    "type": "via",
                    "direction": "Z",
                    "length": zmax - zmin,
                    "width": port.width,
                    "xmin": x - hw if port.direction in ("y", "-y") else x,
                    "xmax": x + hw if port.direction in ("y", "-y") else x,
                    "ymin": y - hw if port.direction in ("x", "-x") else y,
                    "ymax": y + hw if port.direction in ("x", "-x") else y,
                    "zmin": zmin,
                    "zmax": zmax,
                }
            )

        else:
            # Inplane port: horizontal on single layer
            if port.layer is None:
                continue
            target_layer = stack.layers.get(port.layer)
            if target_layer is None:
                continue

            x, y = port.center
            hw = port.width / 2
            z = target_layer.zmin

            hl = (port.length or port.width) / 2
            if port.direction in ("x", "-x"):
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x - hl, y - hw, z, x + hl, y + hw, z
                )
                length = 2 * hl
                width = port.width
            else:
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, x - hw, y - hl, z, x + hw, y + hl, z
                )
                length = port.width
                width = 2 * hl

            port_tags[f"P{port_num}"] = [surfacetag]
            port_info.append(
                {
                    "portnumber": port_num,
                    "Z0": port.impedance,
                    "type": "lumped",
                    "direction": port.direction.upper(),
                    "length": length,
                    "width": width,
                    "xmin": x - hl if port.direction in ("x", "-x") else x - hw,
                    "xmax": x + hl if port.direction in ("x", "-x") else x + hw,
                    "ymin": y - hw if port.direction in ("x", "-x") else y - hl,
                    "ymax": y + hw if port.direction in ("x", "-x") else y + hl,
                    "zmin": z,
                    "zmax": z,
                }
            )

        port_num += 1

    kernel.synchronize()

    return port_tags, port_info


__all__ = [
    "GeometryData",
    "add_dielectrics",
    "add_metals",
    "add_ports",
    "extract_geometry",
    "get_layer_info",
]
