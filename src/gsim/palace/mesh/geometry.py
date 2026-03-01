"""Geometry extraction and creation for Palace mesh generation.

Generic geometry helpers (GeometryData, extract_geometry, get_layer_info,
add_dielectrics) are provided by ``gsim.common.mesh.geometry`` and
re-exported here for backward compatibility.

This module keeps Palace-specific functions: ``add_metals`` and ``add_ports``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gsim.common.mesh import gmsh_utils

# Re-export generic helpers so existing Palace code keeps working
from gsim.common.mesh.geometry import (
    GeometryData,
    add_dielectrics,
    extract_geometry,
    get_layer_info,
)

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.ports.config import PalacePort


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

        for pts_x, pts_y, holes in polys:
            # Create extruded polygon
            surfacetag = gmsh_utils.create_polygon_surface(
                kernel, pts_x, pts_y, zmin, holes=holes
            )
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
