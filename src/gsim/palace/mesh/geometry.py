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

    # Track conductor volumes for deferred surface extraction.
    # Surface loop tags are only queried *after* removeAllDuplicates()
    # so that stale tags from merged sub-entities cannot cause crashes.
    _conductor_volumes: dict[str, list[int]] = {}

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

        # Create surfaces for all polygons on this layer
        surfaces = []
        for pts_x, pts_y, holes in polys:
            surfacetag = gmsh_utils.create_polygon_surface(
                kernel, pts_x, pts_y, zmin, holes=holes
            )
            if surfacetag is not None:
                surfaces.append(surfacetag)

        if not surfaces:
            continue

        if layer_type == "conductor" and (planar_conductors or thickness == 0):
            # Zero-thickness or explicitly planar → 2D PEC surface
            metal_tags[layer_name]["surfaces_xy"].extend(surfaces)
        elif thickness > 0:
            # Fuse overlapping same-layer surfaces before extrusion so that
            # overlapping polygons (e.g. ground planes and spines in a GSG
            # electrode) become a single merged surface per layer.
            if len(surfaces) > 1:
                dimtags = [(2, s) for s in surfaces]
                fused, _ = kernel.fuse(
                    [dimtags[0]],
                    dimtags[1:],
                    removeObject=True,
                    removeTool=True,
                )
                kernel.synchronize()
                surfaces = [t for d, t in fused if d == 2]

            for surfacetag in surfaces:
                result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
                volumetag = result[1][1]

                if layer_type == "via":
                    # Keep vias as volumes
                    metal_tags[layer_name]["volumes"].append(volumetag)
                else:
                    # Defer shell extraction until after removeAllDuplicates
                    _conductor_volumes.setdefault(layer_name, []).append(volumetag)

    kernel.removeAllDuplicates()
    kernel.synchronize()

    # Extract shell surfaces from conductor volumes now that tags are stable
    for layer_name, vol_tags in _conductor_volumes.items():
        for volumetag in vol_tags:
            _, surfaceloops = kernel.getSurfaceLoops(volumetag)
            if surfaceloops:
                metal_tags[layer_name]["volumes"].append((volumetag, surfaceloops[0]))
            kernel.remove([(3, volumetag)])

    if _conductor_volumes:
        kernel.synchronize()

    return metal_tags


def add_dielectrics(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    margin: float,
    air_margin: float = 0.0,
) -> dict:
    """Add dielectric volumes to gmsh.

    All boxes are created at their exact geometric bounds.  Coincident
    faces and overlapping volumes are resolved later by
    ``run_boolean_pipeline``, which performs priority-based cuts and
    conformal fragmentation — no artificial offsets needed.

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with dielectric definitions
        margin: XY margin around design (um)
        air_margin: Extra margin for the surrounding airbox (um).
            When > 0, an enclosing airbox is created.  The boolean
            pipeline will carve the dielectrics out of it
            automatically.

    Returns:
        Dict with material_name -> list of volume_tags
    """
    dielectric_tags: dict[str, list[int]] = {}

    xmin, ymin, xmax, ymax = geometry.bbox
    xmin -= margin
    ymin -= margin
    xmax += margin
    ymax += margin

    z_min_all = math.inf
    z_max_all = -math.inf

    for dielectric in stack.dielectrics:
        material = dielectric["material"]

        # When building an explicit airbox, skip the dielectric air layer
        if material == "air" and air_margin > 0:
            continue

        d_zmin = dielectric["zmin"]
        d_zmax = dielectric["zmax"]

        z_min_all = min(z_min_all, d_zmin)
        z_max_all = max(z_max_all, d_zmax)

        dielectric_tags.setdefault(material, [])

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

    # Surrounding airbox (boolean pipeline handles the overlap)
    if air_margin > 0:
        airbox_tag = gmsh_utils.create_box(
            kernel,
            xmin - air_margin,
            ymin - air_margin,
            z_min_all - air_margin,
            xmax + air_margin,
            ymax + air_margin,
            z_max_all + air_margin,
        )
        dielectric_tags["airbox"] = [airbox_tag]

    kernel.synchronize()

    return dielectric_tags


def build_entities(
    metal_tags: dict,
    dielectric_tags: dict,
    port_tags: dict,
    port_info: list,
) -> list[gmsh_utils.Entity]:
    """Convert geometry tag dicts into Entity objects for the boolean pipeline.

    Mesh-order convention (lower = higher priority, gets cut first):
        0  - conductor (2D PEC) surfaces
        1  - port surfaces
        2  - dielectrics (non-airbox volumes)
        3  - airbox volume (lowest priority, carved by everything else)

    Args:
        metal_tags: from ``add_metals()``
        dielectric_tags: from ``add_dielectrics()``
        port_tags: from ``add_ports()``
        port_info: metadata list from ``add_ports()``

    Returns:
        List of Entity objects ready for ``run_boolean_pipeline()``.
    """
    Entity = gmsh_utils.Entity
    entities: list[gmsh_utils.Entity] = []

    # --- Conductors (dim=2 surfaces, highest priority) ---
    for layer_name, tag_info in metal_tags.items():
        # PEC / zero-thickness surfaces
        if tag_info["surfaces_xy"]:
            entities.append(
                Entity(
                    name=f"{layer_name}_pec",
                    dim=2,
                    mesh_order=0,
                    tags=tag_info["surfaces_xy"],
                )
            )

        # Volumetric conductors: shell surfaces (volume already removed)
        if tag_info["volumes"]:
            xy_tags = []
            z_tags = []
            for item in tag_info["volumes"]:
                if isinstance(item, tuple):
                    _volumetag, surface_tags = item
                    for tag in surface_tags:
                        if gmsh_utils.is_vertical_surface(tag):
                            z_tags.append(tag)
                        else:
                            xy_tags.append(tag)
            if xy_tags:
                entities.append(
                    Entity(
                        name=f"{layer_name}_xy",
                        dim=2,
                        mesh_order=0,
                        tags=xy_tags,
                    )
                )
            if z_tags:
                entities.append(
                    Entity(
                        name=f"{layer_name}_z",
                        dim=2,
                        mesh_order=0,
                        tags=z_tags,
                    )
                )

    # --- Port surfaces (dim=2) ---
    for port_name, surf_tags in port_tags.items():
        port_num = int(port_name[1:])
        info = next(
            (p for p in port_info if p["portnumber"] == port_num),
            None,
        )
        if info and info.get("type") == "cpw":
            # One entity per CPW element
            for i, tag in enumerate(surf_tags):
                entities.append(
                    Entity(
                        name=f"{port_name}_E{i}",
                        dim=2,
                        mesh_order=1,
                        tags=[tag],
                    )
                )
        else:
            entities.append(
                Entity(
                    name=port_name,
                    dim=2,
                    mesh_order=1,
                    tags=surf_tags,
                )
            )

    # --- Dielectric volumes (dim=3) ---
    for material, vol_tags in dielectric_tags.items():
        order = 3 if material == "airbox" else 2
        entities.append(
            Entity(
                name=material,
                dim=3,
                mesh_order=order,
                tags=vol_tags,
            )
        )

    return entities


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
    "build_entities",
    "extract_geometry",
    "get_layer_info",
]
