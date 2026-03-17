"""Geometry extraction and creation for Palace mesh generation.

This module handles extracting polygons from gdsfactory components
and creating 3D geometry in gmsh.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from shapely import Polygon as ShapelyPolygon
from shapely import buffer
from shapely.ops import unary_union

from gsim.palace.ports.config import PortType

from . import gmsh_utils

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.palace.models.pec import PECBlockConfig
    from gsim.palace.ports.config import PalacePort

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


def _merge_via_polygons(
    polys: list[tuple[list[float], list[float], list]],
    merge_distance: float,
) -> list[tuple[list[float], list[float], list]]:
    """Merge nearby via polygons using oversize/union/undersize.

    Converts polygon coordinate lists to shapely, buffers outward by
    half the merge distance so nearby vias overlap, unions them, then
    buffers back inward to restore the original outline.

    Args:
        polys: List of (pts_x, pts_y, holes) tuples
        merge_distance: Max gap between vias to merge (um)

    Returns:
        Merged polygon list in the same format
    """
    if len(polys) <= 1 or merge_distance <= 0:
        return polys

    # Small epsilon ensures vias exactly at merge_distance still overlap after buffering
    _MERGE_EPSILON = 0.01  # um
    offset = merge_distance / 2 + _MERGE_EPSILON

    shapely_polys = []
    for pts_x, pts_y, _holes in polys:
        coords = list(zip(pts_x, pts_y, strict=False))
        if len(coords) >= 3:
            shapely_polys.append(ShapelyPolygon(coords))

    if not shapely_polys:
        return polys

    # Oversize → union → undersize
    buffered = [buffer(p, offset, join_style="mitre") for p in shapely_polys]
    merged = buffer(unary_union(buffered), -offset, join_style="mitre")

    # Convert back to coordinate lists
    result = []
    # unary_union may return Polygon or MultiPolygon
    geoms = merged.geoms if hasattr(merged, "geoms") else [merged]
    for geom in geoms:
        if geom.is_empty:
            continue
        xs, ys = zip(*geom.exterior.coords[:-1], strict=True)  # drop closing duplicate
        holes = []
        for interior in geom.interiors:
            hx, hy = zip(*interior.coords[:-1], strict=True)
            holes.append((list(hx), list(hy)))
        result.append((list(xs), list(ys), holes))

    n_before = len(polys)
    n_after = len(result)
    if n_before != n_after:
        logger.info(
            "Via merging: %d polygons → %d (distance=%.1f um)",
            n_before,
            n_after,
            merge_distance,
        )

    return result


def add_metals(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    planar_conductors: bool = False,
    merge_via_distance: float = 2.0,
) -> dict:
    """Add metal and via geometries to gmsh.

    Creates extruded volumes for vias and shells (surfaces) for conductors.
    If planar_conductors is True, conductors are treated as 2D surfaces (PEC).

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with layer definitions
        planar_conductors: If True, treat conductors as 2D PEC surfaces
        merge_via_distance: Max gap between vias to merge (um). Nearby
            via polygons within this distance are combined into a single
            polygon before meshing, drastically reducing mesh complexity.

    Returns:
        Dict with layer_name -> {"volumes": [...], "surfaces_xy": [...],
        "surfaces_z": [...]} where volumes contains raw int tags for vias
        and (volumetag, surface_tags) tuples for conductors.
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

        # Merge nearby via polygons before creating gmsh surfaces
        if layer_type == "via":
            polys = _merge_via_polygons(polys, merge_via_distance)

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

    # Record bounding boxes of via volumes BEFORE removeAllDuplicates
    # so we can re-identify them after tags get renumbered.
    _via_bboxes: dict[str, list[tuple[float, ...]]] = {}
    kernel.synchronize()
    for layer_name, tag_info in metal_tags.items():
        for vtag in tag_info["volumes"]:
            if isinstance(vtag, int):
                bbox = kernel.getBoundingBox(3, vtag)
                _via_bboxes.setdefault(layer_name, []).append(bbox)

    kernel.removeAllDuplicates()
    kernel.synchronize()

    # Re-identify via volumes by matching bounding boxes
    all_vols = kernel.getEntities(3)
    for layer_name, bboxes in _via_bboxes.items():
        metal_tags[layer_name]["volumes"] = []
        for target_bbox in bboxes:
            for _, vtag in all_vols:
                try:
                    bbox = kernel.getBoundingBox(3, vtag)
                except Exception:
                    logger.debug("Could not get bbox for volume %d, skipping", vtag)
                    continue
                if all(
                    abs(a - b) < 0.01 for a, b in zip(bbox, target_bbox, strict=True)
                ):
                    metal_tags[layer_name]["volumes"].append(vtag)
                    break

    # Extract shell surfaces from conductor volumes.
    # After removeAllDuplicates, some volume tags may have been invalidated
    # (e.g., a conductor volume that shared faces with a via volume).
    current_vols = {t for _, t in kernel.getEntities(3)}
    for layer_name, vol_tags in _conductor_volumes.items():
        for volumetag in vol_tags:
            if volumetag not in current_vols:
                logger.warning(
                    "Conductor volume %d on %s invalidated by dedup",
                    volumetag,
                    layer_name,
                )
                continue
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


def extract_pec_polygons(component, gds_layer: tuple[int, int]) -> list:
    """Extract polygons from an arbitrary GDS layer on a component.

    Uses the same klayout polygon parsing pattern as ``extract_geometry()``.

    Args:
        component: gdsfactory Component
        gds_layer: GDS layer tuple (layer, datatype) to extract polygons from

    Returns:
        List of (pts_x, pts_y, holes) tuples in microns
    """
    polygons_by_index = component.get_polygons()

    layout = component.kcl.layout
    index_to_gds = {}
    for layer_index in range(layout.layers()):
        if layout.is_valid_layer(layer_index):
            info = layout.get_info(layer_index)
            index_to_gds[layer_index] = (info.layer, info.datatype)

    result = []
    for layer_index, polys in polygons_by_index.items():
        if index_to_gds.get(layer_index) != gds_layer:
            continue
        for poly in polys:
            points = list(poly.each_point_hull())
            if len(points) < 3:
                continue
            pts_x = [pt.x / 1000.0 for pt in points]
            pts_y = [pt.y / 1000.0 for pt in points]
            holes = []
            for hole_idx in range(poly.holes()):
                hole_pts = list(poly.each_point_hole(hole_idx))
                if len(hole_pts) >= 3:
                    hx = [pt.x / 1000.0 for pt in hole_pts]
                    hy = [pt.y / 1000.0 for pt in hole_pts]
                    holes.append((hx, hy))
            result.append((pts_x, pts_y, holes))

    return result


def add_pec_blocks(
    kernel,
    component,
    pec_configs: list[PECBlockConfig],
    stack: LayerStack,
) -> dict:
    """Add PEC block geometries to gmsh.

    For each PEC config:
    1. Extract polygons from the specified GDS layer
    2. Create surfaces at ``from_layer.zmin``
    3. Fuse overlapping surfaces
    4. Extrude to ``to_layer.zmax - from_layer.zmin``
    5. Remove volumes but keep shell surfaces
    6. Classify shells as xy/z using ``is_vertical_surface()``

    Args:
        kernel: gmsh OCC kernel
        component: gdsfactory Component
        pec_configs: List of PECBlockConfig objects
        stack: LayerStack with layer definitions

    Returns:
        Dict: ``{"pec_block_0": {"surfaces_xy": [...], "surfaces_z": [...]}, ...}``
    """
    pec_block_tags: dict[str, dict[str, list[int]]] = {}

    for idx, cfg in enumerate(pec_configs):
        block_name = f"pec_block_{idx}"
        from_layer = stack.layers.get(cfg.from_layer)
        to_layer = stack.layers.get(cfg.to_layer)
        if from_layer is None or to_layer is None:
            continue

        polys = extract_pec_polygons(component, cfg.gds_layer)
        if not polys:
            continue

        zmin = from_layer.zmin
        height = to_layer.zmax - from_layer.zmin

        # Create surfaces for each polygon
        surfaces = []
        for pts_x, pts_y, holes in polys:
            surfacetag = gmsh_utils.create_polygon_surface(
                kernel, pts_x, pts_y, zmin, holes=holes
            )
            if surfacetag is not None:
                surfaces.append(surfacetag)

        if not surfaces:
            continue

        # Fuse overlapping surfaces
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

        # Extrude to create volumes
        volumes = []
        for surfacetag in surfaces:
            result = kernel.extrude([(2, surfacetag)], 0, 0, height)
            volumetag = result[1][1]
            volumes.append(volumetag)

        kernel.removeAllDuplicates()
        kernel.synchronize()

        # Extract shell surfaces and classify as xy/z
        xy_tags: list[int] = []
        z_tags: list[int] = []
        for volumetag in volumes:
            _, surfaceloops = kernel.getSurfaceLoops(volumetag)
            if surfaceloops:
                for tag in surfaceloops[0]:
                    if gmsh_utils.is_vertical_surface(tag):
                        z_tags.append(tag)
                    else:
                        xy_tags.append(tag)
            kernel.remove([(3, volumetag)])

        kernel.synchronize()

        pec_block_tags[block_name] = {
            "surfaces_xy": xy_tags,
            "surfaces_z": z_tags,
        }

    return pec_block_tags


def build_entities(
    metal_tags: dict,
    dielectric_tags: dict,
    port_tags: dict,
    port_info: list,
    pec_block_tags: dict | None = None,
    stack: LayerStack | None = None,
) -> list[gmsh_utils.Entity]:
    """Convert geometry tag dicts into Entity objects for the boolean pipeline.

    Mesh-order convention (lower = higher priority, gets cut first):
        0  - conductor (2D PEC) surfaces and PEC block surfaces
        1  - via volumes (3D, higher priority than dielectrics) and port surfaces
        2  - dielectrics (non-airbox volumes)
        3  - airbox volume (lowest priority, carved by everything else)

    Args:
        metal_tags: from ``add_metals()``
        dielectric_tags: from ``add_dielectrics()``
        port_tags: from ``add_ports()``
        port_info: metadata list from ``add_ports()``
        pec_block_tags: from ``add_pec_blocks()``, optional
        stack: LayerStack for distinguishing via vs conductor layers

    Returns:
        List of Entity objects ready for ``run_boolean_pipeline()``.
    """
    Entity = gmsh_utils.Entity
    entities: list[gmsh_utils.Entity] = []

    # Build set of via layer names for quick lookup
    via_layers: set[str] = set()
    if stack:
        via_layers = {
            n for n, layer in stack.layers.items() if layer.layer_type == "via"
        }

    # --- Conductors and vias ---
    for layer_name, tag_info in metal_tags.items():
        is_via = layer_name in via_layers

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

        if tag_info["volumes"]:
            if is_via:
                # Via volumes: 3D entities, higher priority than dielectrics
                via_vol_tags = [
                    item for item in tag_info["volumes"] if isinstance(item, int)
                ]
                if via_vol_tags:
                    entities.append(
                        Entity(
                            name=layer_name,
                            dim=3,
                            mesh_order=1,
                            tags=via_vol_tags,
                        )
                    )
            else:
                # Volumetric conductors: shell surfaces (volume already removed)
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

    # --- PEC block surfaces (dim=2, highest priority) ---
    if pec_block_tags:
        for block_name, tag_info in pec_block_tags.items():
            if tag_info["surfaces_xy"]:
                entities.append(
                    Entity(
                        name=f"{block_name}_xy",
                        dim=2,
                        mesh_order=0,
                        tags=tag_info["surfaces_xy"],
                    )
                )
            if tag_info["surfaces_z"]:
                entities.append(
                    Entity(
                        name=f"{block_name}_z",
                        dim=2,
                        mesh_order=0,
                        tags=tag_info["surfaces_z"],
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

            zmin = target_layer.zmin
            hw = port.width / 2
            hl = (port.length or port.width) / 2

            # Determine axis from orientation
            angle = port.orientation % 360
            is_y_axis = 45 <= angle < 135 or 225 <= angle < 315

            surfaces = []
            for cx, cy in port.centers:
                if is_y_axis:
                    surf = gmsh_utils.create_port_rectangle(
                        kernel, cx - hw, cy - hl, zmin, cx + hw, cy + hl, zmin
                    )
                else:
                    surf = gmsh_utils.create_port_rectangle(
                        kernel, cx - hl, cy - hw, zmin, cx + hl, cy + hw, zmin
                    )
                surfaces.append(surf)

            port_tags[f"P{port_num}"] = surfaces

            port_info.append(
                {
                    "portnumber": port_num,
                    "R0": port.resistance,
                    "L0": port.inductance,
                    "C0": port.capacitance,
                    "type": "cpw",
                    "elements": [
                        {"surface_idx": i, "direction": port.directions[i]}
                        for i in range(len(port.centers))
                    ],
                    "width": port.width,
                    "length": port.length or port.width,
                    "zmin": zmin,
                    "zmax": zmin,
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
                    "R0": port.resistance,
                    "L0": port.inductance,
                    "C0": port.capacitance,
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
            zmin = target_layer.zmin
            zmax = target_layer.zmax

            if port.port_type == PortType.LUMPED:
                hl = (port.length or port.width) / 2
                if port.direction in ("x", "-x"):
                    surfacetag = gmsh_utils.create_port_rectangle(
                        kernel, x - hl, y - hw, zmin, x + hl, y + hw, zmin
                    )
                    length = 2 * hl
                    width = port.width
                else:
                    surfacetag = gmsh_utils.create_port_rectangle(
                        kernel, x - hw, y - hl, zmin, x + hw, y + hl, zmin
                    )
                    length = port.width
                    width = 2 * hl

                port_tags[f"P{port_num}"] = [surfacetag]
                port_info.append(
                    {
                        "portnumber": port_num,
                        "R0": port.resistance,
                        "L0": port.inductance,
                        "C0": port.capacitance,
                        "type": "lumped",
                        "direction": port.direction.upper(),
                        "length": length,
                        "width": width,
                        "xmin": x - hl if port.direction in ("x", "-x") else x - hw,
                        "xmax": x + hl if port.direction in ("x", "-x") else x + hw,
                        "ymin": y - hw if port.direction in ("x", "-x") else y - hl,
                        "ymax": y + hw if port.direction in ("x", "-x") else y + hl,
                        "zmin": zmin,
                        "zmax": zmin,
                    }
                )
            else:
                layer_zmin, layer_zmax = stack.get_z_range()
                zmin = zmin - port.z_margin
                zmax = zmax + port.z_margin
                zmin = max(zmin, layer_zmin)
                zmax = min(zmax, layer_zmax)
                angle = port.orientation % 360
                is_y_axis = 45 <= angle < 135 or 225 <= angle < 315
                if is_y_axis:
                    xmin = x - hw - port.lateral_margin
                    xmax = x + hw + port.lateral_margin
                    ymin = y
                    ymax = y
                else:
                    xmin = x
                    xmax = x
                    ymin = y - hw - port.lateral_margin
                    ymax = y + hw + port.lateral_margin
                surfacetag = gmsh_utils.create_port_rectangle(
                    kernel, xmin, ymin, zmin, xmax, ymax, zmax
                )
                port_tags[f"P{port_num}"] = [surfacetag]
                port_info.append(
                    {
                        "portnumber": port_num,
                        "type": "waveport",
                        "width": port.width + 2 * port.lateral_margin,
                        "xmin": xmin,
                        "xmax": xmax,
                        "ymin": ymin,
                        "ymax": ymax,
                        "zmin": zmin,
                        "zmax": zmax,
                    }
                )
        port_num += 1

    kernel.synchronize()

    return port_tags, port_info


__all__ = [
    "GeometryData",
    "add_dielectrics",
    "add_metals",
    "add_pec_blocks",
    "add_ports",
    "build_entities",
    "extract_geometry",
    "extract_pec_polygons",
    "get_layer_info",
]
