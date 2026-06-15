"""Geometry extraction and creation for Palace mesh generation.

This module handles extracting polygons from gdsfactory components
and creating 3D geometry in gmsh.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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


def extract_geometry(
    component, stack: LayerStack, *, decimate_tolerance: float | None = None
) -> GeometryData:
    """Extract polygon geometry from a gdsfactory component.

    Args:
        component: gdsfactory Component
        stack: LayerStack for layer mapping
        decimate_tolerance: If set, simplify polygons with Douglas-Peucker
            using this relative tolerance (passed to ``decimate()``).
            Typical values: 0.001 (conservative) to 0.01 (aggressive).

    Returns:
        GeometryData with polygons and bounding boxes
    """
    polygons = []
    global_bbox = [math.inf, math.inf, -math.inf, -math.inf]
    layer_bboxes = {}

    # Get polygons from component
    polygons_by_index = component.get_polygons()

    if decimate_tolerance is not None:
        from gsim.common.polygon_utils import decimate

        total_before = 0
        total_after = 0
        decimated_by_index = {}
        for layer_index, polys in polygons_by_index.items():
            total_before += sum(p.num_points_hull() for p in polys)
            decimated_by_index[layer_index] = decimate(
                polys, relative_tolerance=decimate_tolerance, verbose=True
            )
            total_after += sum(
                p.num_points_hull() for p in decimated_by_index[layer_index]
            )
        if total_before > 0:
            pct = (total_before - total_after) / total_before * 100
            logger.info(
                "Decimation: %d -> %d pts (%.1f%% removed)",
                total_before,
                total_after,
                pct,
            )
        polygons_by_index = decimated_by_index

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


def _snap_via_z_range(
    stack: LayerStack,
    via_name: str,
    zmin: float,
    zmax: float,
    tol: float = 0.05,
) -> tuple[float, float]:
    """Snap a via's z-range so it doesn't sliver into adjacent conductor layers.

    PDK stacks routinely model vias with z-ranges that overlap the metal layers
    they connect to (e.g. IHP `vmim` z=[5.58, 6.24] vs `topmetal1` z=[6.23, 8.23]
    — a 10 nm overlap to model fab interdiffusion). After mesh fragmentation,
    that overlap becomes a sliver volume too thin for gmsh to mesh, producing
    "No elements in volume N" warnings.

    If the via's zmax exceeds an adjacent conductor's zmin by less than *tol*,
    snap zmax down to that conductor's zmin (and symmetrically for zmin).
    """
    new_zmin, new_zmax = zmin, zmax
    for other_name, other in stack.layers.items():
        if other_name == via_name or other.layer_type != "conductor":
            continue
        # Via top extends slightly into the conductor above
        if zmin < other.zmin < zmax and (zmax - other.zmin) < tol:
            new_zmax = min(new_zmax, other.zmin)
        # Via bottom extends slightly into the conductor below
        if zmin < other.zmax < zmax and (other.zmax - zmin) < tol:
            new_zmin = max(new_zmin, other.zmax)
    if (new_zmin, new_zmax) != (zmin, zmax):
        logger.info(
            "Snapped via '%s' z-range [%.3f, %.3f] -> [%.3f, %.3f] "
            "to avoid sliver against adjacent conductor",
            via_name,
            zmin,
            zmax,
            new_zmin,
            new_zmax,
        )
    return new_zmin, new_zmax


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

    # Oversize -> union -> undersize
    buffered = [buffer(p, offset, join_style="mitre") for p in shapely_polys]
    merged = buffer(unary_union(buffered), -offset, join_style="mitre")

    # Convert back to coordinate lists
    result = []
    # unary_union may return Polygon or MultiPolygon
    geoms_attr = getattr(merged, "geoms", None)
    geoms = list(geoms_attr) if geoms_attr is not None else [merged]
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
            "Via merging: %d polygons -> %d (distance=%.1f um)",
            n_before,
            n_after,
            merge_distance,
        )

    return result


def _is_covered_by_dielectric_box(layer, stack: LayerStack) -> bool:
    """Check if a layer is already represented as a bulk box in stack.dielectrics.

    When a dielectric layer's material and z-range are covered by a
    ``stack.dielectrics`` entry, the polygon geometry on that layer is
    just a placeholder for a full-domain bulk region (e.g. vacuum or
    cladding) and should NOT be polygon-extruded as a shaped volume.
    """
    from gsim.common.stack.materials import MATERIAL_ALIASES

    layer_mat = MATERIAL_ALIASES.get(layer.material.lower(), layer.material.lower())
    for d in stack.dielectrics:
        d_mat_name = d.get("material", "")
        d_mat = MATERIAL_ALIASES.get(d_mat_name.lower(), d_mat_name.lower())
        if (
            d_mat == layer_mat
            and d["zmin"] <= layer.zmin + 1e-6
            and d["zmax"] >= layer.zmax - 1e-6
        ):
            return True
    return False


def _detect_shaped_dielectric_layers(
    geometry: GeometryData, stack: LayerStack
) -> set[str]:
    """Detect dielectric layers that should be polygon-extruded (shaped).

    A dielectric layer is shaped when it carries polygon geometry in the
    component AND is NOT already represented as a bulk box in
    ``stack.dielectrics``.  Bulk regions (vacuum, cladding) that have
    both a Layer entry and a dielectric box entry are treated as boxes.
    Waveguide cores only exist as Layer entries with polygon geometry —
    they have no corresponding dielectric box, so they must be
    polygon-extruded as shaped volumes.

    Returns:
        Set of layer names that should be treated as shaped dielectrics.
    """
    gds_layers_with_polys = {layernum for layernum, *_ in geometry.polygons}

    shaped: set[str] = set()
    for name, layer in stack.layers.items():
        if layer.layer_type != "dielectric":
            continue
        if layer.gds_layer[0] not in gds_layers_with_polys:
            continue
        if _is_covered_by_dielectric_box(layer, stack):
            continue
        shaped.add(name)

    return shaped


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
    merge_via_distance: float = 2.0,
) -> dict:
    """Add metal, via, and shaped-dielectric geometries to gmsh.

    Creates extruded volumes for vias and shells (surfaces) for conductors.
    Shaped dielectrics (auto-detected: dielectric layers with polygon
    geometry that are NOT already represented as bulk boxes in
    ``stack.dielectrics``) are extruded as 3D solid volumes, similar to
    vias, but are not hollowed out — they retain their full volume and
    carry dielectric permittivity in the Palace config.

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
        and shaped dielectrics, and (volumetag, surface_tags) tuples for
        conductors.
    """
    # layer_name -> {"volumes": [], "surfaces_xy": [], "surfaces_z": []}
    metal_tags: dict[str, dict[str, list]] = {}
    # Detect shaped-dielectric layers once (replaces thickness heuristic)
    shaped_dielectric_names = _detect_shaped_dielectric_layers(geometry, stack)

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
        is_shaped_dielectric = layer_name in shaped_dielectric_names

        if layer_type not in ("conductor", "via") and not is_shaped_dielectric:
            continue

        # Snap via z-range so it does not sliver into an adjacent conductor
        if layer_type == "via":
            zmin, zmax = _snap_via_z_range(stack, layer_name, zmin, zmin + thickness)
            thickness = zmax - zmin

        if layer_name not in metal_tags:
            metal_tags[layer_name] = {
                "volumes": [],
                "surfaces_xy": [],
                "surfaces_z": [],
            }

        if is_shaped_dielectric:
            shaped_dielectric_names.add(layer_name)

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

        min_volume_thickness = 0.05  # um — thinner volumes can't mesh as 3D
        is_planar = (
            planar_conductors or thickness == 0 or thickness < min_volume_thickness
        )

        if is_shaped_dielectric:
            # Shaped dielectric: extrude as solid 3D volume (like a via)
            # but keep the full volume (no shell extraction). The volume
            # carries dielectric permittivity in the Palace config.
            if thickness == 0 or thickness < min_volume_thickness:
                logger.warning(
                    "Shaped dielectric layer '%s' too thin for 3D meshing "
                    "(%.3f um < %.3f um), skipping shaped extrusion",
                    layer_name,
                    thickness,
                    min_volume_thickness,
                )
                continue
            # Fuse overlapping same-layer surfaces before extrusion
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

            logger.info(
                "Shaped dielectric layer '%s': 3D volume "
                "(material=%s, thickness=%.3f um)",
                layer_name,
                layer_info["material"],
                thickness,
            )
            for surfacetag in surfaces:
                result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
                volumetag = result[1][1]
                metal_tags[layer_name]["volumes"].append(volumetag)
        elif layer_type == "conductor" and is_planar:
            # Zero/thin-thickness or explicitly planar -> 2D PEC surface
            metal_tags[layer_name]["surfaces_xy"].extend(surfaces)
            # Also create explicit wire loops for mesh refinement.  Embedded
            # planar surfaces lose their boundary curves after boolean
            # fragmentation, so the conductor edges cannot drive refinement.
            # Adding independent line loops at the conductor z-height gives
            # gmsh explicit curves to refine around the metal perimeter.
            for pts_x, pts_y, holes in polys:
                loop_tag = gmsh_utils._create_wire_loop(  # noqa: SLF001
                    kernel, list(pts_x), list(pts_y), zmin
                )
                if loop_tag is not None:
                    metal_tags[layer_name].setdefault("refinement_lines", []).append(
                        loop_tag
                    )
                for hx, hy in holes:
                    hole_loop = gmsh_utils._create_wire_loop(  # noqa: SLF001
                        kernel, list(hx), list(hy), zmin
                    )
                    if hole_loop is not None:
                        metal_tags[layer_name].setdefault(
                            "refinement_lines", []
                        ).append(hole_loop)
        elif layer_type == "via":
            # Decide between 3D volume (with conductivity) and 2D PEC fallback
            material_name = layer_info["material"]
            mat_props = stack.materials.get(material_name, {})
            conductivity = mat_props.get("conductivity", 0.0)
            via_too_thin = thickness == 0 or thickness < min_volume_thickness

            if via_too_thin:
                logger.warning(
                    "Via layer '%s' too thin for 3D meshing "
                    "(%.3f um < %.3f um), falling back to 2D PEC surface",
                    layer_name,
                    thickness,
                    min_volume_thickness,
                )
                metal_tags[layer_name]["surfaces_xy"].extend(surfaces)
            elif conductivity <= 0:
                logger.warning(
                    "Via layer '%s' has no conductivity for material '%s', "
                    "falling back to 2D PEC surface",
                    layer_name,
                    material_name,
                )
                metal_tags[layer_name]["surfaces_xy"].extend(surfaces)
            else:
                # Extrude via as 3D volume with finite conductivity
                logger.info(
                    "Via layer '%s': 3D volume (material=%s, "
                    "\u03c3=%.2e S/m, thickness=%.3f um)",
                    layer_name,
                    material_name,
                    conductivity,
                    thickness,
                )
                for surfacetag in surfaces:
                    result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
                    volumetag = result[1][1]
                    metal_tags[layer_name]["volumes"].append(volumetag)
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

    # Record bounding boxes of BOTH via and conductor volumes BEFORE
    # removeAllDuplicates. That call renumbers ALL entity tags globally —
    # not just the ones it merges — so original tags are never trustworthy
    # afterwards. We re-identify volumes by bbox after the call.
    _via_bboxes: dict[str, list[tuple[float, ...]]] = {}
    _conductor_bboxes: dict[str, list[tuple[float, ...]]] = {}
    # Also record surface bboxes for planar conductors — these survive as
    # dielectric boundary faces after boolean and are used to find the
    # conductor perimeter curves for mesh refinement.
    _pec_surface_bboxes: dict[str, list[tuple[float, ...]]] = {}
    kernel.synchronize()
    for layer_name, tag_info in metal_tags.items():
        for vtag in tag_info["volumes"]:
            if isinstance(vtag, int):
                bbox = kernel.getBoundingBox(3, vtag)
                _via_bboxes.setdefault(layer_name, []).append(bbox)

        for stag in tag_info.get("surfaces_xy", []):
            try:
                bbox = kernel.getBoundingBox(2, stag)
                _pec_surface_bboxes.setdefault(layer_name, []).append(bbox)
            except Exception:
                logger.debug(
                    "Could not get bbox for planar conductor surface %d, skipping", stag
                )

    for layer_name, vol_tags in _conductor_volumes.items():
        for vtag in vol_tags:
            try:
                bbox = kernel.getBoundingBox(3, vtag)
                _conductor_bboxes.setdefault(layer_name, []).append(bbox)
            except Exception:
                logger.debug(
                    "Could not get bbox for conductor volume %d, skipping", vtag
                )

    kernel.removeAllDuplicates()
    kernel.synchronize()

    # After removeAllDuplicates(), some independent curve-loop entities
    # (created for planar-conductor refinement) may be merged away.
    # Refresh the refinement line tags so downstream consumers see only
    # valid curves.
    for layer_name, tag_info in metal_tags.items():
        if layer_name == "__shaped_dielectrics__":
            continue
        old_line_tags = tag_info.get("refinement_lines", [])
        if not old_line_tags:
            continue
        valid_lines = []
        for ltag in old_line_tags:
            try:
                kernel.getBoundingBox(1, ltag)
                valid_lines.append(ltag)
            except Exception:
                pass  # Curve was merged away
        # Try to find the merged successor by looking at all remaining curves
        # that have the same z-coordinate and bounding box.
        if len(valid_lines) < len(old_line_tags):
            # Some curves were merged — find replacements by matching bboxes.
            # getEntities(1) returns curves that survived dedup.
            all_curves = list(kernel.getEntities(1))
            all_curve_bboxes: dict[int, tuple] = {}
            for _, ctag in all_curves:
                with contextlib.suppress(Exception):
                    all_curve_bboxes[ctag] = kernel.getBoundingBox(1, ctag)
            for ltag in old_line_tags:
                if ltag in valid_lines:
                    continue
                try:
                    old_bbox = kernel.getBoundingBox(1, ltag)
                except Exception:
                    continue
                for ctag, bbox in all_curve_bboxes.items():
                    if ctag in valid_lines:
                        continue
                    if all(
                        abs(a - b) < 0.01 for a, b in zip(bbox, old_bbox, strict=True)
                    ):
                        valid_lines.append(ctag)
                        break
        tag_info["refinement_lines"] = sorted(set(valid_lines))

    # Build a single bbox lookup for all post-dedup volumes (avoids O(n²) calls).
    all_vols = kernel.getEntities(3)
    all_vol_bboxes: dict[int, tuple] = {}
    for _, vtag in all_vols:
        try:
            all_vol_bboxes[vtag] = kernel.getBoundingBox(3, vtag)
        except Exception:
            logger.debug("Could not get bbox for volume %d after dedup", vtag)

    # Re-identify via volumes by matching bounding boxes.
    for layer_name, bboxes in _via_bboxes.items():
        metal_tags[layer_name]["volumes"] = []
        for target_bbox in bboxes:
            for vtag, bbox in all_vol_bboxes.items():
                if all(
                    abs(a - b) < 0.01 for a, b in zip(bbox, target_bbox, strict=True)
                ):
                    metal_tags[layer_name]["volumes"].append(vtag)
                    break

    # Re-identify conductor volumes by bbox and update _conductor_volumes.
    # Without this, removeAllDuplicates()'s global renumbering makes every
    # original tag appear missing, so all conductors are silently dropped.
    for layer_name, bboxes in _conductor_bboxes.items():
        new_vol_tags = []
        for target_bbox in bboxes:
            for vtag, bbox in all_vol_bboxes.items():
                if all(
                    abs(a - b) < 0.01 for a, b in zip(bbox, target_bbox, strict=True)
                ):
                    new_vol_tags.append(vtag)
                    break
            else:
                logger.warning(
                    "Conductor volume on %s lost during dedup (no bbox match)",
                    layer_name,
                )
        _conductor_volumes[layer_name] = new_vol_tags

    # Extract shell surfaces from conductor volumes (now with correct post-dedup tags).
    current_vols = {t for _, t in all_vols}
    for layer_name, vol_tags in _conductor_volumes.items():
        for volumetag in vol_tags:
            if volumetag not in current_vols:
                logger.warning(
                    "Conductor volume %d on %s missing after bbox re-identification",
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

    # Store shaped-dielectric layer names for downstream classification.
    # Uses a reserved key that is skipped by consumers iterating per-layer.
    # TODO: smuggling a set[str] into metal_tags under a reserved key forces a
    # type-ignore here. Cleaner to return shaped_dielectric_names separately and
    # update the consumer in classify_* (see metal_tags.get("__shaped_dielectrics__")).
    metal_tags["__shaped_dielectrics__"] = shaped_dielectric_names  # ty: ignore[invalid-assignment]

    # Store pre-dedup PEC surface bboxes so the boolean pipeline can re-identify
    # merged planar-conductor surfaces by their geometry.
    if _pec_surface_bboxes:
        metal_tags["__pec_surface_bboxes__"] = _pec_surface_bboxes  # type: ignore[invalid-assignment]

    return metal_tags


def add_dielectrics(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    margin_x: float,
    margin_y: float | None = None,
    air_margin: float = 0.0,
    airbox_margin_x: float | None = None,
    airbox_margin_y: float | None = None,
    airbox_z_above: float | None = None,
    airbox_z_below: float | None = None,
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
        margin_x: X margin around design (um). Also used as Y margin
            when *margin_y* is not provided (backward compat).
        margin_y: Y margin around design (um). Defaults to margin_x.
        air_margin: Legacy isotropic extra margin for the surrounding
            airbox (um). Used as a fallback for all airbox directions.
        airbox_margin_x: Extra x-margin for the enclosing airbox (um).
            Falls back to *air_margin* when None.
        airbox_margin_y: Extra y-margin for the enclosing airbox (um).
            Falls back to *air_margin* when None.
        airbox_z_above: Extra +z margin for the enclosing airbox (um).
            Falls back to *air_margin* when None.
        airbox_z_below: Extra -z margin for the enclosing airbox (um).
            Falls back to *air_margin* when None.

    Returns:
        Dict with material_name -> list of volume_tags
    """
    if margin_y is None:
        margin_y = margin_x

    if airbox_margin_x is None:
        airbox_margin_x = air_margin
    if airbox_margin_y is None:
        airbox_margin_y = air_margin
    if airbox_z_above is None:
        airbox_z_above = air_margin
    if airbox_z_below is None:
        airbox_z_below = air_margin

    dielectric_tags: dict[str, list[int]] = {}

    xmin0, ymin0, xmax0, ymax0 = geometry.bbox
    xmin_air = xmin0 - margin_x
    ymin_air = ymin0 - margin_y
    xmax_air = xmax0 + margin_x
    ymax_air = ymax0 + margin_y

    def _contains_air_token(name: str | None) -> bool:
        """Return True when *name* clearly denotes air/vacuum."""
        if not name:
            return False

        normalized = name.strip().lower().replace("-", "_")
        if normalized in {"air", "vacuum"}:
            return True

        tokens = [tok for tok in normalized.split("_") if tok]
        return "air" in tokens or "vacuum" in tokens

    def _is_air_or_vacuum(
        material_name: str,
        dielectric_name: str | None = None,
    ) -> bool:
        """Return True when *material_name* represents air/vacuum.

        Uses stack material metadata (permittivity ~1) first, then
        falls back to name matching for robustness with custom stacks.
        """
        mat = stack.materials.get(material_name)
        if isinstance(mat, dict):
            mat_type = str(mat.get("type", "")).strip().lower()
            eps = mat.get("permittivity")
        else:
            mat_type = str(getattr(mat, "type", "")).strip().lower()
            eps = getattr(mat, "permittivity", None)

        if mat_type == "dielectric":
            try:
                if eps is not None and abs(float(eps) - 1.0) <= 1e-9:
                    return True
            except (TypeError, ValueError):
                pass

        if _contains_air_token(material_name):
            return True

        return _contains_air_token(dielectric_name)

    z_min_all = math.inf
    z_max_all = -math.inf

    use_airbox = any(
        m > 0.0
        for m in (
            airbox_margin_x,
            airbox_margin_y,
            airbox_z_above,
            airbox_z_below,
        )
    )

    for dielectric in stack.dielectrics:
        dielectric_name = dielectric.get("name")
        material = dielectric["material"]

        is_air_like = _is_air_or_vacuum(material, dielectric_name=dielectric_name)

        # When building an explicit airbox, skip explicit air/vacuum layers.
        if is_air_like and use_airbox:
            continue

        d_zmin = dielectric["zmin"]
        d_zmax = dielectric["zmax"]

        z_min_all = min(z_min_all, d_zmin)
        z_max_all = max(z_max_all, d_zmax)

        dielectric_tags.setdefault(material, [])

        xmin = xmin_air if is_air_like else xmin0
        ymin = ymin_air if is_air_like else ymin0
        xmax = xmax_air if is_air_like else xmax0
        ymax = ymax_air if is_air_like else ymax0

        # When shaped dielectrics exist, ALL non-air dielectric boxes must
        # extend to the air margins.  A shaped dielectric (e.g. waveguide
        # core) carves out of the surrounding oxide/substrate boxes, so those
        # boxes need to be large enough to fully surround the shaped volume.
        #
        # Also extend substrates (dielectrics starting at z ~ 0) to margins.
        # A bulk substrate like sapphire or silicon should fill the same
        # transverse extent as the surrounding air so the mesh domain is
        # consistent and the substrate edge does not artificially truncate
        # fields.
        is_bulk_substrate = d_zmin <= 1e-6
        if (
            is_bulk_substrate or _detect_shaped_dielectric_layers(geometry, stack)
        ) and not is_air_like:
            xmin = xmin_air
            ymin = ymin_air
            xmax = xmax_air
            ymax = ymax_air

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

    # Resolve stack z envelope even if dielectric list is sparse.
    if not (math.isfinite(z_min_all) and math.isfinite(z_max_all)):
        for layer in stack.layers.values():
            z_min_all = min(z_min_all, layer.zmin)
            z_max_all = max(z_max_all, layer.zmax)

    # Explicit single airbox (boolean pipeline handles overlap/subtraction).
    if use_airbox:
        if not (math.isfinite(z_min_all) and math.isfinite(z_max_all)):
            raise ValueError(
                "Cannot create airbox because stack z extents could not be resolved"
            )

        airbox_x = airbox_margin_x
        airbox_y = airbox_margin_y
        airbox_above = airbox_z_above
        airbox_below = airbox_z_below
        if (
            airbox_x is None
            or airbox_y is None
            or airbox_above is None
            or airbox_below is None
        ):
            raise ValueError("Explicit airbox margins must all be provided")

        airbox_tag = gmsh_utils.create_box(
            kernel,
            xmin_air - airbox_margin_x,
            ymin_air - airbox_margin_y,
            z_min_all - airbox_z_below,
            xmax_air + airbox_margin_x,
            ymax_air + airbox_margin_y,
            z_max_all + airbox_z_above,
        )
        dielectric_tags["airbox"] = [airbox_tag]

    kernel.synchronize()

    return dielectric_tags


def resolve_mesh_domain_bounds(
    geometry: GeometryData,
    stack: LayerStack,
    *,
    margin_x: float,
    margin_y: float | None = None,
    air_margin: float = 0.0,
    airbox_margin_x: float | None = None,
    airbox_margin_y: float | None = None,
    airbox_z_above: float | None = None,
    airbox_z_below: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """Resolve outer mesh-domain bounds (including explicit airbox when used)."""
    if margin_y is None:
        margin_y = margin_x

    if airbox_margin_x is None:
        airbox_margin_x = air_margin
    if airbox_margin_y is None:
        airbox_margin_y = air_margin
    if airbox_z_above is None:
        airbox_z_above = air_margin
    if airbox_z_below is None:
        airbox_z_below = air_margin

    xmin0, ymin0, xmax0, ymax0 = geometry.bbox
    xmin_air = xmin0 - margin_x
    ymin_air = ymin0 - margin_y
    xmax_air = xmax0 + margin_x
    ymax_air = ymax0 + margin_y

    # Robust stack z-envelope: include dielectric and layer extents.
    z_min_all = math.inf
    z_max_all = -math.inf
    for dielectric in stack.dielectrics:
        z_min_all = min(z_min_all, dielectric["zmin"])
        z_max_all = max(z_max_all, dielectric["zmax"])

    if not (math.isfinite(z_min_all) and math.isfinite(z_max_all)):
        z_try_min, z_try_max = stack.get_z_range()
        z_min_all = min(z_min_all, z_try_min)
        z_max_all = max(z_max_all, z_try_max)

    if stack.layers:
        z_min_layers = min(layer.zmin for layer in stack.layers.values())
        z_max_layers = max(layer.zmax for layer in stack.layers.values())
        z_min_all = min(z_min_all, z_min_layers)
        z_max_all = max(z_max_all, z_max_layers)

    if not (math.isfinite(z_min_all) and math.isfinite(z_max_all)):
        raise ValueError("Cannot resolve stack z extents for domain bounds")

    use_airbox = any(
        m > 0.0
        for m in (
            airbox_margin_x,
            airbox_margin_y,
            airbox_z_above,
            airbox_z_below,
        )
    )

    if use_airbox:
        return (
            xmin_air - airbox_margin_x,
            ymin_air - airbox_margin_y,
            z_min_all - airbox_z_below,
            xmax_air + airbox_margin_x,
            ymax_air + airbox_margin_y,
            z_max_all + airbox_z_above,
        )

    return (xmin_air, ymin_air, z_min_all, xmax_air, ymax_air, z_max_all)


def add_patterned_dielectrics(
    kernel,
    geometry: GeometryData,
    stack: LayerStack,
    min_volume_thickness: float = 0.05,
    curve_fit_mode: Literal["line", "spline", "bspline"] = "line",
    curve_fit_layers: list[str] | None = None,
    curve_fit_tolerance_um: float = 0.0,
    curve_fit_min_points: int = 8,
    curve_fit_corner_angle_deg: float = 45.0,
) -> dict[str, list[int]]:
    """Add patterned dielectric volumes from stack dielectric layers.

    This extrudes component polygons for stack layers classified as
    ``dielectric`` so patterned optical/routing cores become explicit
    3D dielectric regions in the boolean pipeline.

    Args:
        kernel: gmsh OCC kernel
        geometry: Extracted geometry data
        stack: LayerStack with layer definitions
        min_volume_thickness: Skip very thin dielectric layers that cannot
            be robustly meshed as 3D volumes.
        curve_fit_mode: Boundary curve mode for selected layers.
        curve_fit_layers: Layer names where spline/bspline fitting is allowed.
        curve_fit_tolerance_um: Point merge tolerance before curve fitting.
        curve_fit_min_points: Minimum contour points to attempt curve fitting.
        curve_fit_corner_angle_deg: Turn-angle threshold for corner detection
            during spline/bspline segmentation.

    Returns:
        Dict mapping dielectric layer name -> list of volume tags.
    """
    patterned_tags: dict[str, list[int]] = {}
    curve_layers = set(curve_fit_layers or [])

    # Shaped dielectric layers are already extruded by ``add_metals`` and
    # tracked via ``metal_tags["__shaped_dielectrics__"]``. Re-extruding them
    # here would create overlapping volumes with duplicate ``Entity`` names,
    # which collide in ``build_entities`` / ``assign_physical_groups`` and
    # cause the layer to drop out of ``groups["volumes"]``.
    shaped_dielectric_names = _detect_shaped_dielectric_layers(geometry, stack)

    polygons_by_layer: dict[int, list[tuple[list[float], list[float], list]]] = {}
    for layernum, pts_x, pts_y, holes in geometry.polygons:
        polygons_by_layer.setdefault(layernum, []).append((pts_x, pts_y, holes))

    for layernum, polys in polygons_by_layer.items():
        layer_info = get_layer_info(stack, layernum)
        if layer_info is None or layer_info["type"] != "dielectric":
            continue

        layer_name = layer_info["name"]
        if layer_name in shaped_dielectric_names:
            continue

        # Skip dielectric layers already covered by bulk dielectric boxes
        # from stack.dielectrics. Re-extruding them would create overlapping
        # volumes that get consumed in the boolean pipeline, causing the
        # intended bulk material groups to disappear.
        layer_obj = stack.layers.get(layer_name)
        if layer_obj is not None and _is_covered_by_dielectric_box(layer_obj, stack):
            continue

        zmin = layer_info["zmin"]
        thickness = layer_info["thickness"]

        if thickness <= 0 or thickness < min_volume_thickness:
            logger.debug(
                "Skipping patterned dielectric layer '%s' with thickness %.3f um",
                layer_name,
                thickness,
            )
            continue

        surfaces = []
        surface_loop_mode = (
            curve_fit_mode
            if curve_fit_mode != "line" and layer_name in curve_layers
            else "line"
        )
        for pts_x, pts_y, holes in polys:
            surfacetag = gmsh_utils.create_polygon_surface(
                kernel,
                pts_x,
                pts_y,
                zmin,
                holes=holes,
                loop_mode=surface_loop_mode,
                fit_tolerance_um=curve_fit_tolerance_um,
                min_points_for_curve_fit=curve_fit_min_points,
                corner_turn_threshold_deg=curve_fit_corner_angle_deg,
            )
            if surfacetag is not None:
                surfaces.append(surfacetag)

        if not surfaces:
            continue

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

        volumes = []
        for surfacetag in surfaces:
            result = kernel.extrude([(2, surfacetag)], 0, 0, thickness)
            volumes.append(result[1][1])

        if volumes:
            patterned_tags.setdefault(layer_name, []).extend(volumes)

    kernel.synchronize()
    return patterned_tags


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
    patterned_dielectric_tags: dict | None,
    port_tags: dict,
    port_info: list,
    pec_block_tags: dict | None = None,
    stack: LayerStack | None = None,
) -> list[gmsh_utils.Entity]:
    """Convert geometry tag dicts into Entity objects for the boolean pipeline.

    Mesh-order convention (lower = higher priority, gets cut first):
        0  - conductor (2D PEC) surfaces and PEC block surfaces
        1  - via volumes (3D, higher priority than dielectrics) and port surfaces
        2  - patterned dielectric volumes from stack layers
        3  - background dielectric boxes (non-airbox volumes)
        4  - airbox volume (lowest priority, carved by everything else)

    Args:
        metal_tags: from ``add_metals()``
        dielectric_tags: from ``add_dielectrics()``
        patterned_dielectric_tags: from ``add_patterned_dielectrics()``
        port_tags: from ``add_ports()``
        port_info: metadata list from ``add_ports()``
        pec_block_tags: from ``add_pec_blocks()``, optional
        stack: LayerStack for distinguishing via vs conductor vs shaped
            dielectric layers

    Returns:
        List of Entity objects ready for ``run_boolean_pipeline()``.
    """
    Entity = gmsh_utils.Entity
    entities: list[gmsh_utils.Entity] = []

    # Build set of via and shaped-dielectric layer names for quick lookup
    via_layers: set[str] = set()
    shaped_dielectric_layers: set[str] = set()
    if stack:
        via_layers = {
            n for n, layer in stack.layers.items() if layer.layer_type == "via"
        }
    _shaped_meta = metal_tags.get("__shaped_dielectrics__")
    if isinstance(_shaped_meta, set):
        shaped_dielectric_layers = _shaped_meta

    # --- Conductors, vias, and shaped dielectrics ---
    for layer_name, tag_info in metal_tags.items():
        if layer_name.startswith("__") and layer_name.endswith("__"):
            continue

        is_via = layer_name in via_layers
        is_shaped_dielectric = layer_name in shaped_dielectric_layers

        # PEC / zero-thickness surfaces
        if tag_info.get("surfaces_xy"):
            # Via PEC surfaces get higher priority (lower mesh_order) so they
            # are processed first and survive boolean cuts against conductor
            # shell surfaces that sit at the same z-height.
            pec_mesh_order = -1 if is_via else 0
            entities.append(
                Entity(
                    name=f"{layer_name}_pec",
                    dim=2,
                    mesh_order=pec_mesh_order,
                    tags=tag_info["surfaces_xy"],
                )
            )

        if tag_info.get("volumes"):
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
            elif is_shaped_dielectric:
                # Shaped dielectric volumes: 3D entities with same priority as
                # vias so they carve out of surrounding dielectric boxes.
                shaped_vol_tags = [
                    item for item in tag_info["volumes"] if isinstance(item, int)
                ]
                if shaped_vol_tags:
                    entities.append(
                        Entity(
                            name=layer_name,
                            dim=3,
                            mesh_order=1,
                            tags=shaped_vol_tags,
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
                        mesh_order=-1,
                        tags=[tag],
                    )
                )
        else:
            entities.append(
                Entity(
                    name=port_name,
                    dim=2,
                    mesh_order=-1,
                    tags=surf_tags,
                )
            )

    # --- Dielectric volumes (dim=3) ---
    if patterned_dielectric_tags:
        for layer_name, vol_tags in patterned_dielectric_tags.items():
            entities.append(
                Entity(
                    name=layer_name,
                    dim=3,
                    mesh_order=2,
                    tags=vol_tags,
                )
            )

    patterned_names = set(patterned_dielectric_tags or {})
    for material, vol_tags in dielectric_tags.items():
        # If a patterned dielectric entity already uses this name, prefer the
        # patterned volume entity to avoid name collisions in group assignment.
        entity_name = "air" if material == "airbox" else str(material)
        if entity_name in patterned_names:
            continue
        order = 4 if entity_name == "air" else 3
        entities.append(
            Entity(
                name=entity_name,
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
    domain_bbox: tuple[float, float, float, float] | None = None,
    domain_bounds: tuple[float, float, float, float, float, float] | None = None,
) -> tuple[dict, list]:
    """Add port surfaces to gmsh.

    Args:
        kernel: gmsh OCC kernel
        ports: List of PalacePort objects (single or multi-element)
        stack: Layer stack
        domain_bbox: (xmin, ymin, xmax, ymax) of the simulation domain
            (geometry bbox with margin applied). Required when any port
            has ``max_size=True``.
        domain_bounds: (xmin, ymin, zmin, xmax, ymax, zmax) of the outer
            simulation domain. When provided, ``max_size=True`` waveports
            are clipped to this exact 3D domain envelope.

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
                    "name": port.name,
                    "Z0": port.impedance,
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
                    "name": port.name,
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
                        "name": port.name,
                        "Z0": port.impedance,
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
                # Build a robust z-envelope for waveports. When synthetic
                # dielectric boxes are disabled, stack.get_z_range() can miss
                # patterned core levels; include layer extents as well.
                layer_zmin, layer_zmax = stack.get_z_range()
                if stack.layers:
                    zmin_layers = min(layer.zmin for layer in stack.layers.values())
                    zmax_layers = max(layer.zmax for layer in stack.layers.values())
                    layer_zmin = min(layer_zmin, zmin_layers)
                    layer_zmax = max(layer_zmax, zmax_layers)

                if port.max_size and domain_bounds is not None:
                    # Fill the full 3D simulation domain.
                    _, _, zmin, _, _, zmax = domain_bounds
                elif port.max_size:
                    # Backward-compatible fallback when 3D bounds are unavailable.
                    zmin = layer_zmin
                    zmax = layer_zmax
                else:
                    zmin = zmin - port.z_margin
                    zmax = zmax + port.z_margin
                    zmin = max(zmin, layer_zmin)
                    zmax = min(zmax, layer_zmax)

                # Guard against inverted/degenerate z extents.
                if zmax <= zmin:
                    z_center = target_layer.zmin + 0.5 * target_layer.thickness
                    z_half = max(port.z_margin, 0.01)
                    zmin = z_center - z_half
                    zmax = z_center + z_half

                angle = port.orientation % 360
                is_y_axis = 45 <= angle < 135 or 225 <= angle < 315

                if port.max_size:
                    if domain_bounds is not None:
                        dom_xmin, dom_ymin, _, dom_xmax, dom_ymax, _ = domain_bounds
                    elif domain_bbox is not None:
                        dom_xmin, dom_ymin, dom_xmax, dom_ymax = domain_bbox
                    else:
                        raise ValueError(
                            f"Port '{port.name}' has max_size=True but "
                            "domain bounds were not provided to add_ports()"
                        )
                    if is_y_axis:
                        xmin = dom_xmin
                        xmax = dom_xmax
                        ymin = y
                        ymax = y
                    else:
                        xmin = x
                        xmax = x
                        ymin = dom_ymin
                        ymax = dom_ymax
                elif is_y_axis:
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

                effective_width = xmax - xmin if is_y_axis else ymax - ymin

                port_info.append(
                    {
                        "portnumber": port_num,
                        "type": "waveport",
                        "width": effective_width,
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
