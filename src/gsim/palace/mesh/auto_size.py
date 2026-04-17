"""Auto-sizing heuristics for mesh generation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import klayout.db as kdb

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)


def _conductor_gds_tuples(stack: LayerStack) -> set[tuple[int, int]]:
    """Return the set of ``(layer, datatype)`` tuples for conductor layers."""
    return {
        tuple(layer.gds_layer)
        for layer in stack.layers.values()
        if layer.layer_type == "conductor"
    }


def _conductor_polygons_by_layer(
    component, stack: LayerStack
) -> dict[tuple[int, int], list[kdb.Polygon]]:
    """Group conductor polygons by their ``(layer, datatype)`` GDS tuple."""
    conductor_gds = _conductor_gds_tuples(stack)
    if not conductor_gds:
        return {}

    layout = component.kcl.layout
    index_to_gds: dict[int, tuple[int, int]] = {}
    for layer_index in range(layout.layers()):
        if layout.is_valid_layer(layer_index):
            info = layout.get_info(layer_index)
            index_to_gds[layer_index] = (info.layer, info.datatype)

    grouped: dict[tuple[int, int], list[kdb.Polygon]] = {}
    polygons_by_index = component.get_polygons()
    for layer_index, polys in polygons_by_index.items():
        gds_tuple = index_to_gds.get(layer_index)
        if gds_tuple is None or gds_tuple not in conductor_gds:
            continue
        grouped.setdefault(gds_tuple, []).extend(polys)
    return grouped


def _min_bbox_dim(component, stack: LayerStack) -> float | None:
    """Smallest bbox width/height across all conductor polygons (um)."""
    min_dim = math.inf
    for polys in _conductor_polygons_by_layer(component, stack).values():
        for poly in polys:
            points = list(poly.each_point_hull())
            if len(points) < 3:
                continue
            xs = [pt.x / 1000.0 for pt in points]
            ys = [pt.y / 1000.0 for pt in points]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)
            if w > 0:
                min_dim = min(min_dim, w)
            if h > 0:
                min_dim = min(min_dim, h)
    return min_dim if math.isfinite(min_dim) else None


def min_conductor_gap(
    component,
    stack: LayerStack,
    max_gap_um: float = 500.0,
) -> float | None:
    """Smallest edge-to-edge gap between conductor polygons (um).

    Uses klayout's ``kdb.Region.space_check`` with the Euclidean metric to
    find the minimum distance between distinct polygons on the same
    conductor layer. Field concentration happens in these gaps, so mesh
    refinement should resolve them.

    Note:
        Only same-layer gaps are considered. Cross-layer gap measurement
        is significantly more complex (it would require projecting
        polygons onto a common plane and accounting for z-separation)
        and is not attempted here; for the CPW pathology this targets
        (signal + coplanar grounds), same-layer is sufficient.

    Args:
        component: gdsfactory Component whose polygons are inspected.
        stack: Layer stack (used to identify conductor layers).
        max_gap_um: Upper bound on gaps to detect, in um. Gaps larger
            than this are ignored. A generous default (500 um) covers
            any realistic on-chip feature.

    Returns:
        Minimum same-layer conductor gap in um, or ``None`` when no
        conductor layer has a detectable gap (e.g. a single polygon,
        or no conductors declared).
    """
    grouped = _conductor_polygons_by_layer(component, stack)
    if not grouped:
        return None

    dbu = component.kcl.layout.dbu  # um per integer unit
    threshold_dbu = round(max_gap_um / dbu)

    min_gap = math.inf
    for polys in grouped.values():
        if len(polys) < 2:
            continue
        region = kdb.Region()
        for poly in polys:
            region.insert(poly)
        pairs = region.space_check(
            threshold_dbu,
            False,  # whole_edges
            kdb.Region.Euclidian,
        )
        for ep in pairs:
            d_um = ep.distance() * dbu
            if d_um > 0:
                min_gap = min(min_gap, d_um)

    return min_gap if math.isfinite(min_gap) else None


def min_conductor_feature_size(component, stack: LayerStack) -> float | None:
    """Smallest conductor feature to resolve, in um.

    Returns the minimum of:

    * the smallest polygon bbox width / height across all conductor
      layers, and
    * the smallest edge-to-edge gap between same-layer conductor
      polygons (see :func:`min_conductor_gap`).

    The gap is just as important as polygon width for mesh refinement:
    in a CPW, the 15 um gap between signal and ground is the real
    field-concentration site even when the trace itself is wider.

    Returns None if the component has no polygons on conductor layers,
    or the stack declares no conductors.
    """
    bbox_min = _min_bbox_dim(component, stack)
    gap_min = min_conductor_gap(component, stack)

    candidates = [v for v in (bbox_min, gap_min) if v is not None]
    if not candidates:
        return None
    return min(candidates)


def auto_refined_mesh_size(
    component,
    stack: LayerStack,
    preset_size: float,
    cells_per_feature: int = 4,
) -> float:
    """Pick ``refined_mesh_size`` scaled to the smallest conductor feature.

    "Smallest conductor feature" is the minimum of polygon bbox dimension
    and inter-polygon gap on each conductor layer — whichever is tighter.

    Returns ``min(preset_size, min_feature / cells_per_feature)`` so designs
    with small features get proportionally refined meshes while large designs
    keep the preset's size. Falls back to ``preset_size`` when no conductor
    polygons are found.
    """
    min_feature = min_conductor_feature_size(component, stack)
    if min_feature is None or cells_per_feature <= 0:
        return preset_size
    return min(preset_size, min_feature / cells_per_feature)


__all__ = [
    "auto_refined_mesh_size",
    "min_conductor_feature_size",
    "min_conductor_gap",
]
