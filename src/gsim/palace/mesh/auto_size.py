"""Auto-sizing heuristics for mesh generation."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import klayout.db as kdb

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack

logger = logging.getLogger(__name__)

# One-grid notches/slivers can be rounding artifacts, not meshing targets.
_MIN_FEATURE_DBU = 2.0


def _conductor_gds_tuples(stack: LayerStack) -> set[tuple[int, int]]:
    """Return the set of ``(layer, datatype)`` tuples for conductor layers."""
    return {
        (int(layer.gds_layer[0]), int(layer.gds_layer[1]))
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
    for layer_index in layout.layer_indexes():
        if layout.is_valid_layer(layer_index):
            info = layout.get_info(layer_index)
            index_to_gds[layer_index] = (info.layer, info.datatype)

    grouped: dict[tuple[int, int], list[kdb.Polygon]] = {}
    polygons_by_index = component.get_polygons(layers=list(layout.layer_indexes()))
    for layer_index, polys in polygons_by_index.items():
        gds_tuple = index_to_gds.get(layer_index)
        if gds_tuple is None or gds_tuple not in conductor_gds:
            continue
        grouped.setdefault(gds_tuple, []).extend(polys)
    return grouped


def _conductor_regions(component, stack: LayerStack) -> list[kdb.Region]:
    """Merge touching and overlapping conductor polygons within each layer."""
    regions = [
        kdb.Region(polygons).merged()
        for polygons in _conductor_polygons_by_layer(component, stack).values()
    ]
    return [region for region in regions if not region.is_empty()]


def _min_resolved_distance(pairs: kdb.EdgePairs) -> float:
    """Smallest distance spanning at least two layout database units."""
    distances = (pair.distance() for pair in pairs)
    return min(
        (distance for distance in distances if distance >= _MIN_FEATURE_DBU),
        default=math.inf,
    )


def _min_width(regions: list[kdb.Region]) -> float:
    """Find narrow traces/necks using opposing edges; return a size in dbu."""
    # Bounding boxes provide a fallback for shapes without parallel edges and
    # bound the search. Merging first avoids measuring the width of input tiles.
    bbox_widths = (
        min(polygon.bbox().width(), polygon.bbox().height())
        for region in regions
        for polygon in region.each()
    )
    width = min(
        (size for size in bbox_widths if size >= _MIN_FEATURE_DBU), default=math.inf
    )
    if not math.isfinite(width):
        return math.inf
    threshold = math.ceil(width) + 1
    for region in regions:
        pairs = region.width_check(
            threshold,
            False,
            kdb.Region.Projection,
            1.0,  # Nearly parallel edges; ignore acute-corner zero-width artifacts.
        )
        width = min(width, _min_resolved_distance(pairs))
    return width


def _min_gap(regions: list[kdb.Region], threshold: int) -> float:
    """Check both inter-polygon gaps and notches/holes in a single polygon."""
    if threshold <= 0:
        return math.inf
    return min(
        (
            _min_resolved_distance(
                region.space_check(threshold, False, kdb.Region.Euclidian)
            )
            for region in regions
        ),
        default=math.inf,
    )


def min_conductor_gap(
    component,
    stack: LayerStack,
    max_gap_um: float = 500.0,
) -> float | None:
    """Smallest resolved same-layer conductor gap, notch, or hole width (um).

    Merge touching/overlapping polygons before checking spacing. This includes
    opposing edges of one connected conductor, as well as separate conductors.
    Distances below two layout database units are ignored as grid artifacts.
    Cross-layer spacing is not measured.

    Args:
        component: gdsfactory Component whose polygons are inspected.
        stack: Layer stack used to identify conductor layers.
        max_gap_um: Gaps at or above this search limit are ignored (um).

    Returns:
        Minimum detected spacing in um, or None if there is no detectable gap.
    """
    regions = _conductor_regions(component, stack)
    dbu = component.kcl.layout.dbu
    gap = _min_gap(regions, round(max_gap_um / dbu))
    return gap * dbu if math.isfinite(gap) else None


def min_conductor_feature_size(component, stack: LayerStack) -> float | None:
    """Estimate the smallest same-layer conductor feature to resolve (um).

    Use opposing, nearly parallel edges for trace/neck widths, plus spacing
    checks for gaps, notches, and holes. Merged polygon bounding boxes provide
    a fallback. Distances below two database units are ignored. Only gaps below
    500 um are searched; cross-layer spacing and arbitrary tapered/curved widths
    are not measured. This is a sizing heuristic, not an accuracy guarantee.

    Returns None if no conductor polygons are found.
    """
    regions = _conductor_regions(component, stack)
    dbu = component.kcl.layout.dbu
    width = _min_width(regions)
    # Gaps at or above the width cannot lower the result. A broad space_check
    # can enumerate millions of irrelevant edge pairs on perforated ground.
    gap_limit = (
        min(round(500.0 / dbu), math.ceil(width))
        if math.isfinite(width)
        else round(500.0 / dbu)
    )
    feature = min(width, _min_gap(regions, gap_limit))
    return feature * dbu if math.isfinite(feature) else None


def auto_refined_mesh_size(
    component,
    stack: LayerStack,
    preset_size: float,
    cells_per_feature: int = 2,
) -> float:
    """Scale a preset to the smallest detected conductor width or spacing.

    Return ``min(preset_size, min_feature / cells_per_feature)``. Fall back to
    the preset when no feature is found or cells_per_feature is nonpositive.
    The feature detector is a same-layer heuristic; convergence requires
    comparing simulation results across mesh refinements.
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
