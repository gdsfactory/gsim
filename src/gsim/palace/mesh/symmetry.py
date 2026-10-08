"""Pure helpers for symmetry planes (no Gmsh).

All lengths are in um unless a name says ``_nm``. The plane is described by a
:class:`~gsim.palace.models.symmetry.SymmetryPlaneConfig`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import klayout.db as kdb

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gsim.palace.models.symmetry import SymmetryPlaneConfig

logger = logging.getLogger(__name__)

IntervalClass = Literal["kept", "removed", "straddles", "touches", "in_plane"]

DEFAULT_TOL = 1e-6


def classify_interval(
    lo: float,
    hi: float,
    plane: SymmetryPlaneConfig,
    tol: float = DEFAULT_TOL,
) -> IntervalClass:
    """Classify the interval ``[lo, hi]`` along the plane axis.

    Returns:
        ``"in_plane"`` for a zero-width interval on the plane, ``"straddles"``
        when it extends to both sides, ``"touches"`` when it lies on the kept
        side with one end on the plane, ``"kept"`` / ``"removed"`` when it lies
        strictly on one side or, for ``"removed"``, ends on the plane from the
        removed side.
    """
    pos = plane.position
    if hi - lo <= tol and abs(lo - pos) <= tol:
        return "in_plane"

    above = hi > pos + tol
    below = lo < pos - tol
    if above and below:
        return "straddles"

    on_positive_side = above
    if on_positive_side == (plane.keep == "positive"):
        touching = abs(lo - pos) <= tol if above else abs(hi - pos) <= tol
        return "touches" if touching else "kept"
    return "removed"


def clamp_extent(
    lo: float, hi: float, plane: SymmetryPlaneConfig
) -> tuple[float, float]:
    """Cut the interval ``[lo, hi]`` at the plane, keeping the kept side."""
    if plane.keep == "positive":
        return max(lo, plane.position), hi
    return lo, min(hi, plane.position)


def clamp_xy_bounds(
    bounds: tuple[float, float, float, float], plane: SymmetryPlaneConfig
) -> tuple[float, float, float, float]:
    """Cut an ``(xmin, ymin, xmax, ymax)`` box at the plane."""
    xmin, ymin, xmax, ymax = bounds
    if plane.axis == "x":
        xmin, xmax = clamp_extent(xmin, xmax, plane)
    else:
        ymin, ymax = clamp_extent(ymin, ymax, plane)
    return xmin, ymin, xmax, ymax


def check_plane_inside_domain(
    dom_min: float,
    dom_max: float,
    plane: SymmetryPlaneConfig,
    tol: float = DEFAULT_TOL,
) -> None:
    """Raise ``ValueError`` unless the plane is strictly inside the domain."""
    if not dom_min + tol < plane.position < dom_max - tol:
        raise ValueError(
            f"Symmetry plane {plane.axis}={plane.position} lies outside the "
            f"simulation domain [{dom_min}, {dom_max}] along {plane.axis}"
        )


def _to_nm(value_um: float) -> int:
    """Convert um to integer database units (nm)."""
    return round(value_um * 1000.0)


def _half_box(
    bbox: kdb.Box, plane: SymmetryPlaneConfig, side: Literal["kept", "removed"]
) -> kdb.Box:
    """Box covering one side of the plane, spanning ``bbox`` plus one unit."""
    pos = _to_nm(plane.position)
    positive = (side == "kept") == (plane.keep == "positive")
    if plane.axis == "x":
        lo, hi = (pos, bbox.right + 1) if positive else (bbox.left - 1, pos)
        return kdb.Box(lo, bbox.bottom - 1, hi, bbox.top + 1)
    lo, hi = (pos, bbox.top + 1) if positive else (bbox.bottom - 1, pos)
    return kdb.Box(bbox.left - 1, lo, bbox.right + 1, hi)


def clip_region(region: kdb.Region, plane: SymmetryPlaneConfig) -> kdb.Region:
    """Keep the part of a region on the kept side of the plane."""
    if region.is_empty():
        return kdb.Region()
    box = _half_box(region.bbox(), plane, "kept")
    if box.empty():
        return kdb.Region()
    # merged() turns the keyhole cuts left by boolean ops back into real holes
    return (region & kdb.Region(box)).merged()


def clip_polygon(polygon: kdb.Polygon, plane: SymmetryPlaneConfig) -> list[kdb.Polygon]:
    """Clip one polygon (holes included) to the kept side of the plane.

    Clipping a single polygon can split it into several pieces, or remove it
    entirely (empty list).
    """
    return list(clip_region(kdb.Region(polygon), plane).each())


def mirror_region(region: kdb.Region, plane: SymmetryPlaneConfig) -> kdb.Region:
    """Mirror a region about the plane."""
    shift = 2 * _to_nm(plane.position)
    if plane.axis == "y":
        return region.transformed(kdb.Trans.M0).moved(0, shift)
    return region.transformed(kdb.Trans.M90).moved(shift, 0)


def asymmetric_layers(
    regions: Mapping[object, kdb.Region], plane: SymmetryPlaneConfig
) -> list[object]:
    """Return the keys of layers that are not mirror-symmetric about the plane.

    A layer with content on only one side is asymmetric. Differences up to one
    database unit are allowed.
    """
    bad: list[object] = []
    for key, region in regions.items():
        if region.is_empty():
            continue
        diff = (region ^ mirror_region(region, plane)).sized(-1)
        if not diff.is_empty():
            bad.append(key)
    return bad
