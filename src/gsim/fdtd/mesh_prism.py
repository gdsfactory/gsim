"""Construct holed FDTD prisms without planar boolean subtraction."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from math import isclose
from typing import Any

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from gsim.palace.mesh.gmsh_utils import extrude_polygon as _boolean_extrude_polygon

logger = logging.getLogger(__name__)


def _wire(
    kernel: Any,
    coordinates: Sequence[tuple[float, float]],
    z: float,
    meshseed: float,
    created: list[tuple[int, int]],
) -> int:
    """Build one closed wire while tracking entities owned by this attempt."""
    points = []
    for x, y in coordinates:
        tag = kernel.addPoint(x, y, z, meshseed)
        created.append((0, tag))
        points.append(tag)
    lines = []
    for index, start in enumerate(points):
        tag = kernel.addLine(start, points[(index + 1) % len(points)])
        created.append((1, tag))
        lines.append(tag)
    return kernel.addCurveLoop(lines)


def _remove_attempt(kernel: Any, created: list[tuple[int, int]]) -> None:
    """Remove only entities created by an unsuccessful direct-face attempt."""
    present = set(kernel.getEntities())
    owned = sorted(present.intersection(created), reverse=True)
    if owned:
        kernel.remove(owned, recursive=True)


def _check_face(kernel: Any, surface: int, polygon: Polygon) -> None:
    """Check both boundary count and area before accepting the holed face."""
    loops, _ = kernel.getCurveLoops(surface)
    if len(loops) != len(polygon.interiors) + 1:
        raise ValueError("Direct planar face does not preserve polygon holes")
    if not isclose(
        kernel.getMass(2, surface), polygon.area, rel_tol=1e-9, abs_tol=1e-6
    ):
        raise ValueError("Direct planar face does not preserve polygon area")


def _direct_surface(kernel: Any, polygon: Polygon, z: float, meshseed: float) -> int:
    """Create and check a planar face with native inner boundary wires."""
    polygon = orient(polygon, sign=1.0)
    created: list[tuple[int, int]] = []
    try:
        # OCC applies the interior-wire orientation itself. Supply each wire
        # counterclockwise, rather than passing Shapely's clockwise holes.
        loops = [
            _wire(kernel, list(polygon.exterior.coords)[:-1], z, meshseed, created)
        ]
        loops.extend(
            _wire(kernel, list(ring.coords)[:-1][::-1], z, meshseed, created)
            for ring in polygon.interiors
        )
        surface = kernel.addPlaneSurface(loops)
        created.append((2, surface))
        _check_face(kernel, surface, polygon)
    except Exception:
        _remove_attempt(kernel, created)
        raise
    return surface


def extrude_polygon(
    kernel: Any,
    pts_x: list[float],
    pts_y: list[float],
    zmin: float,
    thickness: float,
    meshseed: float = 0,
    holes: list[tuple[list[float], list[float]]] | None = None,
) -> int | None:
    """Extrude native holed faces, retaining the established boolean fallback.

    Coordinates and dimensions use the caller's model units. Ordinary polygons
    retain the existing construction. Direct holed faces must preserve the
    independently computed polygon area before extrusion.
    """
    if not holes:
        return _boolean_extrude_polygon(kernel, pts_x, pts_y, zmin, thickness, meshseed)
    polygon = Polygon(
        list(zip(pts_x, pts_y, strict=True)),
        holes=[list(zip(x, y, strict=True)) for x, y in holes],
    )
    try:
        surface = _direct_surface(kernel, polygon, zmin, meshseed)
    except Exception:
        logger.debug("Falling back to boolean polygon-hole construction", exc_info=True)
        return _boolean_extrude_polygon(
            kernel, pts_x, pts_y, zmin, thickness, meshseed, holes
        )
    if thickness <= 0:
        return surface
    result = kernel.extrude([(2, surface)], 0, 0, thickness)
    volumes = [tag for dimension, tag in result if dimension == 3]
    if len(volumes) != 1:
        raise ValueError("A polygon prism must produce exactly one volume")
    return volumes[0]
