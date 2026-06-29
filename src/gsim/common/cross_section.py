"""Solver-agnostic cross-section extractors.

Given a gdsfactory component and a LayerStack, this module extracts axis-
aligned 2D slices from 3D layer extrusions.

- ``extract_xz_rectangles``: plane ``Y=y_cut`` (returns XZ rectangles)
- ``extract_yz_rectangles``: plane ``X=x_cut`` (returns YZ rectangles)
- ``extract_xy_polygons``: plane ``Z=z_cut`` (returns XY polygons)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack import LayerStack


@dataclass(frozen=True)
class Rect2D:
    """Axis-aligned rectangle in the XZ plane.

    Attributes:
        x0: Low X extent (um).
        x1: High X extent (um).
        zmin: Low Z extent (um).
        zmax: High Z extent (um).
        layer_name: Source layer name from the LayerStack.
        material: Material name from the LayerStack layer.
    """

    x0: float
    x1: float
    zmin: float
    zmax: float
    layer_name: str
    material: str


@dataclass(frozen=True)
class RectYZ2D:
    """Axis-aligned rectangle in the YZ plane.

    Attributes:
        y0: Low Y extent (um).
        y1: High Y extent (um).
        zmin: Low Z extent (um).
        zmax: High Z extent (um).
        layer_name: Source layer name from the LayerStack.
        material: Material name from the LayerStack layer.
    """

    y0: float
    y1: float
    zmin: float
    zmax: float
    layer_name: str
    material: str


@dataclass(frozen=True)
class PolygonXY2D:
    """Polygon in the XY plane at a fixed Z cut.

    Coordinates are in um and represented without repeating the closing point.
    """

    exterior: tuple[tuple[float, float], ...]
    holes: tuple[tuple[tuple[float, float], ...], ...]
    layer_name: str
    material: str


def extract_xz_rectangles(
    component: gf.Component,
    layer_stack: LayerStack,
    y_cut: float,
    *,
    eps: float = 1e-9,
) -> list[Rect2D]:
    """Slice ``component`` at ``Y=y_cut``; return one Rect2D per layer-interval.

    For each layer in ``layer_stack`` with a GDS layer tuple:

    1. Pull polygons for that GDS layer from the component.
    2. Intersect each polygon (with holes) with the horizontal line Y=y_cut
       using shapely.
    3. Union the resulting 1D X-intervals within that layer.
    4. Emit one Rect2D per interval at the layer's (zmin, zmax).

    Args:
        component: gdsfactory Component (may contain references).
        layer_stack: LayerStack describing which layers to extract.
        y_cut: Y coordinate of the cross-section (um).
        eps: Drop intervals shorter than this (um) -- filters out zero-length
            cuts that hit a polygon edge exactly.

    Returns:
        List of Rect2D in layer-stack order, unmerged across layers.
        Layers with no intersection are skipped.
    """
    return _extract_line_slice_rectangles(
        component=component,
        layer_stack=layer_stack,
        cut_axis="y",
        cut_value=y_cut,
        eps=eps,
    )


def extract_yz_rectangles(
    component: gf.Component,
    layer_stack: LayerStack,
    x_cut: float,
    *,
    eps: float = 1e-9,
) -> list[RectYZ2D]:
    """Slice ``component`` at ``X=x_cut``; return one RectYZ2D per interval."""
    return _extract_line_slice_rectangles(
        component=component,
        layer_stack=layer_stack,
        cut_axis="x",
        cut_value=x_cut,
        eps=eps,
    )


def extract_xy_polygons(
    component: gf.Component,
    layer_stack: LayerStack,
    z_cut: float,
    *,
    eps: float = 1e-9,
) -> list[PolygonXY2D]:
    """Slice ``component`` at ``Z=z_cut``; return XY polygons per intersecting layer."""
    dbu = getattr(getattr(component, "kcl", None), "dbu", 0.001)

    polygons: list[PolygonXY2D] = []

    for layer_name, layer in layer_stack.layers.items():
        if z_cut < layer.zmin - eps or z_cut > layer.zmax + eps:
            continue

        gds_layer = getattr(layer, "gds_layer", None)
        if gds_layer is None:
            continue
        gds_layer_tuple = (int(gds_layer[0]), int(gds_layer[1]))

        shapely_polys = _layer_shapely_polys(component, gds_layer_tuple, dbu)
        for poly in shapely_polys:
            if poly.is_empty or poly.area <= eps:
                continue

            exterior = tuple((float(x), float(y)) for x, y in poly.exterior.coords[:-1])
            holes = tuple(
                tuple((float(x), float(y)) for x, y in ring.coords[:-1])
                for ring in poly.interiors
            )
            if len(exterior) < 3:
                continue

            polygons.append(
                PolygonXY2D(
                    exterior=exterior,
                    holes=holes,
                    layer_name=layer_name,
                    material=layer.material,
                )
            )

    return polygons


def extract_plane_section(
    component: gf.Component,
    layer_stack: LayerStack,
    *,
    axis: Literal["x", "y", "z"],
    value: float,
    eps: float = 1e-9,
) -> list[Rect2D] | list[RectYZ2D] | list[PolygonXY2D]:
    """Dispatch to axis-specific extraction.

    Args:
        axis: Cross-section normal axis.
        value: Plane coordinate in um.
    """
    if axis == "x":
        return extract_yz_rectangles(
            component=component,
            layer_stack=layer_stack,
            x_cut=value,
            eps=eps,
        )
    if axis == "y":
        return extract_xz_rectangles(
            component=component,
            layer_stack=layer_stack,
            y_cut=value,
            eps=eps,
        )
    return extract_xy_polygons(
        component=component,
        layer_stack=layer_stack,
        z_cut=value,
        eps=eps,
    )


@overload
def _extract_line_slice_rectangles(
    component,
    layer_stack,
    cut_axis: Literal["y"],
    cut_value: float,
    eps: float,
) -> list[Rect2D]: ...


@overload
def _extract_line_slice_rectangles(
    component,
    layer_stack,
    cut_axis: Literal["x"],
    cut_value: float,
    eps: float,
) -> list[RectYZ2D]: ...


def _extract_line_slice_rectangles(
    component,
    layer_stack,
    cut_axis: Literal["x", "y"],
    cut_value: float,
    eps: float,
) -> list[Rect2D] | list[RectYZ2D]:
    """Shared implementation for X/Y cuts through layer polygons."""
    from shapely.geometry import LineString
    from shapely.ops import unary_union

    dbu = getattr(getattr(component, "kcl", None), "dbu", 0.001)

    if cut_axis == "y":
        rects_xz: list[Rect2D] = []
        for layer_name, layer in layer_stack.layers.items():
            gds_layer = getattr(layer, "gds_layer", None)
            if gds_layer is None:
                continue
            gds_layer_tuple = (int(gds_layer[0]), int(gds_layer[1]))

            shapely_polys = _layer_shapely_polys(component, gds_layer_tuple, dbu)
            if not shapely_polys:
                continue

            merged = unary_union(shapely_polys)
            if merged.is_empty:
                continue

            minx, miny, maxx, maxy = merged.bounds
            if cut_value < miny - eps or cut_value > maxy + eps:
                continue

            cut_line = LineString([(minx - 1.0, cut_value), (maxx + 1.0, cut_value)])
            intersection = merged.intersection(cut_line)
            intervals = _line_intervals(intersection, axis_index=0)

            for x0, x1 in intervals:
                if x1 - x0 <= eps:
                    continue
                rects_xz.append(
                    Rect2D(
                        x0=x0,
                        x1=x1,
                        zmin=layer.zmin,
                        zmax=layer.zmax,
                        layer_name=layer_name,
                        material=layer.material,
                    )
                )

        return rects_xz

    rects_yz: list[RectYZ2D] = []
    for layer_name, layer in layer_stack.layers.items():
        gds_layer = getattr(layer, "gds_layer", None)
        if gds_layer is None:
            continue
        gds_layer_tuple = (int(gds_layer[0]), int(gds_layer[1]))

        shapely_polys = _layer_shapely_polys(component, gds_layer_tuple, dbu)
        if not shapely_polys:
            continue

        merged = unary_union(shapely_polys)
        if merged.is_empty:
            continue

        minx, miny, maxx, maxy = merged.bounds
        if cut_value < minx - eps or cut_value > maxx + eps:
            continue

        cut_line = LineString([(cut_value, miny - 1.0), (cut_value, maxy + 1.0)])
        intersection = merged.intersection(cut_line)
        intervals = _line_intervals(intersection, axis_index=1)

        for y0, y1 in intervals:
            if y1 - y0 <= eps:
                continue
            rects_yz.append(
                RectYZ2D(
                    y0=y0,
                    y1=y1,
                    zmin=layer.zmin,
                    zmax=layer.zmax,
                    layer_name=layer_name,
                    material=layer.material,
                )
            )

    return rects_yz


def _layer_shapely_polys(component, gds_layer_tuple, dbu):
    """Return a list of shapely Polygons for one GDS layer of ``component``."""
    from shapely.geometry import Polygon

    # ``merge=True`` mutates the component and fails for locked @cell instances.
    # We perform geometric union downstream, so request raw polygons here.
    raw = component.get_polygons(layers=(gds_layer_tuple,), merge=False)
    if not isinstance(raw, dict) or not raw:
        return []

    polys: list = []
    for value in raw.values():
        items = list(value) if isinstance(value, list) else [value]
        for obj in items:
            exterior, holes = _poly_to_coords(obj, dbu)
            if exterior is None or len(exterior) < 3:
                continue
            try:
                poly = Polygon(exterior, holes=holes)
            except (ValueError, TypeError) as err:
                logger.debug("Skipping invalid polygon: %s", err)
                continue
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if hasattr(poly, "geoms"):
                polys.extend(poly.geoms)
            else:
                polys.append(poly)
    return polys


def _poly_to_coords(obj, dbu):
    """Return (exterior_coords, list_of_hole_coords) from a polygon-ish object.

    Handles KLayout ``PolygonWithProperties`` objects (via ``each_point_hull``
    / ``each_point_hole``) and legacy numpy-array polygons (no holes).
    """
    # KLayout polygon with optional holes.
    if hasattr(obj, "each_point_hull"):
        exterior = [(pt.x * dbu, pt.y * dbu) for pt in obj.each_point_hull()]
        holes: list = []
        try:
            n_holes = obj.holes()
        except AttributeError:
            n_holes = 0
        for i in range(n_holes):
            try:
                holes.append(
                    [(pt.x * dbu, pt.y * dbu) for pt in obj.each_point_hole(i)]
                )
            except (AttributeError, IndexError) as err:
                logger.debug("Skipping malformed hole %d: %s", i, err)
                continue
        return exterior, holes

    # Legacy numpy / iterable of (x, y) points.
    if hasattr(obj, "__iter__"):
        try:
            return [(float(p[0]), float(p[1])) for p in obj], []
        except (TypeError, IndexError):
            return None, []
    return None, []


def _line_intervals(
    intersection,
    *,
    axis_index: Literal[0, 1] = 0,
) -> list[tuple[float, float]]:
    """Extract sorted, merged intervals from a shapely line intersection.

    Handles LineString, MultiLineString, empty, and GeometryCollection results.
    """
    from shapely.geometry import LineString, MultiLineString

    if intersection.is_empty:
        return []

    lines: list = []
    if isinstance(intersection, LineString):
        lines = [intersection]
    elif isinstance(intersection, MultiLineString):
        lines = list(intersection.geoms)
    else:
        for geom in getattr(intersection, "geoms", []):
            if isinstance(geom, LineString):
                lines.append(geom)

    intervals: list[tuple[float, float]] = []
    for line in lines:
        values = [coord[axis_index] for coord in line.coords]
        intervals.append((min(values), max(values)))

    intervals.sort()

    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [(a, b) for a, b in merged]


__all__ = [
    "PolygonXY2D",
    "Rect2D",
    "RectYZ2D",
    "extract_plane_section",
    "extract_xy_polygons",
    "extract_xz_rectangles",
    "extract_yz_rectangles",
]
