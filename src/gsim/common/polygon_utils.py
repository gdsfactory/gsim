"""Polygon simplification utilities for KLayout ↔ Shapely workflows.

Provides helpers for reducing polygon vertex counts (decimation) used
when converting QPDK etch layers to conductor geometry. Works with
KLayout ``kdb.Polygon`` objects via Shapely's Douglas-Peucker
simplification.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import gdsfactory as gf
import klayout.db as kdb
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# KLayout ↔ Shapely conversion
# ---------------------------------------------------------------------------


def klayout_to_shapely(polygon_kdb: kdb.Polygon) -> Polygon:
    """Convert a KLayout polygon (with optional holes) to a Shapely Polygon."""
    exterior_coords = [
        (gf.kcl.to_um(pt.x), gf.kcl.to_um(pt.y))
        for pt in polygon_kdb.each_point_hull()
    ]
    holes = []
    for hole_idx in range(polygon_kdb.holes()):
        hole_coords = [
            (gf.kcl.to_um(pt.x), gf.kcl.to_um(pt.y))
            for pt in polygon_kdb.each_point_hole(hole_idx)
        ]
        holes.append(hole_coords)

    return Polygon(exterior_coords, holes) if holes else Polygon(exterior_coords)


def shapely_to_klayout(shapely_poly: Polygon) -> kdb.Polygon | None:
    """Convert a Shapely Polygon back to a KLayout polygon.

    Returns ``None`` if the polygon is empty, invalid, or has fewer than
    3 exterior vertices.
    """
    try:
        if shapely_poly.is_empty or not shapely_poly.is_valid:
            return None
        if shapely_poly.geom_type != "Polygon":
            return None

        exterior_coords = list(shapely_poly.exterior.coords[:-1])
        if len(exterior_coords) < 3:
            return None

        exterior_points = [
            kdb.Point(int(gf.kcl.to_dbu(x)), int(gf.kcl.to_dbu(y)))
            for x, y in exterior_coords
        ]
        polygon = kdb.Polygon(kdb.SimplePolygon(exterior_points))

        for hole_ring in shapely_poly.interiors:
            hole_coords = list(hole_ring.coords[:-1])
            if len(hole_coords) >= 3:
                hole_points = [
                    kdb.Point(int(gf.kcl.to_dbu(x)), int(gf.kcl.to_dbu(y)))
                    for x, y in hole_coords
                ]
                polygon.insert_hole(hole_points)

        return polygon
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Shapely → KLayout conversion failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Simplification helpers
# ---------------------------------------------------------------------------


def _adaptive_tolerance(polygon_kdb: kdb.Polygon, relative: float = 0.01) -> float:
    """Calculate an adaptive simplification tolerance based on polygon size.

    Uses the smaller bounding-box dimension scaled by *relative*.  Complex
    polygons (many vertices) get a slightly more aggressive tolerance.
    """
    bbox = polygon_kdb.bbox()
    w = gf.kcl.to_um(bbox.width())
    h = gf.kcl.to_um(bbox.height())
    min_dim = min(w, h)

    n = polygon_kdb.num_points_hull()
    if n > 1000:
        scale = 1.5
    elif n > 100:
        scale = 1.0
    elif n > 20:
        scale = 0.5
    else:
        scale = 0.1

    tol = min_dim * relative * scale
    return max(0.001, min(tol, min_dim * 0.1))


def simplify_polygon(
    polygon_kdb: kdb.Polygon,
    tolerance: float = 1.0,
) -> kdb.Polygon:
    """Simplify a single KLayout polygon via Shapely's Douglas-Peucker.

    Args:
        polygon_kdb: Source polygon.
        tolerance: Simplification tolerance in µm.

    Returns:
        Simplified polygon (or a copy of the original on failure).
    """
    try:
        shapely_poly = klayout_to_shapely(polygon_kdb)
        simplified: BaseGeometry = shapely_poly.simplify(tolerance, preserve_topology=True)
        result = shapely_to_klayout(simplified)  # type: ignore[arg-type]
        return result if result is not None else polygon_kdb.dup()
    except Exception:  # noqa: BLE001
        return polygon_kdb.dup()


def decimate(
    polygons: list[kdb.Polygon],
    relative_tolerance: float = 0.001,
    *,
    verbose: bool = False,
) -> list[kdb.Polygon]:
    """Reduce vertex count of a list of KLayout polygons.

    Only polygons with more than 20 hull vertices are simplified; simpler
    shapes are kept as-is.

    Args:
        polygons: Input polygon list.
        relative_tolerance: Fraction of polygon size used as tolerance.
        verbose: Print per-polygon reduction statistics.

    Returns:
        List of (possibly simplified) polygons.
    """
    out: list[kdb.Polygon] = []
    total_before = total_after = 0

    for i, poly in enumerate(polygons):
        n = poly.num_points_hull()
        total_before += n

        if n > 20:
            tol = _adaptive_tolerance(poly, relative_tolerance)
            reduced = simplify_polygon(poly, tolerance=tol)
            rn = reduced.num_points_hull()
            if verbose and rn != n:
                print(
                    f"  Polygon {i}: {n} → {rn} pts "
                    f"(tol={tol:.3f} µm, {n / rn:.1f}× reduction)"
                )
            out.append(reduced)
            total_after += rn
        else:
            out.append(poly)
            total_after += n

    if verbose:
        pct = (total_before - total_after) / max(total_before, 1) * 100
        print(f"✓ Decimation: {total_before} → {total_after} pts ({pct:.1f}% removed)")

    return out


# ---------------------------------------------------------------------------
# Layer inspection / visualisation
# ---------------------------------------------------------------------------


def inspect_layers(
    component: gf.Component,
    filename: str | Path = "klayout_polygons.png",
    *,
    dpi: int = 400,
) -> None:
    """Plot every layer of *component* to a PNG file for quick inspection.

    Each layer gets its own subplot with all polygons drawn via Shapely.

    Args:
        component: gdsfactory Component to inspect.
        filename: Output image path.
        dpi: Image resolution.
    """
    import matplotlib.pyplot as plt

    polys_by_layer = component.get_polygons()
    if not polys_by_layer:
        raise ValueError("Component has no polygons")

    # Build layer-index → name mapping
    layer_names: dict[int, str] = {}
    try:
        layout = component.kcl.layout
        for info in layout.layer_infos():
            idx = layout.layer(info)
            if info.name:
                layer_names[idx] = info.name
    except Exception:  # noqa: BLE001
        pass

    n = len(polys_by_layer)
    fig, axes = plt.subplots(n, 1, figsize=(12, 6 * n))
    if n == 1:
        axes = [axes]

    for ax, (layer_num, polygons) in zip(axes, polys_by_layer.items()):
        label = layer_names.get(layer_num, f"Layer {layer_num}")
        ax.set_title(f"{label} ({len(polygons)} polygons)", fontsize=14, fontweight="bold")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        for kpoly in polygons:
            try:
                spoly = klayout_to_shapely(kpoly)
                xs, ys = spoly.exterior.xy
                ax.fill(xs, ys, alpha=0.4)
                ax.plot(xs, ys, linewidth=0.5)
            except Exception:  # noqa: BLE001
                pass

    plt.tight_layout()
    plt.savefig(str(filename), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Layer plot saved to {filename}")
