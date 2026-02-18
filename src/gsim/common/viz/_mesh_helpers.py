"""Low-level mesh construction helpers for 3D rendering.

These operate on generic Prism objects and produce numpy vertex/face arrays
(or PyVista/Open3D meshes) without any solver-specific code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gsim.common.geometry_model import GeometryModel, Prism

# ---------------------------------------------------------------------------
# Generic vertex helpers
# ---------------------------------------------------------------------------


def prism_base_top_vertices(
    prism: Prism,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (base_vertices, top_vertices) as (N, 3) numpy arrays."""
    xy = prism.vertices  # (N, 2)
    n = len(xy)
    base = np.column_stack([xy, np.full(n, prism.z_base)])
    top = np.column_stack([xy, np.full(n, prism.z_top)])
    return base, top


# ---------------------------------------------------------------------------
# Simulation-box corner/line helpers
# ---------------------------------------------------------------------------


def simulation_box_corners(
    geometry_model: GeometryModel,
) -> tuple[np.ndarray, list[list[int]]]:
    """Return 8 corners and 12 edge-index pairs for the simulation bbox.

    Returns:
        (points, lines) where points is (8, 3) and lines is a list of
        [i, j] index pairs.
    """
    mn = np.array(geometry_model.bbox[0])
    mx = np.array(geometry_model.bbox[1])

    points = np.array(
        [
            mn,
            [mx[0], mn[1], mn[2]],
            [mx[0], mx[1], mn[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mx[0], mn[1], mx[2]],
            mx,
            [mn[0], mx[1], mx[2]],
        ]
    )

    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],  # bottom
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],  # top
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # verticals
    ]

    return points, lines


# ---------------------------------------------------------------------------
# Shared Delaunay triangulation for polygons with holes / concave polygons
# ---------------------------------------------------------------------------


def triangulate_polygon_with_holes(
    shapely_polygon: Any,
    z_base: float,
    z_top: float,
) -> tuple[np.ndarray, list[np.ndarray], list[list[int]]] | None:
    """Triangulate a Shapely polygon (possibly with holes) via Delaunay.

    Returns:
        ``(verts_3d, valid_triangles, boundary_segments)`` where
        *verts_3d* is (2*N, 3) — bottom then top copies of the 2D points,
        *valid_triangles* is a list of index-triplet arrays (into the first
        N points), and *boundary_segments* is a list of ``[i, j]`` pairs.
        Returns ``None`` when triangulation fails.
    """
    try:
        import shapely.geometry as sg
        from scipy.spatial import Delaunay
    except ImportError:
        return None

    all_points: list[tuple[float, float]] = []
    boundary_segments: list[list[int]] = []

    exterior_coords = list(shapely_polygon.exterior.coords[:-1])
    start_idx = 0
    all_points.extend(exterior_coords)
    boundary_segments = [
        [start_idx + i, start_idx + (i + 1) % len(exterior_coords)]
        for i in range(len(exterior_coords))
    ]

    for interior in shapely_polygon.interiors:
        interior_coords = list(interior.coords[:-1])
        start_idx = len(all_points)
        all_points.extend(interior_coords)
        boundary_segments.extend(
            [start_idx + i, start_idx + (i + 1) % len(interior_coords)]
            for i in range(len(interior_coords))
        )

    if len(all_points) < 3:
        return None

    points_2d = np.array(all_points)
    tri = Delaunay(points_2d)

    valid_triangles = [
        simplex
        for simplex in tri.simplices
        if shapely_polygon.contains(sg.Point(*np.mean(points_2d[simplex], axis=0)))
    ]

    if not valid_triangles:
        return None

    verts_3d = np.array(
        [[pt[0], pt[1], z_base] for pt in points_2d]
        + [[pt[0], pt[1], z_top] for pt in points_2d]
    )

    return verts_3d, valid_triangles, boundary_segments


def collect_triangular_prism_geometry(
    prisms: list[Prism],
) -> tuple[np.ndarray, list[int]] | None:
    """Collect vertices from many triangular prisms for batch meshing.

    Returns:
        ``(combined_vertices, prism_offsets)`` where *combined_vertices*
        is (M, 3) and *prism_offsets[i]* is the vertex offset of the
        *i*-th prism in that array.  Returns ``None`` if *prisms* is empty.
    """
    all_vertices: list[np.ndarray] = []
    prism_offsets: list[int] = []
    offset = 0

    for prism in prisms:
        base, top = prism_base_top_vertices(prism)
        verts = np.vstack([base, top])
        all_vertices.append(verts)
        prism_offsets.append(offset)
        offset += len(verts)

    if not all_vertices:
        return None

    combined = np.vstack(all_vertices)
    return combined, prism_offsets
