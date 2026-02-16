"""Low-level mesh construction helpers for 3D rendering.

These operate on generic Prism objects and produce numpy vertex/face arrays
(or PyVista/Open3D meshes) without any solver-specific code.
"""

from __future__ import annotations

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
