"""Mesh transfer from the shared native-2D pipeline to DEVSIM.

The BoundaryMode pipeline writes the cross-section mesh in um (msh v2.2).
DEVSIM's silicon physics parameters are cm-based, so the same mesh is
rewritten with coordinates scaled to cm before ``create_gmsh_mesh`` loads
it. Physical groups (region and contact names) are preserved verbatim —
this is a unit conversion, not a second meshing path.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
from numpy.typing import NDArray

from gsim.common.mesh_regions import cell_blocks, group_tags

#: Coordinate scale from the gsim mesh unit (um) to DEVSIM's cm.
UM_TO_CM: float = 1e-4


def write_scaled_msh(
    src: str | Path,
    dst: str | Path,
    *,
    scale: float = UM_TO_CM,
) -> Path:
    """Write a copy of a gmsh v2.2 mesh with scaled node coordinates.

    Args:
        src: Source mesh path (msh v2.2, coordinates in um).
        dst: Destination mesh path.
        scale: Multiplicative coordinate scale (default um -> cm).

    Returns:
        The destination path.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    mesh = meshio.read(str(src))
    mesh.points = mesh.points * scale
    meshio.write(str(dst), mesh, file_format="gmsh22", binary=False)
    return Path(dst)


def line_group_points(path: str | Path, group: str) -> NDArray[np.float64]:
    """Node coordinates of a named dim-1 physical group of a gmsh mesh.

    Args:
        path: Mesh path (msh v2.2).
        group: Physical-group name of a set of curves.

    Returns:
        The ``(n, 2)`` in-plane coordinates of the group's distinct nodes,
        in the mesh's own unit.

    Raises:
        ValueError: When the mesh holds no line group of that name, or no
            line cells at all.
    """
    mesh = meshio.read(str(path))
    tag = group_tags(mesh, dim=1).get(group)
    if tag is None:
        raise ValueError(f"The mesh holds no line group named '{group}'.")
    lines, tags = cell_blocks(mesh, "line")
    indices = np.unique(lines[tags == tag].ravel())
    return np.asarray(mesh.points[indices, :2], dtype=np.float64)


__all__ = ["UM_TO_CM", "line_group_points", "write_scaled_msh"]
