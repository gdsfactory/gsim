"""Reading a gmsh mesh's physical groups and cell blocks.

A gmsh mesh names its regions twice over: ``field_data`` carries the
``name -> (tag, dim)`` table, and every cell block carries the tag of the
group its cells belong to. Anything that wants to speak in region names —
which elements are the core, which carriers sit in the slab, which nodes
a contact owns — has to join the two, and the join is the same few lines
whatever the cells are.

This module is that join, written over an explicit dimension rather than
around triangles: :func:`group_tags` and :func:`group_names` are the two
directions of the ``field_data`` lookup (callers want both — one asks
"where is the core", the other "what is this element"), and
:func:`cell_blocks` concatenates the blocks of one cell type with their
tags, in meshio's own block order. On top of them sit the triangle
conveniences the 2D cross-section pipeline reaches for:
:func:`element_regions`, :func:`node_regions` and :func:`region_elements`.

Element order is meshio's block order throughout, which is the order an
element basis is built in, so an index from :func:`region_elements`
addresses the same element a solved Mode's per-element array does.

Pure meshio/numpy: no solver runtime is needed, and nothing here is in
``gsim.common.__all__`` — import it by module path.
"""

from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "cell_blocks",
    "element_regions",
    "group_names",
    "group_tags",
    "node_regions",
    "region_elements",
]


def group_tags(mesh: meshio.Mesh, *, dim: int) -> dict[str, int]:
    """Map the physical-group names of one dimension to their gmsh tags.

    Args:
        mesh: A loaded meshio mesh carrying gmsh ``field_data``.
        dim: Group dimension (1 for curves, 2 for surfaces, 3 for volumes).

    Returns:
        ``{group_name: tag}``, empty when the mesh names no group of that
        dimension.
    """
    return {
        str(name): int(np.asarray(data)[0])
        for name, data in mesh.field_data.items()
        if int(np.asarray(data)[1]) == dim
    }


def group_names(mesh: meshio.Mesh, *, dim: int) -> dict[int, str]:
    """Map the gmsh tags of one dimension to their physical-group names.

    The inverse of :func:`group_tags`: a caller holding a cell's tag asks
    this one, a caller holding a region's name asks that one.

    Args:
        mesh: A loaded meshio mesh carrying gmsh ``field_data``.
        dim: Group dimension (1 for curves, 2 for surfaces, 3 for volumes).

    Returns:
        ``{tag: group_name}``, empty when the mesh names no group of that
        dimension.
    """
    return {tag: name for name, tag in group_tags(mesh, dim=dim).items()}


def cell_blocks(
    mesh: meshio.Mesh, cell_type: str
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Connectivity and physical tags of every block of one cell type.

    meshio splits a gmsh mesh into one block per (cell type, physical
    group) pair. This concatenates the blocks of one type in meshio's
    order, so that a row index into the connectivity is the element index
    the rest of the pipeline uses, and pairs each row with its group tag.

    A block the mesh gives no physical tag reads tag ``0``, gmsh's "no
    physical group", so a mesh written without ``gmsh:physical`` still
    yields its cells — they name no region, and nothing matches a group.

    Args:
        mesh: A loaded meshio mesh.
        cell_type: meshio cell type, e.g. ``"triangle"`` or ``"line"``.

    Returns:
        ``(cells, tags)``: the ``(n, k)`` node indices of the cells and
        the ``(n,)`` gmsh physical tag of each.

    Raises:
        ValueError: When the mesh holds no cells of that type.
    """
    physical = list(mesh.cell_data.get("gmsh:physical", []))
    blocks: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    for index, block in enumerate(mesh.cells):
        if block.type != cell_type:
            continue
        data = np.asarray(block.data, dtype=np.int64)
        tags = (
            np.asarray(physical[index], dtype=np.int64)
            if index < len(physical)
            else np.zeros(data.shape[0], dtype=np.int64)
        )
        blocks.append((data, tags))
    if not blocks:
        raise ValueError(f"Mesh has no {cell_type} elements.")
    cells = np.asarray(np.vstack([data for data, _ in blocks]), dtype=np.int64)
    tags = np.asarray(np.concatenate([phys for _, phys in blocks]), dtype=np.int64)
    return cells, tags


def element_regions(mesh: meshio.Mesh) -> tuple[NDArray[np.int64], list[str]]:
    """Triangle connectivity and the group name of every triangle.

    Args:
        mesh: A loaded meshio mesh.

    Returns:
        ``(triangles, names)``: the ``(n, 3)`` node indices and the
        dim-2 group name of each triangle. A triangle whose tag names no
        group reads ``""``.

    Raises:
        ValueError: When the mesh has no triangle elements.
    """
    tris, tags = cell_blocks(mesh, "triangle")
    names = group_names(mesh, dim=2)
    return tris, [names.get(int(tag), "") for tag in tags]


def node_regions(mesh: meshio.Mesh) -> list[str]:
    """The group name every node inherits from the triangles around it.

    A node on a region boundary belongs to several regions; the
    lowest-numbered incident triangle wins, so the answer is one name per
    node and does not depend on how the caller iterates.

    Args:
        mesh: A loaded meshio mesh.

    Returns:
        One group name per mesh point, in point order. A point no triangle
        touches reads ``""``.

    Raises:
        ValueError: When the mesh has no triangle elements.
    """
    tris, element_names = element_regions(mesh)
    node_names = [""] * int(np.asarray(mesh.points).shape[0])
    for element in range(tris.shape[0] - 1, -1, -1):
        for node in tris[element]:
            node_names[int(node)] = element_names[element]
    return node_names


def region_elements(mesh: meshio.Mesh | str | Path, region: str) -> NDArray[np.int64]:
    """Indices of the triangles belonging to one 2D region.

    The indices are into the mesh's triangle order, which is the order an
    element basis is built in — so they address the same elements a solved
    Mode's per-element arrays do. That is what a power-current impedance
    needs to integrate the current over one conductor of a multi-conductor
    line.

    Args:
        mesh: The mesh (path or loaded meshio mesh).
        region: Name of the dim-2 physical group.

    Returns:
        The element indices, ascending.

    Raises:
        ValueError: When the mesh has no triangles, or no 2D group of
            that name.
    """
    if not isinstance(mesh, meshio.Mesh):
        mesh = meshio.read(str(mesh))
    tags_by_name = group_tags(mesh, dim=2)
    if region not in tags_by_name:
        raise ValueError(
            f"Region '{region}' not found on the mesh. "
            f"Available Regions: {sorted(tags_by_name)}"
        )
    _tris, tags = cell_blocks(mesh, "triangle")
    return np.asarray(np.flatnonzero(tags == tags_by_name[region]), dtype=np.int64)
