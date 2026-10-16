"""Reading a gmsh mesh's physical groups and cell blocks.

A gmsh mesh names its regions twice over: ``field_data`` carries the
``name -> (tag, dim)`` table, and every cell block carries the tag of the
group its cells belong to. Anything that wants to speak in region names —
which nodes a contact owns, which elements a region holds — has to join
the two, and the join is the same few lines whatever the cells are.

:func:`group_tags` is the ``field_data`` lookup over an explicit
dimension, and :func:`cell_blocks` concatenates the blocks of one cell
type with their tags, in meshio's own block order.

Pure meshio/numpy: no solver runtime is needed, and nothing here is in
``gsim.common.__all__`` — import it by module path.
"""

from __future__ import annotations

import meshio
import numpy as np
from numpy.typing import NDArray

__all__ = ["cell_blocks", "group_tags"]


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
