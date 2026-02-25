"""Visualization utilities for gsim.

This module provides visualization tools for meshes and simulation results.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal

import meshio
import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)


def plot_mesh(
    msh_path: str | Path,
    output: str | Path | None = None,
    show_groups: list[str] | None = None,
    interactive: bool = True,
    style: Literal["wireframe", "solid"] = "wireframe",
    transparent_groups: list[str] | None = None,
) -> None:
    """Plot mesh using PyVista.

    Args:
        msh_path: Path to .msh file
        output: Output PNG path (only used if interactive=False)
        show_groups: List of group name patterns to show (None = all).
            Example: ["metal", "P"] to show metal layers and ports.
        interactive: If True, open interactive 3D viewer.
            If False, save static PNG to output path.
        style: Visualization style.
            ``"wireframe"`` (default) – classic wireframe rendering.
            ``"solid"`` – solid faces coloured by physical group with
            selectable transparent groups.
        transparent_groups: Physical-group names rendered at low opacity
            when *style="solid"*.  Defaults to
            ``["air_none", "air_plastic_enclosure"]``.

    Example:
        >>> pa.plot_mesh("./sim/palace.msh", show_groups=["metal", "P"])
        >>> pa.plot_mesh("./sim/palace.msh", style="solid")
    """
    if style == "solid":
        view_mesh(
            mesh_filename=str(msh_path),
            transparent_groups=transparent_groups,
        )
        return
    msh_path = Path(msh_path)

    # Get group info from meshio
    mio = meshio.read(msh_path)
    group_map = {tag: name for name, (tag, _) in mio.field_data.items()}

    # Load mesh with pyvista
    mesh = pv.read(msh_path)

    if interactive:
        plotter = pv.Plotter(window_size=[1200, 900])
    else:
        plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])

    plotter.set_background("white")  # type: ignore[arg-type]

    if show_groups:
        # Filter to matching groups
        ids = [
            tag
            for tag, name in group_map.items()
            if any(p in name for p in show_groups)
        ]
        colors = ["red", "blue", "green", "orange", "purple", "cyan"]
        for i, gid in enumerate(ids):
            subset = mesh.extract_cells(mesh.cell_data["gmsh:physical"] == gid)
            if subset.n_cells > 0:
                plotter.add_mesh(
                    subset,
                    style="wireframe",
                    color=colors[i % len(colors)],
                    line_width=1,
                    label=group_map.get(gid, str(gid)),
                )
        plotter.add_legend()
    else:
        plotter.add_mesh(mesh, style="wireframe", color="black", line_width=1)

    plotter.camera_position = "iso"

    if interactive:
        plotter.show()
    else:
        if output is None:
            output = msh_path.with_suffix(".png")
        plotter.screenshot(str(output))
        plotter.close()
        # Display in notebook if available
        try:
            from IPython.display import Image, display

            display(Image(str(output)))
        except ImportError:
            logger.info("Saved mesh plot to %s", output)


def view_mesh(
    mesh_filename: str | Path = "straight_microstrip.msh",
    transparent_groups: list[str] | None = None,
) -> None:
    """View a mesh file using PyVista with solid faces coloured by physical group.

    Triangle surface cells are extracted and rendered with per-group colouring.
    Selected groups can be made transparent so internal structure is visible.

    Args:
        mesh_filename: Path to the mesh file.
        transparent_groups: Physical-group names to render with low opacity.
            Defaults to ``["air_none", "air_plastic_enclosure"]``.
    """
    if transparent_groups is None:
        transparent_groups = ["air_none", "air_plastic_enclosure"]

    logger.info("Loading mesh file: %s", mesh_filename)
    if transparent_groups:
        logger.info("Groups to render transparent: %s", transparent_groups)

    m = meshio.read(str(mesh_filename))
    logger.info("Mesh loaded successfully with %d cell blocks", len(m.cells))

    # ------------------------------------------------------------------
    # Collect all triangle cells and their physical-group tags
    # ------------------------------------------------------------------
    triangle_cells_list: list[np.ndarray] = []
    triangle_tags_list: list[np.ndarray] = []
    found_triangles = False

    for i, cell_block in enumerate(m.cells):
        if "triangle" in cell_block.type:
            found_triangles = True
            triangle_cells_list.append(cell_block.data)
            if i < len(m.cell_data.get("gmsh:physical", [])):
                triangle_tags_list.append(m.cell_data["gmsh:physical"][i])
            else:
                triangle_tags_list.append(
                    np.full(len(cell_block.data), -1, dtype=int)
                )

    # ------------------------------------------------------------------
    # Fallback: if no triangles, try tetrahedra wireframe
    # ------------------------------------------------------------------
    if not found_triangles:
        logger.info(
            "No triangle cells found – visualizing tetrahedra wireframe."
        )
        tetra_cells_list = [
            cb.data for cb in m.cells if "tetra" in cb.type
        ]
        if not tetra_cells_list:
            raise RuntimeError(
                "No triangle or tetrahedron cells found in the mesh."
            )
        tetra_cells = np.vstack(tetra_cells_list)
        grid = pv.UnstructuredGrid(
            {pv.CellType.TETRA: tetra_cells}, m.points
        )
        plotter = pv.Plotter()
        plotter.add_mesh(
            grid, style="wireframe", color="gray", show_edges=True
        )
        plotter.show_axes()
        plotter.show()
        return

    # ------------------------------------------------------------------
    # Build UnstructuredGrid from triangle surfaces
    # ------------------------------------------------------------------
    triangle_cells = np.vstack(triangle_cells_list)
    triangle_tags = np.concatenate(triangle_tags_list)
    logger.info("Found %d triangles", len(triangle_cells))

    n_tri = triangle_cells.shape[0]
    cells = np.hstack(
        [np.full((n_tri, 1), 3), triangle_cells]
    ).astype(np.int64).flatten()
    celltypes = np.full(n_tri, pv.CellType.TRIANGLE, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, celltypes, m.points)
    grid.cell_data["physical_group"] = triangle_tags

    # ------------------------------------------------------------------
    # Map tags → human-readable names
    # ------------------------------------------------------------------
    tag_to_name = {idx: name for name, (idx, _dim) in m.field_data.items()}
    unique_tags = np.unique(triangle_tags)
    group_names = [tag_to_name.get(tag, str(tag)) for tag in unique_tags]
    logger.info(
        "Physical group tags in mesh: %s", dict(zip(unique_tags, group_names))
    )

    triangle_names = np.array(
        [tag_to_name.get(int(tag), str(tag)) for tag in triangle_tags]
    )
    triangle_names_num = np.array(
        [
            f"{tag_to_name.get(int(tag), str(tag))} ({int(tag)})"
            for tag in triangle_tags
        ]
    )
    grid.cell_data["physical_group_name"] = triangle_names_num

    # ------------------------------------------------------------------
    # Render: opaque groups + transparent groups
    # ------------------------------------------------------------------
    plotter = pv.Plotter()

    transparent_mask = np.isin(triangle_names, transparent_groups)
    opaque_mask = ~transparent_mask

    if np.any(opaque_mask):
        opaque_grid = grid.extract_cells(np.where(opaque_mask)[0])
        plotter.add_mesh(
            opaque_grid,
            scalars="physical_group_name",
            show_edges=True,
            cmap="tab10",
            categories=True,
            opacity=1.0,
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Physical Group",
                "vertical": True,
                "position_x": 0.85,
                "position_y": 0.05,
                "width": 0.1,
                "height": 0.7,
                "title_font_size": 16,
                "label_font_size": 12,
            },
        )

    for group_name in transparent_groups:
        group_mask = triangle_names == group_name
        if not np.any(group_mask):
            continue
        group_grid = grid.extract_cells(np.where(group_mask)[0])

        if group_name == "air_boundary":
            color = "lightblue"
            edge_color = "blue"
        else:
            hash_val = int(
                hashlib.md5(group_name.encode()).hexdigest()[:6], 16
            )
            color = f"#{hash_val:06x}"
            edge_color = color

        plotter.add_mesh(
            group_grid,
            color=color,
            show_edges=True,
            opacity=0.2,
            edge_color=edge_color,
            line_width=0.5,
        )
        logger.info(
            "Added transparent group '%s' with color %s", group_name, color
        )

    plotter.show_axes()
    plotter.show()
