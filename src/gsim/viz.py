"""Visualization utilities for gsim.

This module provides visualization tools for meshes and simulation results.
"""

from __future__ import annotations

__all__ = ["plot_mesh"]

import logging
from pathlib import Path

import meshio
import pyvista as pv

logger = logging.getLogger(__name__)


def plot_mesh(
    msh_path: str | Path,
    output: str | Path | None = None,
    show_groups: list[str] | None = None,
    interactive: bool = True,
) -> None:
    """Plot mesh wireframe using PyVista.

    Args:
        msh_path: Path to .msh file
        output: Output PNG path (only used if interactive=False)
        show_groups: List of group name patterns to show (None = all).
            Example: ["metal", "P"] to show metal layers and ports.
        interactive: If True, open interactive 3D viewer.
            If False, save static PNG to output path.

    Example:
        >>> pa.plot_mesh("./sim/palace.msh", show_groups=["metal", "P"])
    """
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

    # Determine which groups to display
    if show_groups:
        ids = [
            tag
            for tag, name in group_map.items()
            if any(p in name for p in show_groups)
        ]
    else:
        # Show all groups
        ids = list(group_map.keys())

    # Find the largest volume group to render transparently
    largest_vol_id = None
    largest_vol_count = 0
    if not show_groups:
        for gid in ids:
            subset = mesh.extract_cells(mesh.cell_data["gmsh:physical"] == gid)
            if subset.n_cells > largest_vol_count:
                largest_vol_count = subset.n_cells
                largest_vol_id = gid

    colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    color_idx = 0
    for gid in ids:
        subset = mesh.extract_cells(mesh.cell_data["gmsh:physical"] == gid)
        if subset.n_cells == 0:
            continue
        name = group_map.get(gid, str(gid))
        color = colors[color_idx % len(colors)]
        color_idx += 1

        if not show_groups and gid == largest_vol_id:
            # Render the largest volume (e.g. cladding) as transparent
            # wireframe so inner structures are visible
            plotter.add_mesh(
                subset,
                style="wireframe",
                color=color,
                opacity=0.15,
                line_width=1,
                label=name,
            )
        else:
            plotter.add_mesh(
                subset,
                style="wireframe",
                color=color,
                line_width=1,
                label=name,
            )
    plotter.add_legend()

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
