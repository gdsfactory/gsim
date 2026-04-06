"""Visualization utilities for gsim.

This module provides visualization tools for meshes and simulation results.
"""

from __future__ import annotations

__all__ = ["plot_mesh", "plot_mesh_threejs"]

import hashlib
import logging
from pathlib import Path
from typing import Literal

import meshio
import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Re-export Three.js viewer
# ---------------------------------------------------------------------------

from gsim.common.viz.render3d_threejs import plot_mesh_threejs  # noqa: E402

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_mesh(
    msh_path: str | Path,
    output: str | Path | None = None,
    show_groups: list[str] | None = None,
    interactive: bool = True,
    style: Literal["wireframe", "solid"] = "wireframe",
    transparent_groups: list[str] | None = None,
) -> None:
    """Plot a ``.msh`` mesh using PyVista.

    Two rendering styles are available:

    * **wireframe** (default) — edges only, one colour per group when
      *show_groups* is given; black otherwise.
    * **solid** — coloured surfaces per physical group with a legend
      bar.  Groups listed in *transparent_groups* are drawn with low
      opacity so the interior structure remains visible.

    Args:
        msh_path: Path to ``.msh`` file.
        output: Output PNG path (only used when ``interactive=False``).
        show_groups: Group-name patterns to display (``None`` → all).
            Example: ``["metal", "P"]`` to show metal layers and ports.
        interactive: If ``True``, open an interactive 3-D viewer.
            If ``False``, save a static PNG to *output*.
        style: ``"wireframe"`` or ``"solid"``.
        transparent_groups: Group names rendered at low opacity in
            *solid* mode.  Ignored in *wireframe* mode.

    Example:
        >>> pa.plot_mesh("./sim/palace.msh", show_groups=["metal", "P"])
        >>> pa.plot_mesh(
        ...     "sim.msh", style="solid", transparent_groups=["Absorbing_boundary"]
        ... )
    """
    msh_path = Path(msh_path)

    if style == "solid":
        _plot_solid(
            msh_path,
            output=output,
            interactive=interactive,
            transparent_groups=transparent_groups or [],
        )
    else:
        _plot_wireframe(
            msh_path,
            output=output,
            show_groups=show_groups,
            interactive=interactive,
        )


# ---------------------------------------------------------------------------
# Wireframe renderer (original)
# ---------------------------------------------------------------------------


def _plot_wireframe(
    msh_path: Path,
    *,
    output: str | Path | None,
    show_groups: list[str] | None,
    interactive: bool,
) -> None:
    """Wireframe renderer — one colour per matched group."""
    mio = meshio.read(msh_path)
    group_map: dict[int, str] = {tag: name for name, (tag, _) in mio.field_data.items()}

    mesh = pv.read(msh_path)
    plotter = _make_plotter(interactive)

    if show_groups:
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
        if ids:
            plotter.add_legend()  # type: ignore[call-arg]
    else:
        plotter.add_mesh(mesh, style="wireframe", color="black", line_width=1)  # type: ignore[arg-type]

    _finish(plotter, msh_path, output=output, interactive=interactive)


# ---------------------------------------------------------------------------
# Solid renderer (coloured surfaces per physical group)
# ---------------------------------------------------------------------------

_TRANSPARENT_DEFAULTS = ("air_boundary", "air_none", "air_plastic_enclosure")


def _plot_solid(
    msh_path: Path,
    *,
    output: str | Path | None,
    interactive: bool,
    transparent_groups: list[str],
) -> None:
    """Solid renderer — coloured surfaces per physical group."""
    mio = meshio.read(msh_path)
    tag_to_name: dict[int, str] = {
        tag: name for name, (tag, _) in mio.field_data.items()
    }

    # Collect triangle cells and their physical tags ----------------------
    tri_cells: list[np.ndarray] = []
    tri_tags: list[np.ndarray] = []

    phys = mio.cell_data.get("gmsh:physical", [])
    for idx, cb in enumerate(mio.cells):
        if "triangle" not in cb.type:
            continue
        tri_cells.append(cb.data)
        if idx < len(phys):
            tri_tags.append(phys[idx])
        else:
            tri_tags.append(np.full(len(cb.data), -1, dtype=int))

    if not tri_cells:
        logger.warning("No triangle cells — falling back to wireframe.")
        _plot_wireframe(
            msh_path, output=output, show_groups=None, interactive=interactive
        )
        return

    all_cells = np.vstack(tri_cells)
    all_tags = np.concatenate(tri_tags)

    # Build an UnstructuredGrid -------------------------------------------
    n = all_cells.shape[0]
    pv_cells = np.hstack([np.full((n, 1), 3), all_cells]).astype(np.int64).ravel()
    celltypes = np.full(n, pv.CellType.TRIANGLE, dtype=np.uint8)
    grid = pv.UnstructuredGrid(pv_cells, celltypes, mio.points)

    # Annotate each cell with "<name> (<tag>)"
    names = np.array(
        [f"{tag_to_name.get(int(t), str(int(t)))} ({int(t)})" for t in all_tags]
    )
    grid.cell_data["physical_group_name"] = names

    # Plain names (without tag number) for masking
    plain_names = np.array([tag_to_name.get(int(t), str(int(t))) for t in all_tags])

    # Determine which groups should be transparent
    if not transparent_groups:
        transparent_groups = [n for n in _TRANSPARENT_DEFAULTS if n in plain_names]

    transparent_mask = np.isin(plain_names, transparent_groups)
    opaque_mask = ~transparent_mask

    plotter = _make_plotter(interactive)

    # Opaque surfaces with categorical colour map -------------------------
    if np.any(opaque_mask):
        opaque_grid = pv.UnstructuredGrid(grid.extract_cells(np.where(opaque_mask)[0]))
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

    # Transparent surfaces ------------------------------------------------
    for group_name in transparent_groups:
        group_mask = plain_names == group_name
        if not np.any(group_mask):
            continue
        group_grid = pv.UnstructuredGrid(grid.extract_cells(np.where(group_mask)[0]))
        color = _color_for_group(group_name)
        plotter.add_mesh(
            group_grid,
            color=color,
            show_edges=True,
            opacity=0.2,
            edge_color=color,
            line_width=0.5,
        )
        logger.info("Transparent group '%s' (colour %s)", group_name, color)

    _finish(plotter, msh_path, output=output, interactive=interactive)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plotter(interactive: bool) -> pv.Plotter:
    """Create a PyVista plotter with standard window settings."""
    if interactive:
        plotter = pv.Plotter(window_size=[1200, 900])
    else:
        plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.set_background("white")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    return plotter


def _finish(
    plotter: pv.Plotter,
    msh_path: Path,
    *,
    output: str | Path | None,
    interactive: bool,
) -> None:
    """Show or screenshot the plotter and clean up."""
    plotter.camera_position = "iso"
    plotter.show_axes()  # type: ignore[call-arg]
    if interactive:
        plotter.show()
    else:
        if output is None:
            output = msh_path.with_suffix(".png")
        plotter.screenshot(str(output))
        plotter.close()
        try:
            from IPython.display import Image, display

            display(Image(str(output)))
        except ImportError:
            logger.info("Saved mesh plot to %s", output)


def _color_for_group(name: str) -> str:
    """Deterministic colour for a group name."""
    if name == "air_boundary":
        return "lightblue"
    h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
    return f"#{h:06x}"
