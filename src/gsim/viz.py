"""Visualization utilities for gsim.

This module provides visualization tools for meshes and simulation results.
"""

from __future__ import annotations

__all__ = ["plot_cross_section", "plot_mesh"]

import hashlib
import logging
from pathlib import Path
from typing import Literal

import meshio
import numpy as np
import pyvista as pv

logger = logging.getLogger(__name__)


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
            plotter.add_legend()
    else:
        plotter.add_mesh(mesh, style="wireframe", color="black", line_width=1)

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
    plotter.show_axes()
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


# ---------------------------------------------------------------------------
# Cross-section field plot
# ---------------------------------------------------------------------------


def plot_cross_section(
    vol: pv.DataSet,
    *,
    normal: Literal["x", "y", "z"] = "x",
    origin: float = 0.0,
    field: str = "E_real",
    title: str | None = None,
    label: str | None = None,
    zi_range: tuple[float, float] | None = None,
    yi_range: tuple[float, float] | None = None,
    log: bool = False,
    quiver: bool = True,
    figsize: tuple[float, float] = (12, 5),
    cmap: str = "turbo",
    grid_resolution: tuple[int, int] = (200, 100),
) -> None:
    """Plot a 2-D cross-section of a vector field from a Palace volume.

    Slices *vol* along the given *normal* axis at *origin*, interpolates
    the field magnitude onto a regular grid, and overlays quiver arrows
    showing the in-plane field direction.

    This is the reusable version of ``plot_cross_section`` originally
    defined in ``palace_demo_cpw_fields.ipynb``.

    Args:
        vol: PyVista volume dataset (e.g. from ``pv.read("data.pvtu")``).
        normal: Axis perpendicular to the slice (``"x"``, ``"y"``, ``"z"``).
        origin: Position along *normal* where the slice is taken (µm).
        field: Name of a 3-component vector field in ``vol.point_data``.
        title: Plot title.  Defaults to ``"|{field}| cross-section"``.
        label: Colour-bar label.  Defaults to ``"|{field}|"``.
        zi_range: ``(zmin, zmax)`` limits for the vertical axis.
            ``None`` auto-detects from data ± padding.
        yi_range: ``(ymin, ymax)`` limits for the horizontal axis.
            ``None`` auto-detects from data ± padding.
        log: Use logarithmic colour scale.
        quiver: Overlay quiver arrows for in-plane direction.
        figsize: Matplotlib figure size.
        cmap: Colour-map name.
        grid_resolution: ``(n_horiz, n_vert)`` interpolation grid points.

    Example::

        import pyvista as pv
        from gsim.viz import plot_cross_section

        vol = pv.read("output/palace/.../data.pvtu")
        plot_cross_section(vol, normal="x", origin=-400)
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from scipy.interpolate import griddata

    # --- slice the volume ------------------------------------------------
    axis_idx = {"x": 0, "y": 1, "z": 2}[normal]
    origin_pt = [0.0, 0.0, 0.0]
    origin_pt[axis_idx] = origin
    sliced = vol.slice(normal=normal, origin=tuple(origin_pt))

    if sliced.n_points == 0:
        logger.warning("Slice at %s=%s returned 0 points.", normal, origin)
        return

    if field not in sliced.point_data:
        available = list(sliced.point_data.keys())
        msg = f"Field '{field}' not found. Available: {available}"
        raise ValueError(msg)

    raw = sliced.point_data[field]
    if raw.ndim != 2 or raw.shape[1] != 3:
        msg = f"Expected a 3-component vector field, got shape {raw.shape}."
        raise ValueError(msg)

    pts = sliced.points
    mag = np.linalg.norm(raw, axis=1)

    # Determine the two in-plane axes (h = horizontal, v = vertical)
    axes = [i for i in range(3) if i != axis_idx]
    h_idx, v_idx = axes  # e.g. normal="x" → h=y(1), v=z(2)

    h_pts = pts[:, h_idx]
    v_pts = pts[:, v_idx]
    h_pad, v_pad = 5.0, 5.0
    n_h, n_v = grid_resolution

    h_lo = yi_range[0] if yi_range is not None else h_pts.min() - h_pad
    h_hi = yi_range[1] if yi_range is not None else h_pts.max() + h_pad
    v_lo = zi_range[0] if zi_range is not None else v_pts.min() - v_pad
    v_hi = zi_range[1] if zi_range is not None else v_pts.max() + v_pad

    hi = np.linspace(h_lo, h_hi, n_h)
    vi = np.linspace(v_lo, v_hi, n_v)
    Hi, Vi = np.meshgrid(hi, vi)

    coords_2d = np.column_stack([h_pts, v_pts])
    mag_grid = griddata(coords_2d, mag, (Hi, Vi), method="linear")

    # --- colour normalisation -------------------------------------------
    if log:
        pos = mag_grid[mag_grid > 0] if np.any(mag_grid > 0) else None
        pmin = np.nanpercentile(pos, 2) if pos is not None else 1e-10
        norm = LogNorm(vmin=pmin, vmax=np.nanpercentile(mag_grid, 98))
        plot_vmin: float | None = None
        plot_vmax: float | None = None
    else:
        norm = None
        plot_vmin = 0.0
        plot_vmax = float(np.nanpercentile(mag_grid, 98))

    # --- plot -----------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        Hi,
        Vi,
        mag_grid,
        cmap=cmap,
        shading="auto",
        norm=norm,
        vmin=plot_vmin,
        vmax=plot_vmax,
    )

    if quiver:
        Fh_grid = griddata(coords_2d, raw[:, axes[0]], (Hi, Vi), method="linear")
        Fv_grid = griddata(coords_2d, raw[:, axes[1]], (Hi, Vi), method="linear")
        skip = 8
        ref_scale = (plot_vmax or np.nanpercentile(mag_grid, 98)) * 15
        ax.quiver(
            Hi[::skip, ::skip],
            Vi[::skip, ::skip],
            Fh_grid[::skip, ::skip],
            Fv_grid[::skip, ::skip],
            color="white",
            alpha=0.7,
            scale=ref_scale,
            width=0.002,
        )

    ax_labels = {0: "x", 1: "y", 2: "z"}
    ax.set_xlabel(f"{ax_labels[h_idx]} (µm)")
    ax.set_ylabel(f"{ax_labels[v_idx]} (µm)")
    ax.set_title(title or f"|{field}| cross-section at {normal}={origin}")
    ax.set_aspect("equal")

    valid = ~np.isnan(mag_grid)
    if valid.any():
        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)
        ax.set_xlim(hi[cols][0], hi[cols][-1])
        ax.set_ylim(vi[rows][0], vi[rows][-1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    fig.colorbar(im, cax=cax, label=label or f"|{field}|")
    fig.tight_layout(pad=0.5)
    plt.show()
