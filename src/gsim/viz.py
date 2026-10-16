"""Visualization utilities for gsim.

This module provides visualization tools for meshes and simulation results.
"""

from __future__ import annotations

__all__ = [
    "close_interactive_view",
    "close_interactive_views",
    "interactive_mode",
    "interactive_views",
    "plot_cross_section",
    "plot_mesh",
    "plot_topview",
    "sample_topview_field",
    "set_interactive_mode",
    "set_trame_backend",
]

import contextlib
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal, cast

import meshio
import numpy as np

logger = logging.getLogger(__name__)

#: Rendering modes for ``plot_mesh`` / ``plot_fields_2d``.
RenderMode = Literal["auto", "live", "static"]

#: Global rendering mode, overridable per call.  ``auto`` keeps the current
#: live PyVista view interactive and renders any additional views in the same
#: process as static inline images (avoids concurrent live trame servers).
_VIZ_MODE = os.environ.get("GSIM_VIZ_MODE", "auto").strip().lower() or "auto"

#: Trame backend used for notebook live views.  ``"trame"`` uses server-side
#: rendering (PyVista's default and most reliable); ``"client"`` renders
#: in-browser with vtk.js.  Override with ``GSIM_TRAME_BACKEND``.
_TRAME_BACKEND = os.environ.get("GSIM_TRAME_BACKEND", "trame").strip().lower()


def set_trame_backend(backend: Literal["trame", "client"]) -> None:
    """Set the trame backend used for notebook live views.

    ``"trame"`` (default) uses server-side rendering; ``"client"`` renders
    in-browser with vtk.js.  Also configurable via the ``GSIM_TRAME_BACKEND``
    environment variable.
    """
    global _TRAME_BACKEND  # noqa: PLW0603
    if backend not in ("trame", "client"):
        msg = f"backend must be 'trame' or 'client', got {backend!r}"
        raise ValueError(msg)
    _TRAME_BACKEND = backend


#: Master switch for interactive (live) PyVista views.  Off by default: every
#: plot renders to a static image.  When enabled via :func:`set_interactive_mode`,
#: interactive views are allowed: in a notebook each interactive plot renders as
#: its own widget on a single shared trame server; outside a notebook the plot
#: opens a blocking desktop window.  All live views stay registered until closed
#: with :func:`close_interactive_views` (or :func:`close_interactive_view`).
_INTERACTIVE_MODE = False


def set_interactive_mode(enabled: bool) -> None:
    """Enable or disable interactive (live) PyVista views.

    Defaults to ``False``: every plot renders to a static image.  When
    enabled, interactive views are allowed.  In a notebook each interactive
    plot renders as its own widget on a single shared trame server, so several
    views can stay open at once; outside a notebook the plot opens a blocking
    desktop window.  Close the open views with :func:`close_interactive_views`.

    An explicit per-call ``mode="live"`` (or ``GSIM_VIZ_MODE=live``) always
    overrides the flag.
    """
    global _INTERACTIVE_MODE  # noqa: PLW0603
    _INTERACTIVE_MODE = bool(enabled)


def interactive_mode() -> bool:
    """Return whether interactive (live) PyVista views are currently enabled."""
    return _INTERACTIVE_MODE


# -- Headless-safe PyVista initialisation ------------------------------------
def _is_headless() -> bool:
    """Return ``True`` when no usable X display is available.

    An explicit ``GSIM_FORCE_OFFSCREEN=1`` always forces off-screen
    rendering, which is useful in CI where a display may be present but X
    forwarding is undesirable.
    """
    if os.environ.get("GSIM_FORCE_OFFSCREEN") == "1":
        return True
    return not os.environ.get("DISPLAY")


def _start_xvfb() -> None:
    """Start a virtual frame buffer when ``pyvirtualdisplay`` is available.

    Xvfb only exists on Linux. Windows and macOS never set ``DISPLAY`` and
    render off-screen without it, so they skip the attempt.
    """
    if not sys.platform.startswith("linux"):
        return
    try:
        from pyvirtualdisplay import Display  # pyright: ignore[reportMissingImports]

        display = Display(visible=False, size=(1440, 900))
        display.start()
    except Exception:  # optional dependency / headless host
        # Off-screen rendering still works, so this is not a warning.
        logger.debug("pyvirtualdisplay not available; continuing without Xvfb")


def _ensure_pyvista():
    """Import PyVista, forcing off-screen rendering only when headless.

    When a usable ``DISPLAY`` is present normal rendering is left untouched
    so interactive windows work.  In headless contexts PyVista runs
    off-screen (optionally backed by a virtual frame buffer), which is what
    tests and CI rely on.  ``GSIM_FORCE_OFFSCREEN=1`` always forces
    off-screen rendering.
    """
    headless = _is_headless() or _VIZ_MODE == "static"
    if headless:
        os.environ.pop("DISPLAY", None)
        os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")

    import warnings

    import pyvista as pv

    if headless:
        pv.OFF_SCREEN = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _start_xvfb()
        except Exception:
            pass
    return pv


pv = _ensure_pyvista()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_mesh(
    msh_path: str | Path,
    output: str | Path | None = None,
    show_groups: list[str] | None = None,
    interactive: bool | None = None,
    style: Literal["wireframe", "solid"] = "wireframe",
    transparent_groups: list[str] | None = None,
    mode: RenderMode = "auto",
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
        output: Output PNG path (only used when rendering statically).
        show_groups: Group-name patterns to display (``None`` -> all).
            Example: ``["metal", "P"]`` to show metal layers and ports.
        interactive: If ``True``, force an interactive view: in a notebook
            this renders as a widget on the single shared trame server (so
            several views can coexist); otherwise it opens a blocking window.
            ``False`` forces a static PNG.  ``None`` (default) follows the
            session flag from :func:`set_interactive_mode` — off by default,
            so plots render statically until it is enabled.
        style: ``"wireframe"`` or ``"solid"``.
        transparent_groups: Group names rendered at low opacity in
            *solid* mode.  Ignored in *wireframe* mode.
        mode: Rendering mode: ``"auto"`` (default), ``"live"`` or ``"static"``.
            Overrides ``GSIM_VIZ_MODE`` for this call.  By default interactive
            views are off; enable them globally with
            :func:`set_interactive_mode`.  Close open views with
            :func:`close_interactive_views`.

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
            mode=mode,
        )
    else:
        _plot_wireframe(
            msh_path,
            output=output,
            show_groups=show_groups,
            interactive=interactive,
            mode=mode,
        )


# ---------------------------------------------------------------------------
# Wireframe renderer (original)
# ---------------------------------------------------------------------------


def _plot_wireframe(
    msh_path: Path,
    *,
    output: str | Path | None,
    show_groups: list[str] | None,
    interactive: bool | None,
    mode: RenderMode = "auto",
) -> None:
    """Wireframe renderer — one colour per matched group."""
    mio = meshio.read(msh_path, file_format="gmsh")
    group_map: dict[int, str] = {tag: name for name, (tag, _) in mio.field_data.items()}

    mesh = cast(pv.DataSet, pv.read(msh_path))  # ty: ignore[redundant-cast]
    plotter = cast(Any, _make_plotter(interactive, mode=mode))  # ty: ignore[redundant-cast]

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

    reset_camera = True
    axis_idx = _dataset_is_planar(mesh)
    if axis_idx is not None:
        _apply_front_camera(plotter, np.asarray(mesh.points), axis_idx)
        reset_camera = False
    _finish(
        plotter,
        msh_path,
        output=output,
        interactive=interactive,
        mode=mode,
        reset_camera=reset_camera,
    )


# ---------------------------------------------------------------------------
# Solid renderer (coloured surfaces per physical group)
# ---------------------------------------------------------------------------

_TRANSPARENT_DEFAULTS = ("air_boundary", "air_none", "air_plastic_enclosure")
_TRANSPARENT_OPACITY = 0.05

# meshio cell block type -> (PyVista/VTK cell type, nodes per cell, topological dim)
_SOLID_CELLTYPE_MAP: dict[str, tuple[pv.CellType, int, int]] = {
    "triangle": (pv.CellType.TRIANGLE, 3, 2),
    "triangle6": (pv.CellType.QUADRATIC_TRIANGLE, 6, 2),
    "quad": (pv.CellType.QUAD, 4, 2),
    "quad8": (pv.CellType.QUADRATIC_QUAD, 8, 2),
    "quad9": (pv.CellType.BIQUADRATIC_QUAD, 9, 2),
    "tetra": (pv.CellType.TETRA, 4, 3),
    "tetra10": (pv.CellType.QUADRATIC_TETRA, 10, 3),
    "hexahedron": (pv.CellType.HEXAHEDRON, 8, 3),
    "hexahedron20": (pv.CellType.QUADRATIC_HEXAHEDRON, 20, 3),
    "hexahedron27": (pv.CellType.TRIQUADRATIC_HEXAHEDRON, 27, 3),
}


def _normalize_solid_cell_block(
    cell_type: str,
    block_cells: np.ndarray,
) -> tuple[np.ndarray, pv.CellType, int] | None:
    """Normalize a meshio cell block for solid plotting.

    Returns a tuple: (connectivity, pyvista_cell_type, topological_dim).
    """
    if block_cells.ndim != 2 or block_cells.shape[0] == 0:
        return None

    if cell_type in _SOLID_CELLTYPE_MAP:
        pv_type, n_nodes, topo_dim = _SOLID_CELLTYPE_MAP[cell_type]
        if block_cells.shape[1] != n_nodes:
            logger.warning(
                "Skipping '%s': expected %d nodes/cell, got %d",
                cell_type,
                n_nodes,
                block_cells.shape[1],
            )
            return None
        return block_cells.astype(np.int64, copy=False), pv_type, topo_dim

    # Fallback path for unknown high-order surface elements.
    if "triangle" in cell_type and block_cells.shape[1] >= 3:
        logger.info(
            "Linearizing unsupported triangle type '%s' (%d nodes -> 3)",
            cell_type,
            block_cells.shape[1],
        )
        return block_cells[:, :3].astype(np.int64, copy=False), pv.CellType.TRIANGLE, 2

    if "quad" in cell_type and block_cells.shape[1] >= 4:
        logger.info(
            "Linearizing unsupported quad type '%s' (%d nodes -> 4)",
            cell_type,
            block_cells.shape[1],
        )
        return block_cells[:, :4].astype(np.int64, copy=False), pv.CellType.QUAD, 2

    return None


def _aligned_block_tags(phys: list[np.ndarray], idx: int, n_cells: int) -> np.ndarray:
    """Return physical tags resized to match a cell block length."""
    if idx >= len(phys):
        return np.full(n_cells, -1, dtype=int)

    tags = np.asarray(phys[idx], dtype=int)
    if tags.size < n_cells:
        pad = np.full(n_cells - tags.size, -1, dtype=int)
        return np.concatenate([tags, pad])
    if tags.size > n_cells:
        return tags[:n_cells]
    return tags


def _plot_solid(
    msh_path: Path,
    *,
    output: str | Path | None,
    interactive: bool | None,
    transparent_groups: list[str],
    mode: RenderMode = "auto",
) -> None:
    """Solid renderer — coloured surfaces per physical group."""
    mio = meshio.read(msh_path, file_format="gmsh")
    tag_to_name: dict[int, str] = {
        tag: name for name, (tag, _) in mio.field_data.items()
    }

    # Collect supported cells and physical tags ---------------------------
    # Prefer explicit 2D surface cells when available. If absent, fall
    # back to volume cells so solid mode still provides a useful view.
    cell_blocks_2d: list[tuple[np.ndarray, pv.CellType, np.ndarray]] = []
    cell_blocks_3d: list[tuple[np.ndarray, pv.CellType, np.ndarray]] = []

    phys = mio.cell_data.get("gmsh:physical", [])
    for idx, cb in enumerate(mio.cells):
        normalized = _normalize_solid_cell_block(cb.type, cb.data)
        if normalized is None:
            continue

        block_cells, pv_type, topo_dim = normalized
        tags = _aligned_block_tags(phys, idx, len(block_cells))
        if topo_dim == 2:
            cell_blocks_2d.append((block_cells, pv_type, tags))
        else:
            cell_blocks_3d.append((block_cells, pv_type, tags))

    active_blocks = cell_blocks_2d or cell_blocks_3d

    if not active_blocks:
        logger.warning("No supported solid cell blocks — falling back to wireframe.")
        _plot_wireframe(
            msh_path, output=output, show_groups=None, interactive=interactive
        )
        return

    if cell_blocks_2d:
        logger.info("Solid plot: using %d surface cell blocks", len(cell_blocks_2d))
    else:
        logger.info("Solid plot: using %d volume cell blocks", len(cell_blocks_3d))

    # Build an UnstructuredGrid -------------------------------------------
    cell_chunks: list[np.ndarray] = []
    type_chunks: list[np.ndarray] = []
    tag_chunks: list[np.ndarray] = []

    for block_cells, pv_type, tags in active_blocks:
        n = block_cells.shape[0]
        n_nodes = block_cells.shape[1]
        prefixed = np.hstack(
            [np.full((n, 1), n_nodes, dtype=np.int64), block_cells]
        ).ravel()
        cell_chunks.append(prefixed)
        type_chunks.append(np.full(n, int(pv_type), dtype=np.uint8))
        tag_chunks.append(tags)

    pv_cells = np.concatenate(cell_chunks)
    celltypes = np.concatenate(type_chunks)
    all_tags = np.concatenate(tag_chunks)
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

    plotter = cast(Any, _make_plotter(interactive, mode=mode))  # ty: ignore[redundant-cast]

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
    transparent_legend_entries: list[tuple[str, str]] = []
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
            opacity=_TRANSPARENT_OPACITY,
            edge_color=color,
            line_width=0.5,
        )
        transparent_legend_entries.append(
            (f"{group_name} (alpha={_TRANSPARENT_OPACITY:.2f})", color)
        )
        logger.info("Transparent group '%s' (colour %s)", group_name, color)

    if transparent_legend_entries:
        plotter.add_legend(transparent_legend_entries, bcolor="white", border=True)

    reset_camera = True
    axis_idx = _dataset_is_planar(grid)
    if axis_idx is not None:
        _apply_front_camera(plotter, np.asarray(grid.points), axis_idx)
        reset_camera = False
    _finish(
        plotter,
        msh_path,
        output=output,
        interactive=interactive,
        mode=mode,
        reset_camera=reset_camera,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Live (interactive) PyVista plotters currently held open, keyed by plotter id.
# Keeping the plotters referenced keeps their render windows / trame viewers
# alive.  In a notebook every interactive plot adds its own widget to the
# single shared trame server; outside a notebook a live plot blocks the
# process until its window is closed.
_LIVE_PLOTTERS: dict[str, Any] = {}


def _view_id(plotter: Any) -> str:
    """Return a stable registry key for a PyVista plotter."""
    name = getattr(plotter, "_id_name", None)
    if name is not None:
        return str(name)
    return f"plotter-{id(plotter)}"


@contextlib.contextmanager
def _quiet_trame_logging() -> Any:
    """Raise trame server/client loggers above INFO while a live view is built.

    trame's state/translator machinery logs the full Vue template
    tokenisation at INFO on every widget render — ``js_key``/``trigger()``/
    ``before:``/``after:``/``token`` lines — which floods notebook consoles
    with hundreds of lines per view.  These are debugging logs, so they are
    suppressed while the live view widget is created and restored afterwards.
    """
    noisy = (
        "trame",
        "trame_client",
        "trame_server",
        "trame_vtk",
        "trame_vuetify",
    )
    previous: dict[str, int] = {}
    for name in noisy:
        lg = logging.getLogger(name)
        previous[name] = lg.level
        lg.setLevel(max(int(lg.level), logging.WARNING))
    try:
        yield
    finally:
        for name, level in previous.items():
            logging.getLogger(name).setLevel(level)


def _in_notebook() -> bool:
    """Return ``True`` when running inside a Jupyter/IPython kernel."""
    try:
        import scooby

        return bool(scooby.in_ipykernel())
    except Exception:
        return False


def _close_live_views() -> None:
    """Close every currently open live view and clear the registry."""
    for view_id, plotter in list(_LIVE_PLOTTERS.items()):
        try:
            plotter.close()
        except Exception:
            logger.warning("Failed to close live view '%s'", view_id, exc_info=True)
    _LIVE_PLOTTERS.clear()


def close_interactive_views() -> None:
    """Close every interactive (live) view opened in this process.

    Interactive views are registered while they stay open (see
    :func:`set_interactive_mode`); calling this releases their render windows
    and trame server resources.
    """
    _close_live_views()


def close_interactive_view(view_id: str) -> None:
    """Close a single interactive view by its plotter id.

    Args:
        view_id: Plotter id as used in the live-view registry (see
            :func:`interactive_views`).
    """
    plotter = _LIVE_PLOTTERS.pop(view_id, None)
    if plotter is None:
        logger.warning("No interactive view registered with id %r", view_id)
        return
    try:
        plotter.close()
    except Exception:
        logger.warning("Failed to close interactive view '%s'", view_id, exc_info=True)


def interactive_views() -> tuple[str, ...]:
    """Return the ids of all interactive views currently held open."""
    return tuple(_LIVE_PLOTTERS)


def _dataset_is_planar(dataset: pv.DataSet) -> int | None:
    """Return the axis index a dataset is thin along (a 2D plane), else ``None``."""
    bounds = np.asarray(dataset.bounds, dtype=float).reshape(3, 2)
    span = bounds[:, 1] - bounds[:, 0]
    span_ref = max(float(np.max(span)), 1.0)
    thin = np.where(span <= 1e-9 * span_ref)[0]
    if thin.size == 1:
        return int(thin[0])
    return None


def _apply_front_camera(plotter: Any, points: np.ndarray, axis_idx: int) -> None:
    """Point a plotter camera straight at a planar point cloud.

    The camera is positioned along the plane's normal *axis_idx*, facing the
    centre of the point cloud, with the plot's vertical axis aligned to the
    in-plane axis so the result is not rolled.
    """
    axes = [i for i in range(3) if i != axis_idx]
    h = points[:, axes[0]]
    v = points[:, axes[1]]
    center = ((h.min() + h.max()) / 2, (v.min() + v.max()) / 2)
    span = (h.max() - h.min(), v.max() - v.min())
    dist = max(span) * 2.5 if max(span) > 0 else 1.0
    normal_origin = float(points[:, axis_idx].min())
    focal = [0.0, 0.0, 0.0]
    focal[axes[0]] = center[0]
    focal[axes[1]] = center[1]
    focal[axis_idx] = normal_origin
    position = list(focal)
    position[axis_idx] += dist
    up = [0.0, 0.0, 0.0]
    up[axes[1]] = 1.0
    plotter.camera.focal_point = focal
    plotter.camera.position = position
    plotter.camera.up = up


def _mode_for(mode: RenderMode) -> RenderMode:
    """Resolve the effective rendering mode for a plot.

    ``auto`` renders statically unless interactive mode is enabled via
    :func:`set_interactive_mode`, in which case live rendering is used.  In a
    notebook, live plots share a single trame server (one widget per plot);
    outside a notebook a live plot opens a blocking desktop window.
    ``GSIM_VIZ_MODE=live`` forces live rendering; ``GSIM_VIZ_MODE=static``
    (or ``none``/``off``) forces static rendering globally.
    """
    if mode != "auto":
        return mode
    if _VIZ_MODE == "live":
        return "live"
    if _INTERACTIVE_MODE and _VIZ_MODE not in ("static", "none", "off"):
        return "live"
    return "static"


def _want_live(interactive: bool | None, mode: RenderMode) -> bool:
    """Return whether a plot should be shown live.

    ``interactive=True`` forces a live view (unless an explicit ``mode`` /
    ``GSIM_VIZ_MODE`` static override is in force).  ``interactive=False``
    forces a static screenshot.  ``interactive=None`` (default) follows the
    session flag from :func:`set_interactive_mode` — off by default.
    """
    if interactive is False:
        return False
    if _mode_for(mode) == "live":
        return True
    return (
        interactive is True
        and mode == "auto"
        and _VIZ_MODE not in ("static", "none", "off")
    )


def _make_plotter(interactive: bool | None, *, mode: RenderMode = "auto") -> Any:
    """Create a PyVista plotter with standard window settings."""
    pv = _ensure_pyvista()
    off_screen = not _want_live(interactive, mode)
    density = {"window_size": [1200, 900]}
    if off_screen:
        plotter = pv.Plotter(off_screen=True, **density)
    else:
        plotter = pv.Plotter(**density)
    plotter.set_background("white")
    return plotter


def _show_static(plotter: Any) -> None:
    """Render a plotter to a temporary PNG and display it inline.

    Used as a fallback when an interactive trame widget cannot be created, so
    the plot is still shown rather than silently dropped.
    """
    dest = _default_output(None, ".png")
    with contextlib.suppress(Exception):
        plotter.render()
    try:
        plotter.screenshot(str(dest))
    except Exception:
        logger.warning("Failed to render fallback static image", exc_info=True)
        dest = None
    finally:
        with contextlib.suppress(Exception):
            plotter.close()
    if dest is not None and dest.exists():
        _disp_img(dest)


def _show_live(plotter: Any) -> None:
    """Show a plotter live and register it in the open-views registry.

    In a notebook the plot is rendered as an interactive widget on the single
    shared trame server (server-side rendering by default, see
    :func:`set_trame_backend`), so any number of interactive views can coexist
    on one server.  PyVista displays the widget itself; this function only
    renders the scene first and keeps the plotter alive in the registry.  If
    the widget cannot be created, a static image is shown instead.  Outside a
    notebook a blocking desktop window is opened, replacing any previously
    open live view.
    """
    if _in_notebook() and getattr(plotter, "notebook", False):
        try:
            plotter.render()
            with _quiet_trame_logging():
                plotter.show(jupyter_backend=_TRAME_BACKEND)
        except Exception:
            logger.warning(
                "Failed to render interactive trame widget; showing a static image",
                exc_info=True,
            )
            _show_static(plotter)
            return
        _LIVE_PLOTTERS[_view_id(plotter)] = plotter
        return
    _close_live_views()
    plotter.show()
    _LIVE_PLOTTERS[_view_id(plotter)] = plotter


def _show_or_screenshot(
    plotter: Any,
    *,
    msh_path: Path | None = None,
    output: str | Path | None = None,
    interactive: bool | None,
    mode: RenderMode = "auto",
    suffix: str = ".png",
    reset_camera: bool = True,
) -> None:
    """Show a plotter live, or screenshot it, and clean up.

    Interactive (live) views are disabled by default; enable them globally
    with :func:`set_interactive_mode` or force them per call by passing
    ``interactive=True``.  When the plot is shown live it is registered in
    the open-views registry (see :func:`interactive_views`): in a notebook it
    renders as its own widget on the single shared trame server, so several
    views can coexist; outside a notebook it opens a blocking desktop window,
    replacing any previously open live view.  Close open views with
    :func:`close_interactive_views`.

    Otherwise the plot is rendered to a static PNG and displayed inline.

    When *reset_camera* is ``True`` (default) the camera is reset to an
    isometric view; pass ``False`` to preserve a camera already configured on
    the plotter (used by ``plot_fields_2d`` to face the slice plane).
    """
    if reset_camera:
        plotter.camera_position = "iso"
    plotter.show_axes()

    if _want_live(interactive, mode):
        _show_live(plotter)
        return

    # Static screenshot: either interactive=False or the effective mode is
    # "static".
    dest = output if output is not None else _default_output(msh_path, suffix)
    plotter.screenshot(str(dest))
    plotter.close()
    _disp_img(Path(dest))


def _default_output(msh_path: Path | None, suffix: str) -> Path:
    """Return the default screenshot path for a plot.

    Uses the mesh path with a new suffix, or a fresh temporary file when no
    mesh path is available.
    """
    if msh_path is not None:
        return msh_path.with_suffix(suffix)
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        name = tmp.name
    return Path(name)


def _disp_img(path: Path) -> None:
    """Display a saved image inline when running inside IPython."""
    try:
        from IPython.display import Image, display

        display(Image(str(path)))
    except ImportError:
        logger.info("Saved plot to %s", path)


def _finish(
    plotter: Any,
    msh_path: Path,
    *,
    output: str | Path | None = None,
    interactive: bool | None,
    mode: RenderMode = "auto",
    reset_camera: bool = True,
) -> None:
    """Show or screenshot the plotter and clean up."""
    _show_or_screenshot(
        plotter,
        msh_path=msh_path,
        output=output,
        interactive=interactive,
        mode=mode,
        reset_camera=reset_camera,
    )


def _color_for_group(name: str) -> str:
    """Deterministic colour for a group name."""
    if name == "air_boundary":
        return "lightblue"
    h = int(hashlib.md5(name.encode()).hexdigest()[:6], 16)
    return f"#{h:06x}"


def _safe_nanpercentile(values: np.ndarray, q: float, *, default: float) -> float:
    """Return nanpercentile or *default* when all entries are non-finite."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    return float(np.percentile(finite, q))


# ---------------------------------------------------------------------------
# Top-view field plot
# ---------------------------------------------------------------------------


def sample_topview_field(
    dataset: pv.DataSet,
    *,
    field: str,
    z: float | None = None,
    component: int | None = None,
    attribute_values: list[int] | None = None,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    pad: tuple[float, float] = (5.0, 5.0),
    grid_resolution: tuple[int, int] = (500, 150),
    snap_to_closest_point: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a field on a regular XY grid at fixed ``z``.

    Uses VTK/PyVista sampling so interpolation is mesh-connectivity-aware.

    Args:
        dataset: PyVista dataset containing field data.
        field: Field name in ``dataset.point_data``.
        z: Z-plane where the top view is sampled. For boundary-only fields
            (e.g. ``J_s_real``), ``None`` enables automatic top-surface
            detection.
        component: Optional vector component index (0=x, 1=y, 2=z).
            If ``None`` and *field* is vector-valued, returns magnitude.
        attribute_values: Optional list of boundary/region ``attribute`` IDs
            to keep before sampling (material-aware filtering).
        x_range: Optional ``(xmin, xmax)``. Auto if ``None``.
        y_range: Optional ``(ymin, ymax)``. Auto if ``None``.
        pad: Auto-range padding ``(x_pad, y_pad)`` in µm.
        grid_resolution: Number of samples as ``(nx, ny)``.
        snap_to_closest_point: If ``True``, use nearest-point snap in VTK
            probing. Keep ``False`` for strict interpolation.

    Returns:
        ``(Xi, Yi, Gi)`` meshgrid arrays for plotting with ``pcolormesh``.
    """
    surface_fields = {"J_s_real", "J_s_imag", "Q_s_real", "Q_s_imag"}
    is_surface_field = field in surface_fields

    source: pv.DataSet = dataset

    # Automatic material-aware filtering for boundary-only quantities.
    # If the user does not pass attribute_values and the field is surface-only,
    # keep only attributes with strongest activity to avoid smearing over air.
    if (
        attribute_values is None
        and is_surface_field
        and "attribute" in dataset.cell_data
        and field in dataset.point_data
    ):
        attrs = np.asarray(dataset.cell_data["attribute"], dtype=int)
        arr = dataset.point_data[field]
        activity = np.linalg.norm(arr, axis=1) if arr.ndim == 2 else np.abs(arr)

        scores: dict[int, list[float]] = {}
        ztops: dict[int, float] = {}
        for cell_id in range(dataset.n_cells):
            point_ids = dataset.get_cell(cell_id).point_ids
            if len(point_ids) == 0:
                continue
            attr_id = int(attrs[cell_id])
            scores.setdefault(attr_id, []).append(
                float(np.nanmean(activity[point_ids]))
            )
            z_local_max = float(np.nanmax(dataset.points[point_ids, 2]))
            prev = ztops.get(attr_id)
            ztops[attr_id] = z_local_max if prev is None else max(prev, z_local_max)

        if scores:
            global_top_z = float(np.nanmax(dataset.points[:, 2]))
            z_tol = 0.5
            top_candidates = [
                attr_id
                for attr_id, zmax in ztops.items()
                if zmax >= global_top_z - z_tol
            ]

            candidate_ids = top_candidates or list(scores.keys())
            ranked = sorted(
                (
                    (attr_id, np.nanpercentile(scores[attr_id], 90))
                    for attr_id in candidate_ids
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            attribute_values = [attr_id for attr_id, _ in ranked[:4]]
            logger.info(
                "Auto-selected top-surface boundary attributes for %s top-view: %s",
                field,
                attribute_values,
            )

    if attribute_values is not None:
        if "attribute" not in dataset.cell_data:
            msg = "attribute_values provided but dataset has no cell_data['attribute']."
            raise ValueError(msg)
        attrs = np.asarray(dataset.cell_data["attribute"])
        keep_ids = np.where(np.isin(attrs, np.asarray(attribute_values)))[0]
        if keep_ids.size == 0:
            msg = f"No cells found for attribute_values={attribute_values}."
            raise ValueError(msg)
        source = cast(pv.DataSet, dataset.extract_cells(keep_ids))

    # For boundary-only fields, auto-detect the top surface when requested z
    # does not intersect enough geometry (or when z is omitted).
    z_use: float
    if is_surface_field:
        z_vals = source.points[:, 2]
        z_span = (
            float(np.nanmax(z_vals) - np.nanmin(z_vals)) if source.n_points > 0 else 0.0
        )
        z_tol = max(0.2, 0.01 * z_span)

        def _slice_count(z0: float) -> int:
            return source.slice(normal="z", origin=(0.0, 0.0, z0)).n_points

        z_use = float(np.nanmax(z_vals)) if z is None else float(z)

        min_points = max(25, int(0.001 * max(source.n_points, 1)))
        need_auto_top = _slice_count(z_use) < min_points

        if need_auto_top:
            z_top = float(np.nanmax(z_vals))
            top_mask = np.abs(z_vals - z_top) <= z_tol
            top = source.extract_points(
                top_mask, adjacent_cells=True, include_cells=True
            )
            if top.n_points > 0 and field in top.point_data:
                source = cast(pv.DataSet, top)
                z_use = float(np.nanmedian(top.points[:, 2]))
                logger.info(
                    "sample_topview_field: using auto-detected top "
                    "surface at z=%s for %s",
                    z_use,
                    field,
                )
            else:
                z_use = z_top
    else:
        if z is None:
            msg = "z must be provided for non-surface top-view fields."
            raise ValueError(msg)
        z_use = float(z)

    nx, ny = grid_resolution
    x_pad, y_pad = pad

    planar = source.slice(normal="z", origin=(0.0, 0.0, z_use))
    bounds_source = planar if planar.n_points > 0 else source
    if bounds_source.n_points == 0:
        msg = "Dataset has no points available for top-view sampling."
        raise ValueError(msg)

    pts = bounds_source.points
    x_lo = x_range[0] if x_range is not None else float(pts[:, 0].min() - x_pad)
    x_hi = x_range[1] if x_range is not None else float(pts[:, 0].max() + x_pad)
    y_lo = y_range[0] if y_range is not None else float(pts[:, 1].min() - y_pad)
    y_hi = y_range[1] if y_range is not None else float(pts[:, 1].max() + y_pad)

    xi = np.linspace(x_lo, x_hi, nx)
    yi = np.linspace(y_lo, y_hi, ny)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = np.full_like(Xi, z_use)

    probe = pv.StructuredGrid(Xi, Yi, Zi)
    sampled = probe.sample(
        cast(pv.DataSet, source),  # ty: ignore[redundant-cast]
        snap_to_closest_point=snap_to_closest_point,
    )

    if field not in sampled.point_data:
        available = list(sampled.point_data.keys())
        msg = f"Field '{field}' not found in sampled data. Available: {available}"
        raise ValueError(msg)

    arr = sampled.point_data[field]
    if arr.ndim == 2:
        if component is None:
            values = np.linalg.norm(arr, axis=1)
        else:
            if component < 0 or component >= arr.shape[1]:
                msg = (
                    f"Component index {component} is invalid for field '{field}' "
                    f"with {arr.shape[1]} components."
                )
                raise ValueError(msg)
            values = arr[:, component]
    else:
        if component is not None:
            msg = f"Field '{field}' is scalar; component index is invalid."
            raise ValueError(msg)
        values = arr

    Gi = values.reshape(Xi.shape, order="F")

    if "vtkValidPointMask" in sampled.point_data:
        valid = (
            sampled.point_data["vtkValidPointMask"]
            .astype(bool)
            .reshape(
                Xi.shape,
                order="F",
            )
        )
        if not valid.any() and not snap_to_closest_point:
            sampled = probe.sample(cast(pv.DataSet, source), snap_to_closest_point=True)  # ty: ignore[redundant-cast]
            arr = sampled.point_data[field]
            if arr.ndim == 2:
                values = (
                    np.linalg.norm(arr, axis=1)
                    if component is None
                    else arr[:, component]
                )
            else:
                values = arr
            Gi = values.reshape(Xi.shape, order="F")
            valid = (
                sampled.point_data["vtkValidPointMask"]
                .astype(bool)
                .reshape(
                    Xi.shape,
                    order="F",
                )
            )
            logger.warning(
                "sample_topview_field: no strict valid points at z=%s; "
                "falling back to snap_to_closest_point.",
                z_use,
            )

        if valid.any():
            Gi = np.where(valid, Gi, np.nan)

    return Xi, Yi, Gi


def plot_topview(
    dataset: pv.DataSet,
    *,
    field: str,
    z: float | None = None,
    title: str,
    component: int | None = None,
    attribute_values: list[int] | None = None,
    cmap: str = "turbo",
    log: bool = False,
    symmetric: bool = False,
    figsize: tuple[float, float] = (14, 3.5),
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    pad: tuple[float, float] = (5.0, 5.0),
    grid_resolution: tuple[int, int] = (500, 150),
    snap_to_closest_point: bool = False,
    surface_direct: bool | None = None,
) -> None:
    """Plot a sampled top-view field map on the XY plane at fixed ``z``.

    Args:
        dataset: PyVista dataset containing field data.
        field: Field name in ``dataset.point_data``.
        z: Z-plane where the top view is sampled. For boundary-only fields
            (e.g. ``J_s_real``), ``None`` enables automatic top-surface
            detection.
        title: Plot title.
        component: Optional vector component index (0=x, 1=y, 2=z).
            If ``None`` and *field* is vector-valued, plots magnitude.
        attribute_values: Optional list of ``attribute`` IDs to keep before
            sampling (material-aware filtering).
        cmap: Matplotlib colormap.
        log: Use logarithmic color scale (magnitude-like data only).
        symmetric: Force symmetric color limits ``[-v, +v]``.
        figsize: Figure size.
        x_range: Optional horizontal plotting limits.
        y_range: Optional vertical plotting limits.
        pad: Auto-range padding ``(x_pad, y_pad)`` in µm.
        grid_resolution: Number of samples as ``(nx, ny)``.
        snap_to_closest_point: If ``True``, force nearest-point snap in VTK
            probing instead of strict interpolation.
        surface_direct: For boundary-only fields, render directly on the
            triangulated surface mesh instead of resampling to a regular grid.
            ``None`` enables automatic behavior (direct for J_s/Q_s fields).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.tri import Triangulation

    surface_fields = {"J_s_real", "J_s_imag", "Q_s_real", "Q_s_imag"}
    use_surface_direct = (
        field in surface_fields if surface_direct is None else surface_direct
    )

    if use_surface_direct and field in dataset.point_data:
        source = dataset
        if attribute_values is not None:
            if "attribute" not in dataset.cell_data:
                msg = (
                    "attribute_values provided but dataset has no "
                    "cell_data['attribute']."
                )
                raise ValueError(msg)
            attrs = np.asarray(dataset.cell_data["attribute"])
            keep_ids = np.where(np.isin(attrs, np.asarray(attribute_values)))[0]
            if keep_ids.size == 0:
                msg = f"No cells found for attribute_values={attribute_values}."
                raise ValueError(msg)
            source = dataset.extract_cells(keep_ids)

        if source.n_points == 0:
            msg = "No points available for direct surface plotting."
            raise ValueError(msg)

        surf = source.extract_surface(algorithm="dataset_surface").triangulate()  # ty: ignore[unknown-argument]
        if surf.n_cells == 0 or field not in surf.point_data:
            logger.warning(
                "plot_topview: direct surface plot unavailable for %s; "
                "falling back to sampled grid.",
                field,
            )
        else:
            arr = surf.point_data[field]
            if arr.ndim == 2:
                if component is None:
                    point_values = np.linalg.norm(arr, axis=1)
                else:
                    point_values = arr[:, component]
            else:
                point_values = arr

            x = surf.points[:, 0]
            y = surf.points[:, 1]
            faces = surf.faces.reshape(-1, 4)
            triangles = faces[:, 1:4]

            point_valid = np.isfinite(point_values)
            cell_values = np.array(
                [
                    float(np.nanmean(point_values[tri]))
                    if np.any(point_valid[tri])
                    else np.nan
                    for tri in triangles
                ]
            )

            valid_tri = np.isfinite(cell_values)
            zc = np.array([float(np.nanmean(surf.points[tri, 2])) for tri in triangles])
            z_span = float(np.nanmax(surf.points[:, 2]) - np.nanmin(surf.points[:, 2]))
            z_tol = max(0.2, 0.01 * z_span)
            z_target = float(np.nanmax(zc)) if z is None else float(z)
            valid_tri &= np.abs(zc - z_target) <= z_tol

            if not np.any(valid_tri):
                logger.warning(
                    "plot_topview: no valid triangles after filtering for %s; "
                    "falling back to sampled grid.",
                    field,
                )
            else:
                tri = Triangulation(x, y, triangles[valid_tri])
                plot_values = cell_values[valid_tri]

                norm = None
                plot_vmin: float | None = None
                plot_vmax: float | None = None
                if log:
                    pos = plot_values[np.isfinite(plot_values) & (plot_values > 0)]
                    pmin = _safe_nanpercentile(pos, 2, default=1e-10)
                    vmax = _safe_nanpercentile(
                        plot_values,
                        98,
                        default=max(pmin * 10, 1e-9),
                    )
                    norm = LogNorm(vmin=pmin, vmax=max(vmax, pmin * 1.01))
                elif symmetric:
                    vlim = _safe_nanpercentile(np.abs(plot_values), 98, default=1.0)
                    plot_vmin = -vlim
                    plot_vmax = vlim
                else:
                    plot_vmin = 0.0
                    plot_vmax = _safe_nanpercentile(plot_values, 98, default=1.0)

                fig, ax = plt.subplots(figsize=figsize)
                im = ax.tripcolor(
                    tri,
                    facecolors=plot_values,
                    cmap=cmap,
                    shading="flat",
                    norm=norm,
                    vmin=plot_vmin,
                    vmax=plot_vmax,
                )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                ax.set_title(title)
                ax.set_aspect("equal")
                ax.set_xlabel("x (µm)")
                ax.set_ylabel("y (µm)")

                tri_pts = triangles[valid_tri].ravel()
                ax.set_xlim(float(np.nanmin(x[tri_pts])), float(np.nanmax(x[tri_pts])))
                ax.set_ylim(float(np.nanmin(y[tri_pts])), float(np.nanmax(y[tri_pts])))

                fig.tight_layout(pad=0.5)
                plt.show()
                return

    Xi, Yi, Gi = sample_topview_field(
        dataset,
        field=field,
        z=z,
        component=component,
        attribute_values=attribute_values,
        x_range=x_range,
        y_range=y_range,
        pad=pad,
        grid_resolution=grid_resolution,
        snap_to_closest_point=snap_to_closest_point,
    )

    norm = None
    plot_vmin: float | None = None
    plot_vmax: float | None = None

    if log:
        pos = Gi[np.isfinite(Gi) & (Gi > 0)]
        pmin = _safe_nanpercentile(pos, 2, default=1e-10)
        vmax = _safe_nanpercentile(Gi, 98, default=max(pmin * 10, 1e-9))
        norm = LogNorm(vmin=pmin, vmax=max(vmax, pmin * 1.01))
    elif symmetric:
        vlim = _safe_nanpercentile(np.abs(Gi), 98, default=1.0)
        plot_vmin = -vlim
        plot_vmax = vlim
    else:
        plot_vmin = 0.0
        plot_vmax = _safe_nanpercentile(Gi, 98, default=1.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(
        Xi,
        Yi,
        Gi,
        cmap=cmap,
        shading="auto",
        norm=norm,
        vmin=plot_vmin,
        vmax=plot_vmax,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    valid = ~np.isnan(Gi)
    if valid.any():
        xi = Xi[0, :]
        yi = Yi[:, 0]
        rows = np.any(valid, axis=1)
        cols = np.any(valid, axis=0)
        ax.set_xlim(xi[cols][0], xi[cols][-1])
        ax.set_ylim(yi[rows][0], yi[rows][-1])

    fig.tight_layout(pad=0.5)
    plt.show()


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

    Slices *vol* along the given *normal* axis at *origin*, samples the
    vector field onto a regular in-plane grid using VTK/PyVista (cell-
    connectivity-aware interpolation), and overlays quiver arrows showing
    the in-plane field direction.

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

    # --- slice the volume ------------------------------------------------
    axis_idx = {"x": 0, "y": 1, "z": 2}[normal]
    origin_pt = [0.0, 0.0, 0.0]
    origin_pt[axis_idx] = origin
    sliced = cast(pv.DataSet, vol.slice(normal=normal, origin=tuple(origin_pt)))

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

    # Determine the two in-plane axes (h = horizontal, v = vertical)
    axes = [i for i in range(3) if i != axis_idx]
    h_idx, v_idx = axes  # e.g. normal="x" -> h=y(1), v=z(2)

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

    X3 = np.zeros_like(Hi)
    Y3 = np.zeros_like(Hi)
    Z3 = np.zeros_like(Hi)
    if normal == "x":
        X3[:, :] = origin
        Y3[:, :] = Hi
        Z3[:, :] = Vi
    elif normal == "y":
        X3[:, :] = Hi
        Y3[:, :] = origin
        Z3[:, :] = Vi
    else:  # normal == "z"
        X3[:, :] = Hi
        Y3[:, :] = Vi
        Z3[:, :] = origin

    probe_grid = pv.StructuredGrid(X3, Y3, Z3)
    sampled = probe_grid.sample(cast(pv.DataSet, sliced))  # ty: ignore[redundant-cast]

    if field not in sampled.point_data:
        available = list(sampled.point_data.keys())
        msg = f"Sampled field '{field}' not found. Available: {available}"
        raise ValueError(msg)

    vec_raw = sampled.point_data[field]
    vec_grid = np.stack(
        [vec_raw[:, i].reshape((n_v, n_h), order="F") for i in range(3)],
        axis=2,
    )
    mag_grid = np.linalg.norm(vec_grid, axis=2)

    valid_mask = None
    if "vtkValidPointMask" in sampled.point_data:
        valid_mask = (
            sampled.point_data["vtkValidPointMask"]
            .astype(bool)
            .reshape(
                (n_v, n_h),
                order="F",
            )
        )
        if valid_mask.any():
            mag_grid = np.where(valid_mask, mag_grid, np.nan)
        else:
            logger.warning(
                "plot_cross_section: vtkValidPointMask has no valid points at %s=%s; "
                "using unmasked sampled values.",
                normal,
                origin,
            )
            valid_mask = None

    # --- colour normalisation -------------------------------------------
    if log:
        pos = mag_grid[np.isfinite(mag_grid) & (mag_grid > 0)]
        pmin = _safe_nanpercentile(pos, 2, default=1e-10)
        vmax = _safe_nanpercentile(mag_grid, 98, default=max(pmin * 10, 1e-9))
        norm = LogNorm(vmin=pmin, vmax=max(vmax, pmin * 1.01))
        plot_vmin: float | None = None
        plot_vmax: float | None = None
    else:
        norm = None
        plot_vmin = 0.0
        plot_vmax = _safe_nanpercentile(mag_grid, 98, default=1.0)

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
        Fh_grid = vec_grid[:, :, axes[0]]
        Fv_grid = vec_grid[:, :, axes[1]]
        if valid_mask is not None:
            Fh_grid = np.where(valid_mask, Fh_grid, np.nan)
            Fv_grid = np.where(valid_mask, Fv_grid, np.nan)
        skip = 8
        # Keep arrows readable across typical Palace field magnitudes.
        # Smaller scale -> longer/more visible arrows in Matplotlib quiver.
        ref_scale = (plot_vmax or _safe_nanpercentile(mag_grid, 98, default=1.0)) * 5
        ax.quiver(
            Hi[::skip, ::skip],
            Vi[::skip, ::skip],
            Fh_grid[::skip, ::skip],
            Fv_grid[::skip, ::skip],
            color="white",
            alpha=0.7,
            scale=ref_scale,
            width=0.003,
        )

    ax_labels = {0: "x", 1: "y", 2: "z"}
    ax.set_xlabel(f"{ax_labels[h_idx]} (µm)")
    ax.set_ylabel(f"{ax_labels[v_idx]} (µm)")
    ax.set_title(title or f"|{field}| cross-section at {normal}={origin}")
    ax.set_aspect("equal")

    if yi_range is not None:
        ax.set_xlim(*yi_range)
    if zi_range is not None:
        ax.set_ylim(*zi_range)

    if yi_range is None or zi_range is None:
        valid = ~np.isnan(mag_grid)
        if valid.any():
            rows = np.any(valid, axis=1)
            cols = np.any(valid, axis=0)
            if yi_range is None:
                ax.set_xlim(hi[cols][0], hi[cols][-1])
            if zi_range is None:
                ax.set_ylim(vi[rows][0], vi[rows][-1])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    fig.colorbar(im, cax=cax, label=label or f"|{field}|")
    fig.tight_layout(pad=0.5)
    plt.show()
