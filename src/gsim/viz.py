"""Visualization utilities for gsim.

This module provides visualization tools for meshes and simulation results.
"""

from __future__ import annotations

__all__ = ["plot_cross_section", "plot_mesh", "plot_topview", "sample_topview_field"]

import hashlib
import logging
from pathlib import Path
from typing import Any, Literal, cast

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

    mesh = cast(pv.DataSet, pv.read(msh_path))  # ty: ignore[redundant-cast]
    plotter = cast(Any, _make_plotter(interactive))  # ty: ignore[redundant-cast]

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

    plotter = cast(Any, _make_plotter(interactive))  # ty: ignore[redundant-cast]

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


def _make_plotter(interactive: bool) -> Any:
    """Create a PyVista plotter with standard window settings."""
    if interactive:
        plotter = pv.Plotter(window_size=[1200, 900])
    else:
        plotter = pv.Plotter(off_screen=True, window_size=[1200, 900])
    plotter.set_background("white")  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    return plotter


def _finish(
    plotter: Any,
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


# ---------------------------------------------------------------------------
# Top-view field plot
# ---------------------------------------------------------------------------


def _sample_topview_field_duplicate(
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
                    "sample_topview_field: using auto-detected top surface at "
                    "z=%s for %s",
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


def _plot_topview_duplicate(
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


def _plot_cross_section_duplicate(
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
