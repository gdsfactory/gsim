"""2D field visualization utilities for Palace ParaView outputs."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyvista as pv
from pyvista.plotting._typing import ColormapOptions

from gsim.palace.results import load_fields

logger = logging.getLogger(__name__)


def resolve_physical_groups(
    source: str | Path,
    group_names: Sequence[str],
) -> list[int]:
    """Resolve physical group names to Palace attribute values.

    Reads the mesh file's physical group definitions and returns the
    numeric attribute values corresponding to the requested group names.

    Args:
        source: Simulation directory, local ``output/palace`` directory,
            downloaded cloud ``output`` directory, or a ``.msh`` file.
            Cloud meshes are read from the corresponding ``input`` directory.
        group_names: Physical group names to resolve (e.g.
            ``["n_rib", "p_rib", "slab90"]``).

    Returns:
        List of attribute values (integer tags) to use for cell filtering
        in Palace Paraview output.
    """
    import meshio

    path = Path(source)
    if path.suffix != ".msh":
        mesh_roots = [path, path / "input"]
        if path.name == "output":
            mesh_roots.append(path.parent / "input")
        elif path.name == "palace" and path.parent.name == "output":
            mesh_roots.append(path.parent.parent)
        candidates = [root / "palace.msh" for root in mesh_roots]
        path = next(
            (candidate for candidate in candidates if candidate.is_file()),
            candidates[0],
        )
    if not path.exists():
        msg = f"Mesh file not found: {path}"
        raise FileNotFoundError(msg)

    m = meshio.read(str(path))
    if not hasattr(m, "field_data") or not m.field_data:
        msg = f"No physical groups found in mesh: {path}"
        raise ValueError(msg)

    requested = set(group_names)
    found: list[int] = []
    missing = set(requested)
    for name, (tag, _dim) in m.field_data.items():
        if name in requested:
            found.append(int(tag))
            missing.discard(name)
    if missing:
        available = sorted(m.field_data.keys())
        msg = f"Physical group(s) not found: {sorted(missing)}. Available: {available}"
        raise ValueError(msg)
    return found


Axis = Literal["x", "y", "z"]


@dataclass(frozen=True)
class StreamplotInputs2D:
    """2D arrays and seed points suitable for matplotlib streamplot.

    Attributes mirror PalaceToolkit streamplot inputs:
    ``x``, ``y``, ``u``, ``v``, and optional ``start_points``.
    """

    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    start_points: np.ndarray
    et_mag: np.ndarray
    en_mag: np.ndarray
    normal: Axis


def _axis_name(idx: int) -> Axis:
    """Map axis index 0/1/2 to axis name x/y/z."""
    return cast(Axis, {0: "x", 1: "y", 2: "z"}[idx])


def _source_to_dataset(
    source: str | Path | dict | pv.DataSet,
    *,
    excitation: int,
    cycle: int | None,
    boundary: bool,
) -> Any:
    """Return a dataset from either an existing DataSet or Palace source."""
    if isinstance(source, pv.DataSet):
        return source
    return load_fields(source, excitation=excitation, cycle=cycle, boundary=boundary)


def _slice_plane(
    dataset: pv.DataSet,
    *,
    normal: Axis,
    origin: float,
) -> tuple[Any, Axis, int, list[int]]:
    """Return a planar dataset and axis mapping information.

    For already-planar datasets (single thin axis), the function keeps the
    native plane and ignores the requested slicing axis.
    """
    requested_idx = {"x": 0, "y": 1, "z": 2}[normal]

    bounds = np.asarray(dataset.bounds, dtype=float).reshape(3, 2)
    span = bounds[:, 1] - bounds[:, 0]
    span_ref = max(float(np.max(span)), 1.0)
    thin = np.where(span <= 1e-9 * span_ref)[0]

    if thin.size == 1:
        axis_idx = int(thin[0])
        if axis_idx != requested_idx:
            logger.info(
                "Input field is already 2D in %s-normal plane; ignoring normal=%s",
                _axis_name(axis_idx),
                normal,
            )
        axes = [i for i in range(3) if i != axis_idx]
        surf = dataset.extract_surface(algorithm="dataset_surface").triangulate()  # ty: ignore[unknown-argument]
        if surf.n_points > 0:
            return surf, _axis_name(axis_idx), axis_idx, axes

    axis_idx = requested_idx
    axes = [i for i in range(3) if i != axis_idx]

    origin_pt = [0.0, 0.0, 0.0]
    origin_pt[axis_idx] = origin
    sliced = dataset.slice(normal=normal, origin=tuple(origin_pt))

    if sliced.n_points > 0:
        return sliced, normal, axis_idx, axes

    # Fallback for already-planar datasets where exact slicing may miss points.
    bounds = np.asarray(dataset.bounds, dtype=float).reshape(3, 2)
    span = bounds[:, 1] - bounds[:, 0]
    if span[axis_idx] <= 1e-9 * max(float(np.max(span)), 1.0):
        surf = dataset.extract_surface(algorithm="dataset_surface").triangulate()  # ty: ignore[unknown-argument]
        if surf.n_points > 0:
            return surf, normal, axis_idx, axes

    msg = f"No points found on slice {normal}={origin}."
    raise ValueError(msg)


def _plane_grid(
    planar: pv.DataSet,
    *,
    normal: Axis,
    origin: float,
    grid_resolution: tuple[int, int],
) -> tuple[pv.StructuredGrid, np.ndarray, np.ndarray]:
    """Create a regular 2D grid embedded in 3D at the requested plane."""
    pts = planar.points
    axis_idx = {"x": 0, "y": 1, "z": 2}[normal]
    axes = [i for i in range(3) if i != axis_idx]
    h_idx, v_idx = axes

    n_h, n_v = grid_resolution
    pad_h, pad_v = 2.0, 2.0

    h_vals = pts[:, h_idx]
    v_vals = pts[:, v_idx]
    h = np.linspace(
        float(np.min(h_vals)) - pad_h,
        float(np.max(h_vals)) + pad_h,
        n_h,
    )
    v = np.linspace(
        float(np.min(v_vals)) - pad_v,
        float(np.max(v_vals)) + pad_v,
        n_v,
    )
    H, V = np.meshgrid(h, v)

    X3 = np.zeros_like(H)
    Y3 = np.zeros_like(H)
    Z3 = np.zeros_like(H)

    if normal == "x":
        X3[:, :] = origin
        Y3[:, :] = H
        Z3[:, :] = V
    elif normal == "y":
        X3[:, :] = H
        Y3[:, :] = origin
        Z3[:, :] = V
    else:
        X3[:, :] = H
        Y3[:, :] = V
        Z3[:, :] = origin

    return pv.StructuredGrid(X3, Y3, Z3), H, V


def _complex_vector(dataset: Any, field: str) -> np.ndarray:
    """Return complex vector data using paired *_real/*_imag arrays when present."""
    real = np.asarray(dataset.point_data[field], dtype=float)
    if field.endswith("_real"):
        imag_name = f"{field[:-5]}_imag"
        if imag_name in dataset.point_data:
            imag = np.asarray(dataset.point_data[imag_name], dtype=float)
            if imag.shape == real.shape:
                return real + 1j * imag
    return real.astype(np.complex128)


def _complex_normal_component(dataset: Any, field: str, size: int) -> np.ndarray:
    """Return complex normal component for 2-component fields when available."""
    if field.endswith("_real"):
        base = field[:-5]
        nreal_name = f"{base}n_real"
        if nreal_name in dataset.point_data:
            nreal = np.asarray(dataset.point_data[nreal_name], dtype=float)
            if nreal.shape == (size,):
                imag = np.zeros_like(nreal)
                nimag_name = f"{base}n_imag"
                if nimag_name in dataset.point_data:
                    nimag = np.asarray(dataset.point_data[nimag_name], dtype=float)
                    if nimag.shape == nreal.shape:
                        imag = nimag
                return nreal + 1j * imag
    return np.zeros(size, dtype=np.complex128)


def _phase_lock_inplane(
    inplane_h: np.ndarray,
    inplane_v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply global phase-locking like PalaceToolkit before taking real vectors."""
    phi_ref = np.angle(np.nansum(inplane_h) + 1j * np.nansum(inplane_v))
    rot = np.exp(-1j * phi_ref)
    return np.real(inplane_h * rot), np.real(inplane_v * rot)


def _plane_axis_labels(normal: Axis) -> tuple[str, str]:
    """Return in-plane axis labels for the selected normal axis."""
    if normal == "x":
        return "y", "z"
    if normal == "y":
        return "x", "z"
    return "x", "y"


def _filter_start_points_in_bounds(
    points: np.ndarray,
    *,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Keep only start points that fall inside streamplot bounds."""
    if points.size == 0:
        return points.reshape(0, 2)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    keep = (
        (points[:, 0] >= x_min)
        & (points[:, 0] <= x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] <= y_max)
    )
    return points[keep]


def plot_fields_2d(
    source: str | Path | dict | pv.DataSet,
    *,
    field: str = "E_real",
    normal: Axis = "x",
    origin: float = 0.0,
    excitation: int = 1,
    cycle: int | None = None,
    boundary: bool = False,
    physical_groups: Sequence[str] | None = None,
    cmap: ColormapOptions = "hot",
    clim: tuple[float, float] | None = None,
    title: str = "|E|",
    show_edges: bool = False,
    opacity: float = 1.0,
    show: bool = True,
    interactive: bool | None = None,
    screenshot: str | Path | None = None,
    mode: Literal["auto", "live", "static"] = "auto",
) -> Any:
    """Plot 2D field using PyVista, rendering directly on the mesh.

    When *physical_groups* is given, only cells belonging to those mesh
    physical groups are kept before plotting.  The group names are resolved
    to attribute values via :func:`resolve_physical_groups`.

    The color scale is clamped to the 98th percentile of the data by
    default to prevent outlier values (e.g. surface currents on
    conductors) from washing out the visualization.  Pass *clim* to
    override.

    Args:
        source: Simulation output directory, path to ``.pvtu`` / ``.vtu``,
            or a pre-loaded ``pv.DataSet``.
        field: Field name to plot (e.g. ``"E_real"``, ``"E_imag"``).
        normal: Slice normal (``"x"``, ``"y"``, ``"z"``).
        origin: Slice origin along the normal axis.
        physical_groups: Optional list of physical group names to
            restrict the plot to (e.g. ``["n_rib", "p_rib", "slab90"]``).
        cmap: Colour-map name passed to PyVista.
        clim: Optional ``(vmin, vmax)`` for the color scale.  If None,
            ``vmax`` is set to the 98th percentile of the data.
        title: Plot title.
        show_edges: Toggle mesh edges on/off.
        opacity: Opacity of the mesh (0-1).
        show: If True, display the plot (live view or static image); if
            False, the plotter is created and returned without rendering.
        interactive: If ``True``, force an interactive view: in a notebook
            this renders as a widget on the single shared trame server (so
            several views can coexist); otherwise it opens a blocking window.
            ``False`` forces a static PNG.  ``None`` (default) follows the
            session flag from ``gsim.viz.set_interactive_mode`` — off by
            default, so plots render statically until it is enabled.
        screenshot: If given, save a screenshot to this path.
        mode: Rendering mode: ``"auto"`` (default), ``"live"`` or
            ``"static"``.  Overrides ``GSIM_VIZ_MODE`` for this call.
            Interactive views are off by default; enable them globally with
            ``gsim.viz.set_interactive_mode(True)``.  Close open views with
            ``gsim.viz.close_interactive_views()``.  The camera always faces
            the cross-section plane, including for interactive views.

    Returns:
        The ``pv.Plotter`` instance.
    """
    from gsim.viz import _apply_front_camera, _ensure_pyvista, _show_or_screenshot

    _ensure_pyvista()
    import pyvista as pv

    dataset = _source_to_dataset(
        source,
        excitation=excitation,
        cycle=cycle,
        boundary=boundary,
    )

    if physical_groups is not None:
        if not isinstance(source, (str, Path)):
            msg = "physical_groups requires a file path (not a DataSet or dict)"
            raise TypeError(msg)
        source_path = Path(source)
        output_dir = source_path if source_path.is_dir() else source_path.parent
        attribute_values = resolve_physical_groups(output_dir, physical_groups)

        if "attribute" not in dataset.cell_data:
            msg = (
                "Dataset has no cell_data['attribute'] "
                "— cannot filter by physical group"
            )
            raise ValueError(msg)
        attrs = np.asarray(dataset.cell_data["attribute"])
        keep = np.where(np.isin(attrs, np.asarray(attribute_values)))[0]
        if keep.size == 0:
            msg = (
                f"No cells found for physical_groups={physical_groups} "
                f"(attrs={attribute_values})"
            )
            raise ValueError(msg)
        logger.info(
            "Selected %d / %d cells for physical_groups=%s",
            len(keep),
            dataset.n_cells,
            physical_groups,
        )
        dataset = dataset.extract_cells(keep)

    # Slice to the cross-section plane.
    sliced, _used_normal, _axis_idx, _axes = _slice_plane(
        dataset,
        normal=normal,
        origin=origin,
    )

    if field not in sliced.point_data:
        available = list(sliced.point_data.keys())
        msg = f"Field '{field}' not found in dataset. Available: {available}"
        raise ValueError(msg)

    # Determine the 2D scalar to plot (magnitude for vector fields).
    raw = np.asarray(sliced.point_data[field])
    if raw.ndim == 2:
        scalars = np.linalg.norm(raw, axis=1)
        scalar_name = f"|{field}|"
        sliced.point_data[scalar_name] = scalars
    else:
        scalar_name = field

    # Clamp color scale to 98th percentile (3D-style), unless user overrides.
    if clim is not None:
        _clim = clim
    else:
        s = np.asarray(sliced.point_data[scalar_name])
        finite = s[np.isfinite(s)]
        if finite.size > 0:
            vmax = float(np.percentile(finite, 98))
            _clim = (0.0, vmax) if vmax > 0 else None
        else:
            _clim = None

    # Camera: face the 2D plane directly (always, not just for zoom).
    pl = pv.Plotter(window_size=[1200, 900])
    pl.add_mesh(
        sliced,
        scalars=scalar_name,
        cmap=cmap,
        clim=_clim,
        show_edges=show_edges,
        opacity=opacity,
        scalar_bar_args={"title": scalar_name, "vertical": True},
    )
    pl.add_title(title, font_size=12)

    _apply_front_camera(pl, np.asarray(sliced.points), _axis_idx)
    if screenshot is not None:
        pl.screenshot(str(screenshot))
        pl.close()
        return pl

    if show:
        _show_or_screenshot(pl, interactive=interactive, mode=mode, reset_camera=False)
    else:
        pl.close()
    return pl


def extract_streamplot_inputs_2d(
    source: str | Path | dict | pv.DataSet,
    *,
    field: str = "E_real",
    normal: Axis = "x",
    origin: float = 0.0,
    excitation: int = 1,
    cycle: int | None = None,
    boundary: bool = False,
    streamplot_density: float = 1.2,
    streamplot_normalize: bool = True,
    streamplot_seed_from_field: bool = True,
    streamplot_seed_frac: float = 0.1,
    streamplot_seed_stride: int = 2,
    streamplot_mask_weak: bool = True,
    streamplot_min_frac: float = 0.08,
    grid_resolution: tuple[int, int] = (180, 120),
) -> StreamplotInputs2D:
    """Extract streamplot arrays compatible with PalaceToolkit's workflow.

    Returns the regular-grid arrays used for streamline tracing:
    ``x``, ``y``, ``u``, ``v`` and optional ``start_points``.
    """
    dataset = _source_to_dataset(
        source,
        excitation=excitation,
        cycle=cycle,
        boundary=boundary,
    )
    planar, used_normal, normal_idx, axes = _slice_plane(
        dataset,
        normal=normal,
        origin=origin,
    )

    if field not in planar.point_data:
        available = list(planar.point_data.keys())
        msg = f"Field '{field}' not found. Available: {available}"
        raise ValueError(msg)

    probe, H, V = _plane_grid(
        planar,
        normal=used_normal,
        origin=origin,
        grid_resolution=grid_resolution,
    )
    sampled = probe.sample(planar, snap_to_closest_point=True)

    vec_c = _complex_vector(sampled, field)
    if vec_c.ndim != 2 or vec_c.shape[1] not in {2, 3}:
        msg = f"Field '{field}' must be a 2- or 3-component vector, got {vec_c.shape}."
        raise ValueError(msg)

    i_h, i_v = axes
    if vec_c.shape[1] == 3:
        inplane_h_c = vec_c[:, i_h]
        inplane_v_c = vec_c[:, i_v]
        normal_comp_c = vec_c[:, normal_idx]
    else:
        inplane_h_c = vec_c[:, 0]
        inplane_v_c = vec_c[:, 1]
        normal_comp_c = _complex_normal_component(sampled, field, vec_c.shape[0])

    inplane_h, inplane_v = _phase_lock_inplane(inplane_h_c, inplane_v_c)

    et = np.sqrt(np.abs(inplane_h_c) ** 2 + np.abs(inplane_v_c) ** 2)
    en = np.abs(normal_comp_c)

    n_v, n_h = H.shape
    u_grid = inplane_h.reshape((n_v, n_h), order="F")
    v_grid = inplane_v.reshape((n_v, n_h), order="F")

    if streamplot_mask_weak:
        mag = np.sqrt(u_grid**2 + v_grid**2)
        ref = float(np.nanmax(mag)) if np.any(np.isfinite(mag)) else 0.0
        if ref > 0:
            weak = mag < (streamplot_min_frac * ref)
            u_grid = np.where(weak, np.nan, u_grid)
            v_grid = np.where(weak, np.nan, v_grid)

    if streamplot_normalize:
        mag = np.sqrt(u_grid**2 + v_grid**2)
        u_grid = u_grid / (mag + 1e-14)
        v_grid = v_grid / (mag + 1e-14)

    if streamplot_seed_from_field:
        mag = np.sqrt(u_grid**2 + v_grid**2)
        finite_counts = np.sum(np.isfinite(mag), axis=1)
        y_profile = np.divide(
            np.nansum(mag, axis=1),
            finite_counts,
            out=np.full(mag.shape[0], np.nan, dtype=float),
            where=finite_counts > 0,
        )
        if np.any(np.isfinite(y_profile)):
            iy = int(np.nanargmax(y_profile))
            mline = mag[iy, :]
            mmax = float(np.nanmax(mline)) if np.any(np.isfinite(mline)) else 0.0
            if np.isfinite(mmax) and mmax > 0:
                mask = mline >= (streamplot_seed_frac * mmax)
                idx = np.where(mask)[0][:: max(1, int(streamplot_seed_stride))]
                if idx.size >= 2:
                    start_points = np.column_stack([H[iy, idx], V[iy, idx]])
                else:
                    start_points = np.empty((0, 2), dtype=float)
            else:
                start_points = np.empty((0, 2), dtype=float)
        else:
            start_points = np.empty((0, 2), dtype=float)
    else:
        stride = max(2, round(14 / max(streamplot_density, 0.2)))
        start_points = np.column_stack(
            [H[::stride, ::stride].ravel(), V[::stride, ::stride].ravel()]
        )

    return StreamplotInputs2D(
        x=H[0, :],
        y=V[:, 0],
        u=np.nan_to_num(u_grid, nan=0.0),
        v=np.nan_to_num(v_grid, nan=0.0),
        start_points=start_points,
        et_mag=et.reshape((n_v, n_h), order="F"),
        en_mag=en.reshape((n_v, n_h), order="F"),
        normal=used_normal,
    )
