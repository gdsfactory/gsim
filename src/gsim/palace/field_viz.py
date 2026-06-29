"""2D field visualization utilities for Palace ParaView outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyvista as pv

from gsim.palace.results import load_fields

logger = logging.getLogger(__name__)


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
    grid_resolution: tuple[int, int] = (360, 240),
    streamplot_density: float = 1.0,
    streamplot_linewidth: float = 0.9,
    streamplot_color: str = "lightskyblue",
    streamplot_show_arrows: bool = True,
    streamplot_normalize: bool = False,
    streamplot_seed_from_field: bool = True,
    streamplot_seed_frac: float = 0.2,
    streamplot_seed_stride: int = 2,
    streamplot_mask_weak: bool = False,
    streamplot_min_frac: float = 0.08,
    use_targeted_gap_seeds: bool = True,
    targeted_seed_offset: float = 8.0,
    streamplot_minlength: float = 0.1,
    streamplot_maxlength: float = 2.8,
    cmap: str = "hot",
    title: str = "In-plane |E_t|",
    figsize: tuple[float, float] = (10.0, 5.0),
    dpi: float = 140.0,
    show: bool = True,
) -> tuple[Any, Any, StreamplotInputs2D]:
    """Plot 2D in-plane field magnitude and streamlines using Matplotlib.

    This utility centralizes the notebook plotting workflow and returns
    ``(fig, ax, stream_inputs)`` for custom post-processing.
    """
    import matplotlib.pyplot as plt

    stream_inputs = extract_streamplot_inputs_2d(
        source,
        field=field,
        normal=normal,
        origin=origin,
        excitation=excitation,
        cycle=cycle,
        boundary=boundary,
        streamplot_density=streamplot_density,
        streamplot_normalize=streamplot_normalize,
        streamplot_seed_from_field=streamplot_seed_from_field,
        streamplot_seed_frac=streamplot_seed_frac,
        streamplot_seed_stride=streamplot_seed_stride,
        streamplot_mask_weak=streamplot_mask_weak,
        streamplot_min_frac=streamplot_min_frac,
        grid_resolution=grid_resolution,
    )

    x_grid, y_grid = np.meshgrid(stream_inputs.x, stream_inputs.y)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    et = np.asarray(stream_inputs.et_mag, dtype=float)
    im = ax.pcolormesh(x_grid, y_grid, et, cmap=cmap, shading="gouraud")

    start_points = stream_inputs.start_points
    if use_targeted_gap_seeds and start_points.shape[0] >= 2:
        start_points = np.vstack(
            [
                start_points + np.array([0.0, targeted_seed_offset]),
                start_points + np.array([0.0, -targeted_seed_offset]),
            ]
        )
    start_points = _filter_start_points_in_bounds(
        start_points,
        x=stream_inputs.x,
        y=stream_inputs.y,
    )

    stream_kwargs: dict[str, Any] = {
        "x": stream_inputs.x,
        "y": stream_inputs.y,
        "u": stream_inputs.u,
        "v": stream_inputs.v,
        "color": streamplot_color,
        "density": streamplot_density,
        "linewidth": streamplot_linewidth,
        "arrowsize": 1.0 if streamplot_show_arrows else 1e-6,
        "arrowstyle": "-|>" if streamplot_show_arrows else "-",
        "minlength": streamplot_minlength,
        "maxlength": streamplot_maxlength,
        "integration_direction": "both",
    }
    if start_points.shape[0] >= 2:
        stream_kwargs["start_points"] = start_points

    ax.streamplot(**stream_kwargs)
    ax.set_title(title)
    ax.set_aspect("equal")
    label_h, label_v = _plane_axis_labels(stream_inputs.normal)
    ax.set_xlabel(label_h)
    ax.set_ylabel(label_v)
    fig.colorbar(im, ax=ax, label="E_t")
    plt.tight_layout()
    if show:
        plt.show()
    return fig, ax, stream_inputs


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
