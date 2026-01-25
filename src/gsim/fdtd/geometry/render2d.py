"""2D rendering utilities for FDTD geometry visualization.

This module provides 2D cross-sectional plotting capabilities for both
Tidy3D PolySlabs and MEEP Prisms, with support for multi-view layouts
and consistent legend placement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from gsim.fdtd.util import sort_layers

if TYPE_CHECKING:
    pass


def plot_prism_slices(
    geometry_obj,
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str = "core",
    ax: plt.Axes | None = None,
    legend: bool = True,
    slices: str = "z",
) -> plt.Axes | None:
    """Plot cross sections of MEEP prisms with multi-view support.

    Args:
        geometry_obj: Geometry object with meep_prisms property and helper methods
        x: The x-coordinate for the cross section. If str, uses layer name.
        y: The y-coordinate for the cross section. If str, uses layer name.
        z: The z-coordinate for the cross section. If str, uses layer name.
        ax: The Axes instance to plot on. If None, creates new figure.
        legend: Whether to include a legend in the plot.
        slices: Which slice(s) to plot. Can be single ("x", "y", "z") or any combination
            ("xy", "xz", "yz", "xyz", etc.).

    Returns:
        plt.Axes or None: Returns None when creating new figure (displays directly),
            returns Axes if ax was provided.
    """
    slices_to_plot = sorted(set(slices.lower()))
    if not all(s in "xyz" for s in slices_to_plot):
        raise ValueError(f"slices must only contain 'x', 'y', 'z'. Got: {slices}")

    if ax is not None:
        if len(slices_to_plot) > 1:
            raise ValueError("Cannot plot multiple slices when ax is provided")
        slice_axis = slices_to_plot[0]
        if slice_axis == "x":
            x_val = x if x is not None else "core"
            return _plot_single_prism_slice(
                geometry_obj, x=x_val, y=None, z=None, ax=ax, legend=legend
            )
        elif slice_axis == "y":
            y_val = y if y is not None else "core"
            return _plot_single_prism_slice(
                geometry_obj, x=None, y=y_val, z=None, ax=ax, legend=legend
            )
        elif slice_axis == "z":
            return _plot_single_prism_slice(
                geometry_obj, x=None, y=None, z=z, ax=ax, legend=legend
            )

    _plot_multi_view(geometry_obj, slices_to_plot, x, y, z, show_legend=legend)
    return None


def _plot_multi_view(
    geometry_obj,
    slices_to_plot: list[str],
    x: float | str | None,
    y: float | str | None,
    z: float | str | None,
    show_legend: bool = True,
) -> None:
    """Create multi-view plot with shared legend panel."""
    num_plots = len(slices_to_plot)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(ncols=2, nrows=num_plots, width_ratios=(3, 1))

    axes = []
    for i in range(num_plots):
        axes.append(fig.add_subplot(gs[i, 0]))

    for ax_i, slice_axis in zip(axes, slices_to_plot):
        if slice_axis == "x":
            x_val = x if x is not None else "core"
            _plot_single_prism_slice(
                geometry_obj, x=x_val, y=None, z=None, ax=ax_i, legend=False
            )
        elif slice_axis == "y":
            y_val = y if y is not None else "core"
            _plot_single_prism_slice(
                geometry_obj, x=None, y=y_val, z=None, ax=ax_i, legend=False
            )
        elif slice_axis == "z":
            _plot_single_prism_slice(
                geometry_obj, x=None, y=None, z=z, ax=ax_i, legend=False
            )

    if show_legend:
        all_handles = []
        all_labels = []
        seen_labels = set()

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                if label not in seen_labels:
                    all_handles.append(handle)
                    all_labels.append(label)
                    seen_labels.add(label)

        legend_row = num_plots // 2
        axl = fig.add_subplot(gs[legend_row, 1])
        if all_handles:
            axl.legend(all_handles, all_labels, loc="center")
        axl.axis("off")

    plt.show()


def _plot_single_prism_slice(
    geometry_obj,
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str | None = None,
    ax: plt.Axes | None = None,
    legend: bool = True,
) -> plt.Axes:
    """Plot a single cross section using MEEP prisms."""
    if ax is None:
        _, ax = plt.subplots()

    x, y, z = (
        geometry_obj.get_layer_center(c)[i] if isinstance(c, str) else c
        for i, c in enumerate((x, y, z))
    )

    slice_axis = sum([x is not None, y is not None, z is not None])
    if slice_axis != 1:
        raise ValueError("Specify exactly one of x, y, or z for the slice plane")

    colors = dict(
        zip(
            geometry_obj.meep_prisms.keys(),
            plt.colormaps.get_cmap("Spectral")(
                np.linspace(0, 1, len(geometry_obj.meep_prisms))
            ),
        )
    )

    layers = sort_layers(geometry_obj.geometry_layers, sort_by="zmin", reverse=True)

    meshorders = np.unique([v.mesh_order for v in layers.values()])
    order_map = dict(zip(meshorders, range(0, -len(meshorders), -1)))

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf

    for name, layer in layers.items():
        try:
            bbox = geometry_obj.get_layer_bbox(name)
            if z is not None:
                xmin, xmax = min(xmin, bbox[0][0]), max(xmax, bbox[1][0])
                ymin, ymax = min(ymin, bbox[0][1]), max(ymax, bbox[1][1])
            elif x is not None:
                ymin, ymax = min(ymin, bbox[0][1]), max(ymax, bbox[1][1])
            elif y is not None:
                xmin, xmax = min(xmin, bbox[0][0]), max(xmax, bbox[1][0])
        except Exception:
            continue

    for name, layer in layers.items():
        color = colors.get(name, "lightgray")

        bbox = geometry_obj.get_layer_bbox(name)

        if z is not None:
            z_min, z_max = bbox[0][2], bbox[1][2]
            if not (z_min <= z <= z_max):
                continue

            if name in geometry_obj.meep_prisms:
                prisms = geometry_obj.meep_prisms[name]
                for idx, prism in enumerate(prisms):
                    vertices_3d = prism.vertices
                    height = prism.height

                    z_base = vertices_3d[0].z
                    if not (z_base <= z <= z_base + height):
                        continue

                    xy_points = [(v.x, v.y) for v in vertices_3d]

                    patch = Polygon(
                        xy_points,
                        facecolor=color,
                        edgecolor="k",
                        linewidth=0.5,
                        label=name if idx == 0 else None,
                        zorder=order_map[layer.mesh_order],
                    )
                    ax.add_patch(patch)

            else:
                rect = Rectangle(
                    (bbox[0][0], bbox[0][1]),
                    bbox[1][0] - bbox[0][0],
                    bbox[1][1] - bbox[0][1],
                    facecolor=color,
                    edgecolor="k",
                    linewidth=0.5,
                    label=name,
                    zorder=order_map[layer.mesh_order],
                )
                ax.add_patch(rect)

        elif x is not None:
            rect = Rectangle(
                (bbox[0][1], bbox[0][2]),
                bbox[1][1] - bbox[0][1],
                bbox[1][2] - bbox[0][2],
                facecolor=color,
                edgecolor="k",
                linewidth=0.5,
                label=name,
                zorder=order_map[layer.mesh_order],
            )
            ax.add_patch(rect)

        elif y is not None:
            rect = Rectangle(
                (bbox[0][0], bbox[0][2]),
                bbox[1][0] - bbox[0][0],
                bbox[1][2] - bbox[0][2],
                facecolor=color,
                edgecolor="k",
                linewidth=0.5,
                label=name,
                zorder=order_map[layer.mesh_order],
            )
            ax.add_patch(rect)

    size = list(geometry_obj.size)
    cmin = list(geometry_obj.bbox[0])

    if z is not None:
        size = size[:2]
        cmin = cmin[:2]
        xlabel, ylabel = "x (μm)", "y (μm)"
        ax.set_title(f"XY cross section at z={z:.2f}")
    elif x is not None:
        size = [size[1], size[2]]
        cmin = [cmin[1], cmin[2]]
        xlabel, ylabel = "y (μm)", "z (μm)"
        ax.set_title(f"YZ cross section at x={x:.2f}")
        xmin, xmax = cmin[0], cmin[0] + size[0]
        ymin, ymax = cmin[1], cmin[1] + size[1]
    elif y is not None:
        size = [size[0], size[2]]
        cmin = [cmin[0], cmin[2]]
        xlabel, ylabel = "x (μm)", "z (μm)"
        ax.set_title(f"XZ cross section at y={y:.2f}")
        ymin, ymax = cmin[1], cmin[1] + size[1]

    sim_roi = Rectangle(
        cmin,
        *size,
        facecolor="none",
        edgecolor="k",
        linestyle="--",
        linewidth=1,
        label="Simulation",
    )
    ax.add_patch(sim_roi)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    if legend:
        ax.legend(fancybox=True, framealpha=1.0)

    return ax
