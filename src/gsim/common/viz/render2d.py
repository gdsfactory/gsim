"""2D cross-sectional plotting for GeometryModel.

Provides matplotlib-based 2D slicing views (XY, XZ, YZ) of a generic
GeometryModel without any solver-specific imports.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

from gsim.common.geometry_model import GeometryModel

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_prism_slices(
    geometry_model: GeometryModel,
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str = "core",
    ax: plt.Axes | None = None,
    legend: bool = True,
    slices: str = "z",
) -> plt.Axes | None:
    """Plot cross sections of a GeometryModel with multi-view support.

    Args:
        geometry_model: GeometryModel with prisms and bbox.
        x: X-coordinate (or layer name) for the slice plane.
        y: Y-coordinate (or layer name) for the slice plane.
        z: Z-coordinate (or layer name) for the slice plane.
        ax: Axes to draw on.  If ``None``, a new figure is created.
        legend: Whether to show the legend.
        slices: Which slice(s) to plot -- "x", "y", "z", or combinations
            like "xy", "xz", "yz", "xyz".

    Returns:
        ``plt.Axes`` when *ax* was provided, otherwise ``None``
        (the figure is shown directly).
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
                geometry_model, x=x_val, y=None, z=None, ax=ax, legend=legend
            )
        if slice_axis == "y":
            y_val = y if y is not None else "core"
            return _plot_single_prism_slice(
                geometry_model, x=None, y=y_val, z=None, ax=ax, legend=legend
            )
        if slice_axis == "z":
            return _plot_single_prism_slice(
                geometry_model, x=None, y=None, z=z, ax=ax, legend=legend
            )

    _plot_multi_view(geometry_model, slices_to_plot, x, y, z, show_legend=legend)
    return None


# ---------------------------------------------------------------------------
# Multi-view helper
# ---------------------------------------------------------------------------


def _plot_multi_view(
    geometry_model: GeometryModel,
    slices_to_plot: list[str],
    x: float | str | None,
    y: float | str | None,
    z: float | str | None,
    show_legend: bool = True,
) -> None:
    """Create multi-view plot with a shared legend panel."""
    num_plots = len(slices_to_plot)
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(ncols=2, nrows=num_plots, width_ratios=(3, 1))

    axes: list[plt.Axes] = [fig.add_subplot(gs[i, 0]) for i in range(num_plots)]

    for ax_i, slice_axis in zip(axes, slices_to_plot, strict=True):
        if slice_axis == "x":
            x_val = x if x is not None else "core"
            _plot_single_prism_slice(
                geometry_model, x=x_val, y=None, z=None, ax=ax_i, legend=False
            )
        elif slice_axis == "y":
            y_val = y if y is not None else "core"
            _plot_single_prism_slice(
                geometry_model, x=None, y=y_val, z=None, ax=ax_i, legend=False
            )
        elif slice_axis == "z":
            _plot_single_prism_slice(
                geometry_model, x=None, y=None, z=z, ax=ax_i, legend=False
            )

    if show_legend:
        all_handles: list = []
        all_labels: list[str] = []
        seen: set[str] = set()

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels, strict=True):
                if label not in seen:
                    all_handles.append(handle)
                    all_labels.append(label)
                    seen.add(label)

        legend_row = num_plots // 2
        axl = fig.add_subplot(gs[legend_row, 1])
        if all_handles:
            axl.legend(all_handles, all_labels, loc="center")
        axl.axis("off")

    plt.show()


# ---------------------------------------------------------------------------
# Single-slice renderer
# ---------------------------------------------------------------------------


def _plot_single_prism_slice(
    geometry_model: GeometryModel,
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str | None = None,
    ax: plt.Axes | None = None,
    legend: bool = True,
) -> plt.Axes:
    """Plot a single cross-section using generic Prisms."""
    if ax is None:
        _, ax = plt.subplots()

    # Resolve layer-name shortcuts to coordinates
    x, y, z = (
        geometry_model.get_layer_center(c)[i] if isinstance(c, str) else c
        for i, c in enumerate((x, y, z))
    )

    active_count = sum([x is not None, y is not None, z is not None])
    if active_count != 1:
        raise ValueError("Specify exactly one of x, y, or z for the slice plane")

    # Layer colours
    layer_names = geometry_model.layer_names
    colors = dict(
        zip(
            layer_names,
            plt.colormaps.get_cmap("Spectral")(
                np.linspace(0, 1, max(len(layer_names), 1))
            ),
            strict=True,
        )
    )

    # Sort layers by zmin (descending) for drawing order -- mimic
    # the old sort_layers(..., sort_by="zmin", reverse=True)
    sorted_names = sorted(
        layer_names,
        key=lambda n: geometry_model.get_layer_bbox(n)[0][2],
        reverse=True,
    )

    # Mesh-order -> zorder mapping
    mesh_orders = np.unique(
        [geometry_model.layer_mesh_orders.get(n, 0) for n in sorted_names]
    )
    order_map = dict(zip(mesh_orders, range(0, -len(mesh_orders), -1), strict=True))

    xmin, xmax = np.inf, -np.inf
    ymin, ymax = np.inf, -np.inf

    # First pass: compute axis limits from layer bboxes
    for name in sorted_names:
        try:
            bbox = geometry_model.get_layer_bbox(name)
            if z is not None:
                xmin = min(xmin, bbox[0][0])
                xmax = max(xmax, bbox[1][0])
                ymin = min(ymin, bbox[0][1])
                ymax = max(ymax, bbox[1][1])
            elif x is not None:
                ymin = min(ymin, bbox[0][1])
                ymax = max(ymax, bbox[1][1])
            elif y is not None:
                xmin = min(xmin, bbox[0][0])
                xmax = max(xmax, bbox[1][0])
        except Exception:  # noqa: S112
            continue

    # Second pass: draw patches
    for name in sorted_names:
        color = colors.get(name, "lightgray")
        mesh_order = geometry_model.layer_mesh_orders.get(name, 0)
        bbox = geometry_model.get_layer_bbox(name)
        layer_zorder = order_map.get(mesh_order, 0)

        if z is not None:
            z_min_layer = bbox[0][2]
            z_max_layer = bbox[1][2]
            if not (z_min_layer <= z <= z_max_layer):
                continue

            if name in geometry_model.prisms:
                prisms = geometry_model.prisms[name]
                for idx, prism in enumerate(prisms):
                    if not (prism.z_base <= z <= prism.z_top):
                        continue

                    xy_points = prism.vertices.tolist()
                    patch = Polygon(
                        xy_points,
                        facecolor=color,
                        edgecolor="k",
                        linewidth=0.5,
                        label=name if idx == 0 else None,
                        zorder=layer_zorder,
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
                    zorder=layer_zorder,
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
                zorder=layer_zorder,
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
                zorder=layer_zorder,
            )
            ax.add_patch(rect)

    # Axis labels and simulation-box outline
    size = list(geometry_model.size)
    cmin = list(geometry_model.bbox[0])

    if z is not None:
        size = size[:2]
        cmin = cmin[:2]
        xlabel, ylabel = "x (um)", "y (um)"
        ax.set_title(f"XY cross section at z={z:.2f}")
    elif x is not None:
        size = [size[1], size[2]]
        cmin = [cmin[1], cmin[2]]
        xlabel, ylabel = "y (um)", "z (um)"
        ax.set_title(f"YZ cross section at x={x:.2f}")
        xmin, xmax = cmin[0], cmin[0] + size[0]
        ymin, ymax = cmin[1], cmin[1] + size[1]
    elif y is not None:
        size = [size[0], size[2]]
        cmin = [cmin[0], cmin[2]]
        xlabel, ylabel = "x (um)", "z (um)"
        ax.set_title(f"XZ cross section at y={y:.2f}")
        ymin, ymax = cmin[1], cmin[1] + size[1]

    sim_roi = Rectangle(
        tuple(cmin),  # type: ignore[arg-type]
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
