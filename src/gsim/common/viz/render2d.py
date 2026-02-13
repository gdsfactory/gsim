"""2D cross-sectional plotting for GeometryModel.

Provides matplotlib-based 2D slicing views (XY, XZ, YZ) of a generic
GeometryModel without any solver-specific imports.

When a ``SimOverlay`` is provided, the plot also draws:
- Simulation cell boundary (dashed black)
- PML regions (semi-transparent orange)
- Port markers (source in red, monitors in blue)
"""

from __future__ import annotations

from typing import Any

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
    *,
    overlay: Any | None = None,
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
        overlay: Optional SimOverlay with sim cell / PML / port metadata.

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
                geometry_model,
                x=x_val,
                y=None,
                z=None,
                ax=ax,
                legend=legend,
                overlay=overlay,
            )
        if slice_axis == "y":
            y_val = y if y is not None else "core"
            return _plot_single_prism_slice(
                geometry_model,
                x=None,
                y=y_val,
                z=None,
                ax=ax,
                legend=legend,
                overlay=overlay,
            )
        if slice_axis == "z":
            return _plot_single_prism_slice(
                geometry_model,
                x=None,
                y=None,
                z=z,
                ax=ax,
                legend=legend,
                overlay=overlay,
            )

    _plot_multi_view(
        geometry_model,
        slices_to_plot,
        x,
        y,
        z,
        show_legend=legend,
        overlay=overlay,
    )
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
    overlay: Any | None = None,
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
                geometry_model,
                x=x_val,
                y=None,
                z=None,
                ax=ax_i,
                legend=False,
                overlay=overlay,
            )
        elif slice_axis == "y":
            y_val = y if y is not None else "core"
            _plot_single_prism_slice(
                geometry_model,
                x=None,
                y=y_val,
                z=None,
                ax=ax_i,
                legend=False,
                overlay=overlay,
            )
        elif slice_axis == "z":
            _plot_single_prism_slice(
                geometry_model,
                x=None,
                y=None,
                z=z,
                ax=ax_i,
                legend=False,
                overlay=overlay,
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
    overlay: Any | None = None,
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
            plt.colormaps.get_cmap("tab10")(
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

    # Draw overlay or fallback geometry bbox
    if overlay is not None:
        _draw_overlay(ax, overlay, x=x, y=y, z=z)
        # Expand axis limits to include full sim cell
        if z is not None:
            xmin = min(xmin, overlay.cell_min[0])
            xmax = max(xmax, overlay.cell_max[0])
            ymin = min(ymin, overlay.cell_min[1])
            ymax = max(ymax, overlay.cell_max[1])
        elif x is not None:
            xmin = min(xmin, overlay.cell_min[1])
            xmax = max(xmax, overlay.cell_max[1])
            ymin = min(ymin, overlay.cell_min[2])
            ymax = max(ymax, overlay.cell_max[2])
        elif y is not None:
            xmin = min(xmin, overlay.cell_min[0])
            xmax = max(xmax, overlay.cell_max[0])
            ymin = min(ymin, overlay.cell_min[2])
            ymax = max(ymax, overlay.cell_max[2])
    else:
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


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

_PML_COLOR = (1.0, 0.6, 0.0, 0.20)  # semi-transparent orange
_PML_EDGE = (1.0, 0.6, 0.0, 0.50)
_SRC_COLOR = "red"
_MON_COLOR = "royalblue"

# Dielectric background colours — keyed by lowercase material name
_DIELECTRIC_COLORS: dict[str, tuple[float, float, float, float]] = {
    "sio2": (0.68, 0.85, 0.90, 0.25),  # light blue
    "silicon": (0.60, 0.60, 0.65, 0.30),  # warm grey
    "si": (0.60, 0.60, 0.65, 0.30),
    "air": (1.00, 1.00, 1.00, 0.08),  # near-transparent
}
_DIELECTRIC_DEFAULT_COLOR = (0.80, 0.80, 0.80, 0.20)  # light grey fallback
_DIELECTRIC_ZORDER = -10


def _draw_overlay(
    ax: plt.Axes,
    overlay: Any,
    *,
    x: float | None,
    y: float | None,
    z: float | None,
) -> None:
    """Draw dielectric backgrounds, cell boundary, PML regions, and port markers."""
    cmin = overlay.cell_min
    cmax = overlay.cell_max
    pml = overlay.dpml

    # Draw dielectric background slabs first (lowest zorder)
    dielectrics = getattr(overlay, "dielectrics", [])
    if dielectrics:
        if z is not None:
            _draw_dielectrics_xy(ax, dielectrics, z, cmin, cmax)
        elif x is not None:
            _draw_dielectrics_side(ax, dielectrics, cmin[1], cmax[1], axis="yz")
        elif y is not None:
            _draw_dielectrics_side(ax, dielectrics, cmin[0], cmax[0], axis="xz")

    if z is not None:
        _draw_overlay_xy(ax, cmin, cmax, pml, overlay.ports)
    elif x is not None:
        _draw_overlay_yz(ax, cmin, cmax, pml, overlay.ports, x)
    elif y is not None:
        _draw_overlay_xz(ax, cmin, cmax, pml, overlay.ports, y)


def _draw_overlay_xy(
    ax: plt.Axes,
    cmin: tuple[float, float, float],
    cmax: tuple[float, float, float],
    pml: float,
    ports: list,
) -> None:
    """Draw overlay elements for an XY (z-slice) view."""
    x0, y0 = cmin[0], cmin[1]
    x1, y1 = cmax[0], cmax[1]
    w, h = x1 - x0, y1 - y0

    # Sim cell boundary
    ax.add_patch(
        Rectangle(
            (x0, y0),
            w,
            h,
            facecolor="none",
            edgecolor="k",
            linestyle="--",
            linewidth=1,
            label="Sim cell",
            zorder=90,
        )
    )

    # PML rectangles (4 edges)
    _add_pml_rect(ax, x0, y0, pml, h, label="PML")  # left
    _add_pml_rect(ax, x1 - pml, y0, pml, h)  # right
    _add_pml_rect(ax, x0 + pml, y0, w - 2 * pml, pml)  # bottom
    _add_pml_rect(ax, x0 + pml, y1 - pml, w - 2 * pml, pml)  # top

    # Ports
    labeled: set[str] = set()
    for port in ports:
        cx, cy, _cz = port.center
        color = _SRC_COLOR if port.is_source else _MON_COLOR
        legend_key = "Source" if port.is_source else "Monitor"
        label = legend_key if legend_key not in labeled else None
        labeled.add(legend_key)
        hw = port.width / 2

        if port.normal_axis == 0:  # x-normal → vertical line
            ax.plot(
                [cx, cx],
                [cy - hw, cy + hw],
                color=color,
                linewidth=2,
                zorder=95,
                label=label,
            )
            if port.is_source:
                dx = 0.15 if port.direction == "+" else -0.15
                ax.annotate(
                    "",
                    xy=(cx + dx, cy),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    zorder=96,
                )
        else:  # y-normal → horizontal line
            ax.plot(
                [cx - hw, cx + hw],
                [cy, cy],
                color=color,
                linewidth=2,
                zorder=95,
                label=label,
            )
            if port.is_source:
                dy = 0.15 if port.direction == "+" else -0.15
                ax.annotate(
                    "",
                    xy=(cx, cy + dy),
                    xytext=(cx, cy),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    zorder=96,
                )

        ax.annotate(
            port.name,
            (cx, cy),
            fontsize=7,
            ha="center",
            va="bottom",
            color=color,
            zorder=96,
            xytext=(0, 4),
            textcoords="offset points",
        )


def _draw_overlay_yz(
    ax: plt.Axes,
    cmin: tuple[float, float, float],
    cmax: tuple[float, float, float],
    pml: float,
    ports: list,
    x_slice: float,
) -> None:
    """Draw overlay elements for a YZ (x-slice) view."""
    y0, z0 = cmin[1], cmin[2]
    y1, z1 = cmax[1], cmax[2]
    w, h = y1 - y0, z1 - z0

    # Sim cell boundary
    ax.add_patch(
        Rectangle(
            (y0, z0),
            w,
            h,
            facecolor="none",
            edgecolor="k",
            linestyle="--",
            linewidth=1,
            label="Sim cell",
            zorder=90,
        )
    )

    # PML rectangles
    _add_pml_rect(ax, y0, z0, pml, h, label="PML")  # left
    _add_pml_rect(ax, y1 - pml, z0, pml, h)  # right
    _add_pml_rect(ax, y0 + pml, z0, w - 2 * pml, pml)  # bottom
    _add_pml_rect(ax, y0 + pml, z1 - pml, w - 2 * pml, pml)  # top

    # Ports that intersect this x-slice (x-normal ports at x_slice)
    labeled: set[str] = set()
    for port in ports:
        cx, cy, cz = port.center
        if port.normal_axis == 0 and abs(cx - x_slice) < 0.01:
            color = _SRC_COLOR if port.is_source else _MON_COLOR
            legend_key = "Source" if port.is_source else "Monitor"
            label = legend_key if legend_key not in labeled else None
            labeled.add(legend_key)
            hw = port.width / 2
            hz = port.z_span / 2
            ax.add_patch(
                Rectangle(
                    (cy - hw, cz - hz),
                    port.width,
                    port.z_span,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.5,
                    label=label,
                    zorder=95,
                )
            )
            ax.annotate(
                port.name,
                (cy, cz + hz),
                fontsize=7,
                ha="center",
                va="bottom",
                color=color,
                zorder=96,
            )


def _draw_overlay_xz(
    ax: plt.Axes,
    cmin: tuple[float, float, float],
    cmax: tuple[float, float, float],
    pml: float,
    ports: list,
    y_slice: float,
) -> None:
    """Draw overlay elements for an XZ (y-slice) view."""
    x0, z0 = cmin[0], cmin[2]
    x1, z1 = cmax[0], cmax[2]
    w, h = x1 - x0, z1 - z0

    # Sim cell boundary
    ax.add_patch(
        Rectangle(
            (x0, z0),
            w,
            h,
            facecolor="none",
            edgecolor="k",
            linestyle="--",
            linewidth=1,
            label="Sim cell",
            zorder=90,
        )
    )

    # PML rectangles
    _add_pml_rect(ax, x0, z0, pml, h, label="PML")  # left
    _add_pml_rect(ax, x1 - pml, z0, pml, h)  # right
    _add_pml_rect(ax, x0 + pml, z0, w - 2 * pml, pml)  # bottom
    _add_pml_rect(ax, x0 + pml, z1 - pml, w - 2 * pml, pml)  # top

    # Ports that intersect this y-slice (y-normal ports at y_slice)
    labeled: set[str] = set()
    for port in ports:
        cx, cy, cz = port.center
        if port.normal_axis == 1 and abs(cy - y_slice) < 0.01:
            color = _SRC_COLOR if port.is_source else _MON_COLOR
            legend_key = "Source" if port.is_source else "Monitor"
            label = legend_key if legend_key not in labeled else None
            labeled.add(legend_key)
            hw = port.width / 2
            hz = port.z_span / 2
            ax.add_patch(
                Rectangle(
                    (cx - hw, cz - hz),
                    port.width,
                    port.z_span,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=1.5,
                    label=label,
                    zorder=95,
                )
            )
            ax.annotate(
                port.name,
                (cx, cz + hz),
                fontsize=7,
                ha="center",
                va="bottom",
                color=color,
                zorder=96,
            )


def _add_pml_rect(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str | None = None,
) -> None:
    """Add a single semi-transparent PML rectangle."""
    if w <= 0 or h <= 0:
        return
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor=_PML_COLOR,
            edgecolor=_PML_EDGE,
            linewidth=0.5,
            label=label,
            zorder=80,
        )
    )


# ---------------------------------------------------------------------------
# Dielectric background drawing
# ---------------------------------------------------------------------------


def _diel_color(material: str) -> tuple[float, float, float, float]:
    """Look up the fill colour for a dielectric material name."""
    return _DIELECTRIC_COLORS.get(material.lower(), _DIELECTRIC_DEFAULT_COLOR)


def _draw_dielectrics_side(
    ax: plt.Axes,
    dielectrics: list,
    h_min: float,
    h_max: float,
    axis: str = "xz",
) -> None:
    """Draw dielectric background bands for a side view (XZ or YZ).

    Each dielectric is a horizontal band spanning the full horizontal extent
    of the simulation cell at the dielectric's z-range.
    """
    labeled: set[str] = set()
    for diel in dielectrics:
        zmin, zmax = diel.zmin, diel.zmax
        if zmax <= zmin:
            continue
        color = _diel_color(diel.material)
        label = diel.name if diel.name not in labeled else None
        labeled.add(diel.name)
        ax.add_patch(
            Rectangle(
                (h_min, zmin),
                h_max - h_min,
                zmax - zmin,
                facecolor=color,
                edgecolor="none",
                label=label,
                zorder=_DIELECTRIC_ZORDER,
            )
        )


def _draw_dielectrics_xy(
    ax: plt.Axes,
    dielectrics: list,
    z_slice: float,
    cmin: tuple[float, float, float],
    cmax: tuple[float, float, float],
) -> None:
    """Draw dielectric background for an XY (z-slice) view.

    If the z-slice is within a dielectric slab, fill the entire cell with
    that material's colour.  If multiple slabs overlap, draw all of them
    (later ones on top).
    """
    labeled: set[str] = set()
    for diel in dielectrics:
        if not (diel.zmin <= z_slice <= diel.zmax):
            continue
        color = _diel_color(diel.material)
        label = diel.name if diel.name not in labeled else None
        labeled.add(diel.name)
        ax.add_patch(
            Rectangle(
                (cmin[0], cmin[1]),
                cmax[0] - cmin[0],
                cmax[1] - cmin[1],
                facecolor=color,
                edgecolor="none",
                label=label,
                zorder=_DIELECTRIC_ZORDER,
            )
        )
