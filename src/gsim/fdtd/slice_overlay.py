"""Simulation-domain, PML, source, and monitor marks for static FDTD views."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.patches import Rectangle

from gsim.fdtd.api import (
    DipoleSource,
    GaussianBeamSource,
    LineCurrentSource,
    PortSource,
)
from gsim.fdtd.models import FDTDGeometryError
from gsim.fdtd.monitor_bounds import monitor_region_nm

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from gsim.fdtd.config import GaussianBeamConfig, PlaneMonitorConfig
    from gsim.fdtd.visualization import VisualizationMixin


def _region(
    ax: Axes,
    lower: Any,
    upper: Any,
    normal_axis: int,
    position: float,
    label: str,
    color: str,
) -> None:
    """Draw a plane, line, or point only where it intersects the requested cut."""
    if not lower[normal_axis] - 1e-9 <= position <= upper[normal_axis] + 1e-9:
        return
    plane_axes = [index for index in range(3) if index != normal_axis]
    start = np.asarray(lower)[plane_axes]
    end = np.asarray(upper)[plane_axes]
    size = end - start
    if np.all(size > 1e-9):
        ax.add_patch(
            Rectangle(
                (float(start[0]), float(start[1])),
                *size,
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                label=label,
                zorder=10,
            )
        )
    else:
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=color,
            linewidth=2,
            marker="o" if np.all(size < 1e-9) else None,
            label=label,
            zorder=10,
        )
    ax.annotate(
        label,
        (float(end[0]), float(end[1])),
        xytext=(3, 4),
        textcoords="offset points",
        color=color,
        fontsize=7,
        zorder=11,
    )


def pml_thickness_um(simulation: VisualizationMixin) -> np.ndarray:
    """Return per-axis PML thickness with the Yee-grid z alignment."""
    alignment = np.asarray([1, 1, 16])
    cells = simulation.domain.pml_cells
    aligned_cells = ((cells + alignment - 1) // alignment) * alignment
    return aligned_cells * simulation.solver.cell_size_nm / 1000


def draw_setup_overlay(
    ax: Axes,
    simulation: VisualizationMixin,
    lower: Any,
    upper: Any,
    normal_axis: int,
    position: float,
    show_groups: Sequence[str] | None,
    hide_groups: Sequence[str],
    vertical_ports: Mapping[str, tuple[GaussianBeamConfig, PlaneMonitorConfig]],
) -> None:
    """Show the physical domain and PML separately from material geometry."""
    plane_axes = [index for index in range(3) if index != normal_axis]
    start, end = lower[plane_axes], upper[plane_axes]
    size = end - start
    ax.add_patch(
        Rectangle(
            start,
            *size,
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            linewidth=1,
            label="Physical domain",
            zorder=8,
        )
    )
    pml_x, pml_y = pml_thickness_um(simulation)[plane_axes]
    if pml_x or pml_y:
        rectangles = [
            ((start[0] - pml_x, start[1] - pml_y), pml_x, size[1] + 2 * pml_y),
            ((end[0], start[1] - pml_y), pml_x, size[1] + 2 * pml_y),
            ((start[0], start[1] - pml_y), size[0], pml_y),
            ((start[0], end[1]), size[0], pml_y),
        ]
        for index, (origin, width, height) in enumerate(rectangles):
            patch = Rectangle(
                origin,
                width,
                height,
                facecolor="none",
                edgecolor="#999999",
                hatch="////",
                linewidth=0,
                label="PML" if index == 0 else None,
                zorder=7,
            )
            patch.set_hatch_linewidth(0.6)
            ax.add_patch(patch)
    for name, port in simulation.resolved.ports.items():
        if port.is_vertical:
            continue
        group_name = f"port_{name}"
        if group_name in hide_groups or (
            show_groups is not None and group_name not in show_groups
        ):
            continue
        port_axis = next(index for index, value in enumerate(port.normal) if value)
        width_axis = 1 - port_axis
        port_lower = np.asarray(port.center).copy()
        port_upper = port_lower.copy()
        port_lower[width_axis] -= port.width / 2
        port_upper[width_axis] += port.width / 2
        if port.layer_key is None:
            raise FDTDGeometryError(f"Guided port {name!r} has no resolved layer.")
        owners = [simulation.resolved.layers[key] for key in port.physical_layer_keys]
        port_lower[2], port_upper[2] = port.z_bounds or (
            min(layer.z_bounds[0] for layer in owners),
            max(layer.z_bounds[1] for layer in owners),
        )
        is_source = (
            isinstance(simulation.source, PortSource) and simulation.source.port == name
        )
        _region(
            ax,
            port_lower,
            port_upper,
            normal_axis,
            position,
            f"{'Source' if is_source else 'Monitor'} {name}",
            "#d62728" if is_source else "#1f77b4",
        )
    source = simulation.source
    if isinstance(source, DipoleSource):
        _region(
            ax,
            source.position_um,
            source.position_um,
            normal_axis,
            position,
            "Source",
            "#d62728",
        )
    elif isinstance(source, LineCurrentSource):
        source_lower = np.asarray(source.position_um).copy()
        source_upper = source_lower.copy()
        line_axis = "xyz".index(source.line_axis)
        source_lower[line_axis] -= source.length_um / 2
        source_upper[line_axis] += source.length_um / 2
        _region(
            ax, source_lower, source_upper, normal_axis, position, "Source", "#d62728"
        )
    elif isinstance(source, GaussianBeamSource):
        center, half_size = np.asarray(source.center_um), np.asarray(source.size_um) / 2
        _region(
            ax,
            center - half_size,
            center + half_size,
            normal_axis,
            position,
            "Source",
            "#d62728",
        )
    if isinstance(source, PortSource):
        for name, (beam, monitor) in vertical_ports.items():
            group_name = f"port_{name}"
            if group_name in hide_groups or (
                show_groups is not None and group_name not in show_groups
            ):
                continue
            is_source = source.port == name
            region = beam if is_source else monitor
            _region(
                ax,
                np.asarray(region.region_min) / 1000,
                np.asarray(region.region_max) / 1000,
                normal_axis,
                position,
                f"{'Source' if is_source else 'Monitor'} {name}",
                "#d62728" if is_source else "#1f77b4",
            )
    for monitor in simulation.monitors:
        monitor_lower, monitor_upper = monitor_region_nm(
            monitor,
            (
                lower[0] * 1000,
                lower[1] * 1000,
                lower[2] * 1000,
                upper[0] * 1000,
                upper[1] * 1000,
                upper[2] * 1000,
            ),
        )
        _region(
            ax,
            np.asarray(monitor_lower) / 1000,
            np.asarray(monitor_upper) / 1000,
            normal_axis,
            position,
            f"Monitor {monitor.name}",
            "#1f77b4",
        )
