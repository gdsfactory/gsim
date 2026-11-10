"""Static Matplotlib cross-sections of the FDTD transfer mesh."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import meshio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, Normalize

from gsim.common.viz._matplotlib import add_bottom_legend
from gsim.fdtd.mesh_sections import tetrahedron_section
from gsim.fdtd.slice_overlay import draw_setup_overlay, pml_thickness_um
from gsim.fdtd.viz import validate_slice

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from gsim.common.materials import MaterialSnapshot
    from gsim.fdtd.config import GaussianBeamConfig, PlaneMonitorConfig
    from gsim.fdtd.models import MeshManifest
    from gsim.fdtd.visualization import VisualizationMixin
    from gsim.fdtd.viz import Axis


def _mesh_sections(mesh_path: Path, axis: int, position: float) -> dict[int, list[Any]]:
    """Read nanometer mesh coordinates and collect sections by physical tag."""
    mesh = meshio.read(mesh_path)
    points = np.asarray(mesh.points, dtype=float) / 1000
    sections: dict[int, list[Any]] = {}
    physical_blocks = mesh.cell_data["gmsh:physical"]
    for block, tags in zip(mesh.cells, physical_blocks, strict=True):
        if block.type != "tetra":
            continue
        tetrahedra = points[block.data]
        coordinates = tetrahedra[:, :, axis]
        intersecting = (coordinates.min(axis=1) <= position + 1e-9) & (
            coordinates.max(axis=1) >= position - 1e-9
        )
        for vertices, tag in zip(
            tetrahedra[intersecting], tags[intersecting], strict=True
        ):
            polygon = tetrahedron_section(vertices, axis, position)
            if polygon is not None:
                sections.setdefault(int(tag), []).append(polygon)
    return sections


def plot_mesh_slice(
    mesh_path: Path,
    manifest: MeshManifest,
    materials: Mapping[str, MaterialSnapshot],
    simulation: VisualizationMixin,
    *,
    axis: Axis,
    position_um: float,
    show_mesh: bool = False,
    show_groups: Sequence[str] | None = None,
    hide_groups: Sequence[str] = (),
    ax: Axes | None = None,
    cmap: str = "Blues",
    legend: bool = True,
    aspect: Literal["equal", "auto"] = "equal",
    figsize: tuple[float, float] | None = None,
    vertical_ports: Mapping[str, tuple[GaussianBeamConfig, PlaneMonitorConfig]]
    | None = None,
) -> Axes:
    """Render priority-ordered material sections, index scale, and setup marks."""
    validate_slice(axis, position_um)
    if aspect not in {"equal", "auto"}:
        raise ValueError("aspect must be 'equal' or 'auto'")
    normal_axis = "xyz".index(axis)
    lower = np.asarray(manifest.domain_min_nm) / 1000
    upper = np.asarray(manifest.domain_max_nm) / 1000
    if not lower[normal_axis] <= position_um <= upper[normal_axis]:
        raise ValueError(
            f"Slice {axis}={position_um:g} µm is outside "
            f"[{lower[normal_axis]:g}, {upper[normal_axis]:g}] µm"
        )
    sections = _mesh_sections(mesh_path, normal_axis, position_um)
    groups = sorted(
        {**manifest.volumes, **manifest.layers}.values(),
        key=lambda group: group.priority,
    )
    values = [1.0, *(material.refractive_index for material in materials.values())]
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        minimum -= 0.05
        maximum += 0.05
    normalization = Normalize(vmin=minimum, vmax=maximum)
    colors = plt.colormaps.get_cmap(cmap).resampled(256)(np.linspace(0, 1, 256))
    if minimum == 1:
        colors[0] = (1, 1, 1, 1)
    colormap = ListedColormap(colors)
    created_figure = None
    if ax is None:
        created_figure, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    for group in groups:
        if group.name in hide_groups or (
            show_groups is not None and group.name not in show_groups
        ):
            continue
        polygons = sections.get(group.physical_tag, [])
        if not polygons:
            continue
        collection = PolyCollection(
            polygons,
            facecolors=[
                colormap(normalization(materials[group.material].refractive_index))
            ],
            edgecolors="#444444" if show_mesh else "none",
            linewidths=0.35 if show_mesh else 0,
            antialiaseds=show_mesh,
            label=f"{group.name} ({group.material})",
            # Rasterize the mesh so vector exports (SVG, PDF) stay small.
            rasterized=True,
        )
        ax.add_collection(collection)
    plane_axes = [index for index in range(3) if index != normal_axis]
    draw_setup_overlay(
        ax,
        simulation,
        lower,
        upper,
        normal_axis,
        position_um,
        show_groups,
        hide_groups,
        vertical_ports or {},
    )
    pml_x, pml_y = pml_thickness_um(simulation)[plane_axes]
    ax.set_xlim(lower[plane_axes[0]] - pml_x, upper[plane_axes[0]] + pml_x)
    ax.set_ylim(lower[plane_axes[1]] - pml_y, upper[plane_axes[1]] + pml_y)
    ax.set_xlabel(f"{'xyz'[plane_axes[0]]} (µm)")
    ax.set_ylabel(f"{'xyz'[plane_axes[1]]} (µm)")
    ax.set_title(f"FDTD · {axis} = {position_um:g} µm")
    ax.set_aspect(aspect)
    ax.figure.colorbar(
        ScalarMappable(norm=normalization, cmap=colormap),
        ax=ax,
        label=f"Refractive index n at \u03bb={simulation.source.wavelength_um:g} µm",
    )
    if legend:
        if created_figure is not None:
            add_bottom_legend(created_figure, ax)
        else:
            _, labels = ax.get_legend_handles_labels()
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.2),
                ncols=min(4, len(labels)),
                framealpha=1,
            )
    return ax
