"""Static Matplotlib cross-sections of the FDTD transfer mesh."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import meshio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.path import Path as MatplotlibPath
from shapely import get_parts, union_all
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from gsim.fdtd.mesh_sections import tetrahedron_section
from gsim.fdtd.slice_overlay import draw_setup_overlay, pml_thickness_um
from gsim.fdtd.viz import validate_slice

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from gsim.common.materials import MaterialSnapshot
    from gsim.fdtd.config import GaussianBeamConfig, PlaneMonitorConfig
    from gsim.fdtd.models import MeshGroup, MeshManifest
    from gsim.fdtd.visualization import VisualizationMixin
    from gsim.fdtd.viz import Axis


@dataclass
class _MaterialBatch:
    """Consecutive equally colored groups that can share one material outline."""

    groups: list[MeshGroup]
    polygons: list[Any]
    color: Any


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


def _material_paths(polygons: Sequence[Any]) -> list[MatplotlibPath]:
    """Merge mesh cells so only material boundaries are antialiased."""
    # Match the precision used by tetrahedron_section for shared intersections.
    merged = union_all([Polygon(vertices) for vertices in polygons], grid_size=1e-12)
    paths = []
    for polygon in get_parts(merged):
        if polygon.is_empty:
            continue
        # Opposite ring winding keeps holes transparent with nonzero path filling.
        polygon = orient(polygon, sign=1)
        rings = [polygon.exterior, *polygon.interiors]
        paths.append(
            MatplotlibPath.make_compound_path(
                *[
                    MatplotlibPath(np.asarray(ring.coords), closed=True)
                    for ring in rings
                ]
            )
        )
    return paths


def _add_slice_legend(
    ax: Axes,
    created_figure: Figure | None,
    group_handles: Mapping[PolyCollection, list[Patch]],
) -> None:
    """Retain each physical group's legend entry after combining its fill."""
    handles = []
    for handle in ax.get_legend_handles_labels()[0]:
        if isinstance(handle, PolyCollection) and handle in group_handles:
            handles.extend(group_handles[handle])
        else:
            handles.append(handle)
    if created_figure is not None:
        unique_handles = {}
        for handle in handles:
            unique_handles.setdefault(str(handle.get_label()), handle)
        if unique_handles:
            created_figure.legend(
                list(unique_handles.values()),
                list(unique_handles),
                loc="outside lower center",
                ncols=min(4, len(unique_handles)),
                fancybox=True,
                framealpha=1,
            )
    elif handles:
        ax.legend(
            handles,
            [str(handle.get_label()) for handle in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncols=min(4, len(handles)),
            framealpha=1,
        )


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
    batches: list[_MaterialBatch] = []
    for group in groups:
        if group.name in hide_groups or (
            show_groups is not None and group.name not in show_groups
        ):
            continue
        polygons = sections.get(group.physical_tag, [])
        if not polygons:
            continue
        color = colormap(normalization(materials[group.material].refractive_index))
        if not show_mesh and batches and batches[-1].color == color:
            batches[-1].groups.append(group)
            batches[-1].polygons.extend(polygons)
        else:
            batches.append(_MaterialBatch([group], list(polygons), color))
    group_handles = {}
    for batch in batches:
        labels = [f"{group.name} ({group.material})" for group in batch.groups]
        collection = PolyCollection(
            batch.polygons if show_mesh else [],
            facecolors=[batch.color],
            edgecolors="#444444" if show_mesh else "none",
            linewidths=0.35 if show_mesh else 0,
            antialiaseds=True,
            label=", ".join(labels),
            # Detailed mesh overlays stay compact; material outlines stay vector.
            rasterized=show_mesh,
        )
        if not show_mesh:
            # Shared edges must be antialiased once, including across groups.
            paths = _material_paths(batch.polygons)
            collection.set_verts_and_codes(
                [path.vertices for path in paths],
                # Matplotlib's stubs omit the outer sequence of code arrays.
                [path.codes for path in paths],  # ty: ignore[invalid-argument-type]
            )
        ax.add_collection(collection)
        group_handles[collection] = [
            Patch(
                facecolor=batch.color,
                edgecolor="#444444" if show_mesh else "none",
                linewidth=0.35 if show_mesh else 0,
                label=label,
            )
            for label in labels
        ]
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
        _add_slice_legend(ax, created_figure, group_handles)
    return ax
