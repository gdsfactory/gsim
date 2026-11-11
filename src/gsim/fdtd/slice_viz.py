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
from matplotlib.collections import LineCollection, PolyCollection
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
    from shapely.geometry.base import BaseGeometry

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


def _merge_sections(polygons: Sequence[Any]) -> BaseGeometry:
    """Merge tetrahedron sections with a consistent intersection precision."""
    # Match the precision used by tetrahedron_section for shared intersections.
    return union_all([Polygon(vertices) for vertices in polygons], grid_size=1e-12)


def _material_paths(polygons: Sequence[Any]) -> list[MatplotlibPath]:
    """Merge mesh cells so only material boundaries are antialiased."""
    merged = _merge_sections(polygons)
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


def _plot_material_outlines(
    sections: Mapping[int, list[Any]],
    groups: Sequence[MeshGroup],
    lower: Any,
    upper: Any,
    *,
    axis: Axis,
    position_um: float,
    ax: Axes | None,
    aspect: Literal["equal", "auto"],
    figsize: tuple[float, float] | None,
) -> Axes:
    """Draw visible material interfaces once, preserving supplied field axes."""
    created_axes = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    covered = union_all([])
    visible_materials: dict[str, list[BaseGeometry]] = {}
    for group in reversed(groups):
        polygons = sections.get(group.physical_tag, [])
        if not polygons:
            continue
        region = _merge_sections(polygons)
        visible = region.difference(covered)
        if not visible.is_empty:
            visible_materials.setdefault(group.material, []).append(visible)
        covered = union_all([covered, region])
    # Merge touching regions of the same material, then deduplicate interfaces
    # shared by different materials. Hidden lower-priority edges stay hidden.
    boundaries = [union_all(regions).boundary for regions in visible_materials.values()]
    linework = union_all(boundaries)
    segments = [
        np.asarray(line.coords) for line in get_parts(linework) if not line.is_empty
    ]
    if segments:
        zorder = (
            max(
                (artist.get_zorder() for artist in [*ax.collections, *ax.images]),
                default=1,
            )
            + 1
        )
        ax.add_collection(
            LineCollection(
                segments,
                colors="#444444",
                linewidths=0.6,
                zorder=zorder,
                label="_nolegend_",
            ),
            autolim=False,
        )
    if created_axes:
        plane_axes = [index for index in range(3) if index != "xyz".index(axis)]
        ax.set(
            xlim=(lower[plane_axes[0]], upper[plane_axes[0]]),
            ylim=(lower[plane_axes[1]], upper[plane_axes[1]]),
            xlabel=f"{'xyz'[plane_axes[0]]} (µm)",
            ylabel=f"{'xyz'[plane_axes[1]]} (µm)",
            title=f"FDTD · {axis} = {position_um:g} µm",
            aspect=aspect,
        )
    return ax


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
    outline_only: bool = False,
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
    if outline_only and show_mesh:
        raise ValueError("outline_only=True cannot be combined with show_mesh=True")
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
    if outline_only:
        return _plot_material_outlines(
            sections,
            [
                group
                for group in groups
                if group.name not in hide_groups
                and (show_groups is None or group.name in show_groups)
            ],
            lower,
            upper,
            axis=axis,
            position_um=position_um,
            ax=ax,
            aspect=aspect,
            figsize=figsize,
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
