"""FDTD configuration for the shared interactive Gmsh viewer."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from gsim.common.viz.gmsh import Axis, MeshViewer, ViewerMode, plot_mesh_interactive


def plot_mesh(
    mesh_path: str | Path,
    *,
    mode: ViewerMode = "surface",
    axis: Axis = "z",
    position_um: float | None = None,
    show_groups: Sequence[str] | None = None,
    hide_groups: Sequence[str] = (),
    include_internal_groups: bool = False,
    zoom_to_cursor: bool = True,
    cell_size_nm: float | None = None,
    pml_cells: int | None = None,
    footer_title: str = "Simulation estimate",
    cell_count_label: str = "Estimated Yee cells",
    height: int = 600,
) -> MeshViewer:
    """Build an FDTD-configured interactive viewer for a Gmsh mesh."""
    return plot_mesh_interactive(
        mesh_path,
        mode=mode,
        axis=axis,
        position_um=position_um,
        show_groups=show_groups,
        hide_groups=hide_groups,
        include_internal_groups=include_internal_groups,
        zoom_to_cursor=zoom_to_cursor,
        mesh_unit_um=1e-3,
        title="GDSFactory FDTD",
        footer_title=footer_title,
        cell_count_label=cell_count_label,
        footer_tooltip=(
            "Estimated Yee-grid dimensions and cell count include PML cells "
            "on both sides of each axis."
        ),
        port_group_prefixes=("port_",),
        cell_size_nm=cell_size_nm,
        pml_cells=pml_cells,
        height=height,
    )


__all__ = ["Axis", "MeshViewer", "ViewerMode", "plot_mesh"]
