"""Interactive Three.js views of Gmsh simulation meshes."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from importlib.resources import files
from math import isfinite
from pathlib import Path
from typing import Literal

ViewerMode = Literal["surface", "mesh", "slice", "mesh_slice"]
Axis = Literal["x", "y", "z"]


@dataclass(frozen=True)
class MeshViewer:
    """A self-contained interactive mesh document displayable in notebooks."""

    html: str
    height: int = 600

    def _repr_html_(self) -> str:
        """Embed the viewer in an isolated notebook iframe."""
        source = escape(self.html, quote=True)
        return (
            f'<iframe srcdoc="{source}" width="100%" height="{self.height}" '
            'style="border:0;border-radius:8px" '
            'sandbox="allow-scripts"></iframe>'
        )

    def save(self, path: str | Path) -> Path:
        """Save the standalone viewer document and return its path."""
        output_path = Path(path)
        output_path.write_text(self.html, encoding="utf8")
        return output_path


def plot_mesh_interactive(
    mesh_path: str | Path,
    *,
    mode: ViewerMode = "surface",
    axis: Axis = "z",
    position_um: float | None = None,
    show_groups: Sequence[str] | None = None,
    hide_groups: Sequence[str] = (),
    zoom_to_cursor: bool = True,
    mesh_unit_um: float = 1.0,
    title: str = "GDSFactory Mesh",
    footer_title: str = "Mesh statistics",
    cell_count_label: str = "Mesh elements",
    footer_tooltip: str = "Number of rendered highest-dimensional mesh elements.",
    port_group_prefixes: Sequence[str] = (),
    cell_size_nm: float | None = None,
    pml_cells: int | None = None,
    height: int = 600,
) -> MeshViewer:
    """Build an interactive Three.js viewer for a Gmsh mesh.

    The mesh is embedded in the returned HTML, so the source ``.msh`` file is
    no longer needed once this function returns. Three.js is loaded from the
    same public CDN used by the original viewer.

    Args:
        mesh_path: ASCII Gmsh ``.msh`` file to embed.
        mode: Surface, mesh-edge, slice, or mesh-edge slice view.
        axis: Slice normal axis.
        position_um: Initial slice position in micrometers.
        show_groups: Physical groups to show initially, or all when omitted.
        hide_groups: Physical groups to hide initially.
        zoom_to_cursor: Zoom toward the pointer instead of the view center.
        mesh_unit_um: Micrometers represented by one mesh coordinate unit.
        title: Viewer title shown in the browser and control panel.
        footer_title: Heading for the statistics footer.
        cell_count_label: Solver-specific label for the cell or element count.
        footer_tooltip: Explanation shown for the statistics footer.
        port_group_prefixes: Name prefixes used to identify port groups.
        cell_size_nm: FDTD cell size used for an estimated Yee-grid count.
        pml_cells: PML cells added to each side of an estimated Yee grid.
        height: Notebook iframe height in pixels.
    """
    path = Path(mesh_path)
    if not path.is_file():
        raise FileNotFoundError(f"Gmsh mesh not found: {path}")
    if path.suffix.casefold() != ".msh":
        raise ValueError(f"Expected a .msh file, got {path.name!r}")
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")
    if position_um is not None and not isfinite(position_um):
        raise ValueError("position_um must be finite when provided")
    if not isfinite(mesh_unit_um) or mesh_unit_um <= 0:
        raise ValueError("mesh_unit_um must be positive and finite")
    for label, value in (
        ("title", title),
        ("footer_title", footer_title),
        ("cell_count_label", cell_count_label),
    ):
        if not value.strip():
            raise ValueError(f"{label} must not be empty")
    if cell_size_nm is not None and (not isfinite(cell_size_nm) or cell_size_nm <= 0):
        raise ValueError("cell_size_nm must be positive and finite when provided")
    if pml_cells is not None and pml_cells < 0:
        raise ValueError("pml_cells must be nonnegative when provided")
    if height <= 0:
        raise ValueError("height must be positive")

    template = (
        files("gsim.common.viz").joinpath("assets/mesh-viewer.html").read_text("utf8")
    )
    options = {
        "axis": axis,
        "cellCountLabel": cell_count_label,
        "cellSizeNm": cell_size_nm,
        "footerTitle": footer_title,
        "footerTooltip": footer_tooltip,
        "hideGroups": list(hide_groups),
        "meshUnitUm": mesh_unit_um,
        "mode": mode,
        "pmlCells": pml_cells,
        "portGroupPrefixes": list(port_group_prefixes),
        "positionUm": position_um,
        "showGroups": None if show_groups is None else list(show_groups),
        "title": title,
        "zoomToCursor": zoom_to_cursor,
    }
    html = template.replace(
        "__GSIM_MESH_DATA__", _script_json(path.read_text(encoding="utf8"))
    ).replace("__GSIM_VIEW_OPTIONS__", _script_json(options))
    return MeshViewer(html=html, height=height)


def _script_json(value: object) -> str:
    """Encode data for a script block without permitting tag termination."""
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


__all__ = ["Axis", "MeshViewer", "ViewerMode", "plot_mesh_interactive"]
