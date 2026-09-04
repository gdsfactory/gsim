"""Interactive Three.js views of GDSFactory FDTD Gmsh simulation meshes."""

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


def plot_mesh(
    mesh_path: str | Path,
    *,
    mode: ViewerMode = "surface",
    axis: Axis = "z",
    position_um: float | None = None,
    show_groups: Sequence[str] | None = None,
    hide_groups: Sequence[str] = (),
    zoom_to_cursor: bool = True,
    cell_size_nm: float | None = None,
    pml_cells: int | None = None,
    height: int = 600,
) -> MeshViewer:
    """Build an interactive GDSFactory FDTD viewer for a Gmsh mesh.

    The mesh is embedded in the returned HTML, so the source ``.msh`` file is
    no longer needed once this function returns. Three.js is loaded from the
    same public CDN used by the original GDSFactory FDTD viewer.
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
    if cell_size_nm is not None and (not isfinite(cell_size_nm) or cell_size_nm <= 0):
        raise ValueError("cell_size_nm must be positive and finite when provided")
    if pml_cells is not None and pml_cells < 0:
        raise ValueError("pml_cells must be nonnegative when provided")
    if height <= 0:
        raise ValueError("height must be positive")

    template = files("gsim.fdtd").joinpath("assets/mesh-viewer.html").read_text("utf8")
    options = {
        "axis": axis,
        "cellSizeNm": cell_size_nm,
        "hideGroups": list(hide_groups),
        "mode": mode,
        "pmlCells": pml_cells,
        "positionUm": position_um,
        "showGroups": None if show_groups is None else list(show_groups),
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


__all__ = ["MeshViewer", "plot_mesh"]
