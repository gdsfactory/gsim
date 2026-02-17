"""PyVista-based 3D rendering for GeometryModel.

Provides desktop-quality interactive visualisation using the PyVista library.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gsim.common.geometry_model import GeometryModel, Prism
from gsim.common.viz._colors import generate_layer_colors
from gsim.common.viz._mesh_helpers import (
    collect_triangular_prism_geometry,
    prism_base_top_vertices,
    triangulate_polygon_with_holes,
)

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_prisms_3d(
    geometry_model: GeometryModel,
    *,
    show_edges: bool = True,
    opacity: float = 0.8,  # noqa: ARG001
    color_by_layer: bool = True,
    show_simulation_box: bool = True,
    camera_position: str | None = "isometric",
    notebook: bool = True,
    theme: str = "default",
    **kwargs: Any,
) -> Any | None:
    """Create interactive 3D visualisation of prisms using PyVista.

    Args:
        geometry_model: A GeometryModel with prisms and bbox.
        show_edges: Whether to show edges of the prisms.
        opacity: Base opacity (0.0-1.0). Core layer forced opaque.
        color_by_layer: Colour by layer name.
        show_simulation_box: Draw the simulation bounding box.
        camera_position: "isometric", "xy", "xz", "yz", or custom tuple.
        notebook: Whether running inside Jupyter.
        theme: PyVista theme ("default", "dark", "document").
        **kwargs: Extra args forwarded to ``pv.Plotter``.

    Returns:
        PyVista plotter object for further customisation.
    """
    if not PYVISTA_AVAILABLE:
        raise ImportError(
            "PyVista is required for 3D visualization. "
            "Install with: pip install pyvista"
        )

    plotter = (
        pv.Plotter(notebook=notebook, **kwargs) if notebook else pv.Plotter(**kwargs)
    )

    if theme == "dark":
        pv.set_plot_theme("dark")
    elif theme == "document":
        pv.set_plot_theme("document")

    layer_meshes = _convert_prisms_to_meshes(geometry_model)
    colors = generate_layer_colors(list(layer_meshes.keys()))

    for layer_name, meshes in layer_meshes.items():
        color = colors[layer_name] if color_by_layer else None
        layer_opacity = 1.0 if layer_name == "core" else 0.2

        for mesh in meshes:
            plotter.add_mesh(
                mesh,
                color=color,
                opacity=layer_opacity,
                show_edges=show_edges,
                label=layer_name,
                name=f"{layer_name}_{id(mesh)}",
            )

    if show_simulation_box:
        sim_box = _create_simulation_box_pv(geometry_model)
        plotter.add_mesh(
            sim_box,
            style="wireframe",
            color="black",
            line_width=2,
            label="Simulation Box",
        )

    _set_camera_position(plotter, camera_position)

    if color_by_layer and len(layer_meshes) > 1:
        try:  # noqa: SIM105
            plotter.add_legend()
        except Exception:
            pass

    if notebook:
        try:
            pv.set_jupyter_backend("trame")
            return plotter.show()
        except Exception:
            pv.set_jupyter_backend("static")
            return plotter.show()
    else:
        return plotter.show()


def export_3d_mesh(
    geometry_model: GeometryModel,
    filename: str,
    fmt: str = "auto",
) -> None:
    """Export 3D geometry to mesh file (STL, PLY, OBJ, VTK, glTF)."""
    if not PYVISTA_AVAILABLE:
        raise ImportError(
            "PyVista is required for mesh export. Install with: pip install pyvista"
        )

    layer_meshes = _convert_prisms_to_meshes(geometry_model)
    combined = pv.MultiBlock()
    for layer_name, meshes in layer_meshes.items():
        for i, mesh in enumerate(meshes):
            combined[f"{layer_name}_{i}"] = mesh

    if fmt == "auto":
        fmt = filename.rsplit(".", 1)[-1].lower()

    if fmt == "stl":
        combined.combine().save(filename)
    elif fmt in ("ply", "obj", "vtk"):
        combined.save(filename)
    elif fmt == "gltf":
        try:
            combined.save(filename)
        except Exception as e:
            raise ValueError(
                f"glTF export failed. May need additional dependencies: {e}"
            ) from e
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def create_web_export(
    geometry_model: GeometryModel,
    filename: str = "geometry_3d.html",
    title: str = "3D Geometry Visualization",  # noqa: ARG001
) -> str:
    """Export 3D visualisation as standalone HTML (via PyVista)."""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for web export.")

    plotter = pv.Plotter(notebook=False, off_screen=True)
    layer_meshes = _convert_prisms_to_meshes(geometry_model)
    colors = generate_layer_colors(list(layer_meshes.keys()))

    for layer_name, meshes in layer_meshes.items():
        color = colors[layer_name]
        for mesh in meshes:
            plotter.add_mesh(mesh, color=color, opacity=0.8)

    plotter.export_html(filename)
    return filename


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _convert_prisms_to_meshes(
    geometry_model: GeometryModel,
) -> dict[str, list[Any]]:
    """Convert generic Prisms to PyVista meshes, with triangle-merge optimisation."""
    layer_meshes: dict[str, list[Any]] = {}

    for layer_name, prisms in geometry_model.prisms.items():
        triangular_count = sum(1 for p in prisms if len(p.vertices) == 3)

        if triangular_count > 100:
            triangular = [p for p in prisms if len(p.vertices) == 3]
            non_triangular = [p for p in prisms if len(p.vertices) != 3]

            meshes: list[Any] = []
            if triangular:
                merged = _merge_triangular_prisms(triangular)
                if merged is not None:
                    meshes.append(merged)

            for prism in non_triangular:
                base, top = prism_base_top_vertices(prism)
                meshes.append(_create_prism_mesh(base, top, prism))

            layer_meshes[layer_name] = meshes
        else:
            meshes = []
            for prism in prisms:
                base, top = prism_base_top_vertices(prism)
                meshes.append(_create_prism_mesh(base, top, prism))
            layer_meshes[layer_name] = meshes

    return layer_meshes


def _merge_triangular_prisms(prisms: list[Prism]) -> Any:
    """Merge many triangular prisms into one PyVista mesh for performance."""
    result = collect_triangular_prism_geometry(prisms)
    if result is None:
        return None

    combined, prism_offsets = result
    n = 3  # triangular prisms
    all_faces: list[int] = []

    for offset in prism_offsets:
        # bottom face
        all_faces.extend([3, offset + 2, offset + 1, offset + 0])
        # top face
        all_faces.extend([3, offset + n, offset + n + 1, offset + n + 2])
        # side quads
        for i in range(n):
            ni = (i + 1) % n
            all_faces.extend(
                [4, offset + i, offset + ni, offset + ni + n, offset + i + n]
            )

    return pv.PolyData(combined, all_faces)


def _create_prism_mesh(
    base_vertices: np.ndarray,
    top_vertices: np.ndarray,
    prism: Prism | None = None,
) -> Any:
    """Create a PyVista mesh from base/top 3D vertices, with hole/concave support."""
    if prism is not None and prism.original_polygon is not None:
        poly = prism.original_polygon
        has_holes = hasattr(poly, "interiors") and poly.interiors
        is_concave = (
            not has_holes
            and hasattr(poly, "convex_hull")
            and poly.convex_hull.area > poly.area * 1.001
        )
        if has_holes or is_concave:
            return _create_prism_mesh_with_holes(poly, base_vertices, top_vertices)

    n = len(base_vertices)
    all_verts = np.vstack([base_vertices, top_vertices])

    faces: list[int] = []
    # bottom
    faces.extend([n, *list(range(n))[::-1]])
    # top
    faces.extend([n, *[i + n for i in range(n)]])
    # sides
    for i in range(n):
        ni = (i + 1) % n
        faces.extend([4, i, ni, ni + n, i + n])

    return pv.PolyData(all_verts, faces)


def _create_prism_mesh_with_holes(
    shapely_polygon: Any,
    base_vertices: np.ndarray,
    top_vertices: np.ndarray,
) -> Any:
    """Create PyVista mesh from a Shapely polygon with holes using Delaunay."""
    z_base = base_vertices[0, 2] if len(base_vertices) > 0 else 0
    z_top = top_vertices[0, 2] if len(top_vertices) > 0 else z_base + 1

    result = triangulate_polygon_with_holes(shapely_polygon, z_base, z_top)
    if result is None:
        return _create_prism_mesh(base_vertices, top_vertices)

    verts_3d, valid_triangles, boundary_segments = result
    n_pts = len(verts_3d) // 2
    faces_pv: list[int] = []

    # bottom triangles
    for tri_idx in valid_triangles:
        faces_pv.extend([3, tri_idx[0], tri_idx[1], tri_idx[2]])
    # top triangles (reversed winding)
    for tri_idx in valid_triangles:
        faces_pv.extend([3, tri_idx[0] + n_pts, tri_idx[2] + n_pts, tri_idx[1] + n_pts])
    # side quads
    for seg in boundary_segments:
        i, j = seg
        faces_pv.extend([4, i, j, j + n_pts, i + n_pts])

    try:
        return pv.PolyData(verts_3d, faces_pv)
    except Exception:
        return _create_prism_mesh(base_vertices, top_vertices)


def _create_simulation_box_pv(geometry_model: GeometryModel) -> Any:
    """Create a wireframe PyVista box for the simulation bbox."""
    mn = geometry_model.bbox[0]
    mx = geometry_model.bbox[1]
    bounds = [mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]]
    return pv.Box(bounds=bounds)


def _set_camera_position(plotter: Any, position: str | None) -> None:
    """Configure camera position for optimal viewing."""
    if position == "isometric":
        plotter.camera_position = "iso"
    elif position == "xy":
        plotter.view_xy()
    elif position == "xz":
        plotter.view_xz()
    elif position == "yz":
        plotter.view_yz()
    elif isinstance(position, (tuple, list)) and len(position) == 3:
        plotter.camera_position = position
    else:
        plotter.camera_position = "iso"
    plotter.reset_camera()
