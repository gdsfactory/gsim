"""Open3D / Plotly-based 3D rendering for GeometryModel.

Provides Jupyter-compatible interactive visualisation via Open3D meshes
displayed through Plotly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gsim.common.geometry_model import GeometryModel, Prism
from gsim.common.viz._colors import generate_layer_colors_with_opacity
from gsim.common.viz._mesh_helpers import (
    prism_base_top_vertices,
    simulation_box_corners,
)

try:
    import open3d as o3d

    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_prisms_3d_open3d(
    geometry_model: GeometryModel,
    *,
    show_edges: bool = False,
    color_by_layer: bool = True,
    show_simulation_box: bool = True,
    notebook: bool = True,
    layer_opacity: dict[str, float] | None = None,
    **kwargs: Any,
) -> None:
    """Create interactive 3D visualisation using Open3D + Plotly.

    Args:
        geometry_model: GeometryModel containing prisms and bbox.
        show_edges: Show wireframe edges.
        color_by_layer: Colour each layer differently.
        show_simulation_box: Draw the simulation box.
        notebook: Display inside Jupyter notebook.
        layer_opacity: Per-layer opacity override (default: core=1.0, else 0.2).
        **kwargs: Extra Plotly figure options.
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("Open3D is required. Install with: pip install open3d")

    try:
        import plotly.graph_objects as go
    except ImportError as err:
        raise ImportError(
            "Plotly is required for Open3D notebook visualization. "
            "Install with: pip install plotly"
        ) from err

    layer_meshes = _convert_prisms_to_open3d(geometry_model)
    colors, opacity_dict = generate_layer_colors_with_opacity(
        list(layer_meshes.keys()), layer_opacity
    )

    plotly_meshes: list[Any] = []

    for layer_name, meshes in layer_meshes.items():
        layer_color = colors[layer_name] if color_by_layer else [0.7, 0.7, 0.7]
        layer_opacity_val = opacity_dict.get(layer_name, 0.8)

        for i, mesh in enumerate(meshes):
            if color_by_layer:
                mesh.paint_uniform_color(layer_color[:3])

            plotly_meshes.append(
                _mesh_to_mesh3d(
                    mesh,
                    opacity=layer_opacity_val,
                    name=f"{layer_name}_{i}",
                    color=layer_color[:3] if color_by_layer else [0.7, 0.7, 0.7],
                )
            )

            if show_edges:
                plotly_meshes.append(
                    _wireframe_to_scatter3d(mesh, name=f"{layer_name}_edges_{i}")
                )

    if show_simulation_box:
        plotly_meshes.append(_create_simulation_box_plotly(geometry_model))

    fig = go.Figure(data=plotly_meshes)

    # Compute axis ranges for zoom
    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []
    for meshes in layer_meshes.values():
        for mesh in meshes:
            verts = np.asarray(mesh.vertices)
            all_x.extend(verts[:, 0])
            all_y.extend(verts[:, 1])
            all_z.extend(verts[:, 2])

    if all_x:
        cx = np.mean([min(all_x), max(all_x)])
        cy = np.mean([min(all_y), max(all_y)])
        cz = np.mean([min(all_z), max(all_z)])
        rs = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            max(all_z) - min(all_z),
        )
    else:
        cx = cy = cz = 0.0
        rs = 10.0

    fig.update_layout(
        scene=dict(
            xaxis_title="X (um)",
            yaxis_title="Y (um)",
            zaxis_title="Z (um)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                center=dict(x=0, y=0, z=0),
                projection=dict(type="perspective"),
            ),
            xaxis=dict(range=[cx - rs * 0.6, cx + rs * 0.6]),
            yaxis=dict(range=[cy - rs * 0.6, cy + rs * 0.6]),
            zaxis=dict(range=[cz - rs * 0.6, cz + rs * 0.6]),
        ),
        title="3D Geometry Visualization",
        dragmode="orbit",
        **kwargs,
    )

    config = {
        "scrollZoom": True,
        "doubleClick": "reset+autosize",
        "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
        "displayModeBar": True,
        "responsive": True,
    }

    if notebook:
        fig.show(config=config)
    else:
        fig.write_html("geometry_3d.html", config=config)
        import webbrowser

        webbrowser.open("geometry_3d.html")


# ---------------------------------------------------------------------------
# Open3D mesh conversion
# ---------------------------------------------------------------------------


def _convert_prisms_to_open3d(
    geometry_model: GeometryModel,
) -> dict[str, list[Any]]:
    """Convert generic Prisms to Open3D meshes, with triangle-merge optimisation."""
    layer_meshes: dict[str, list[Any]] = {}

    for layer_name, prisms in geometry_model.prisms.items():
        if not prisms:
            continue

        triangular_count = sum(1 for p in prisms if len(p.vertices) == 3)

        if triangular_count > 100:
            triangular = [p for p in prisms if len(p.vertices) == 3]
            non_triangular = [p for p in prisms if len(p.vertices) != 3]

            meshes: list[Any] = []
            if triangular:
                merged = _merge_triangular_prisms_o3d(triangular)
                if merged is not None:
                    meshes.append(merged)
            for prism in non_triangular:
                base, top = prism_base_top_vertices(prism)
                meshes.append(_create_prism_mesh_o3d(base, top, prism))
        else:
            meshes = []
            for prism in prisms:
                base, top = prism_base_top_vertices(prism)
                meshes.append(_create_prism_mesh_o3d(base, top, prism))

        layer_meshes[layer_name] = meshes

    return layer_meshes


def _merge_triangular_prisms_o3d(prisms: list[Prism]) -> Any:
    """Merge many triangular prisms into one Open3D mesh for performance."""
    all_vertices: list[np.ndarray] = []
    all_triangles: list[list[int]] = []
    offset = 0

    for prism in prisms:
        base, top = prism_base_top_vertices(prism)
        verts = np.vstack([base, top])
        all_vertices.append(verts)
        n = len(base)

        # bottom
        all_triangles.append([offset, offset + 2, offset + 1])
        # top
        all_triangles.append([offset + n, offset + n + 1, offset + n + 2])
        # sides (2 tris per quad)
        for i in range(n):
            ni = (i + 1) % n
            all_triangles.append([offset + i, offset + ni, offset + ni + n])
            all_triangles.append([offset + i, offset + ni + n, offset + i + n])

        offset += len(verts)

    if all_vertices:
        combined = np.vstack(all_vertices)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(combined)
        mesh.triangles = o3d.utility.Vector3iVector(all_triangles)
        mesh.compute_vertex_normals()
        return mesh
    return None


def _create_prism_mesh_o3d(
    base_vertices: np.ndarray,
    top_vertices: np.ndarray,
    prism: Prism | None = None,
) -> Any:
    """Create Open3D mesh from base/top 3D vertices, with hole support."""
    if (
        prism is not None
        and prism.original_polygon is not None
        and hasattr(prism.original_polygon, "interiors")
        and prism.original_polygon.interiors
    ):
        return _create_prism_mesh_with_holes_o3d(
            prism.original_polygon, base_vertices, top_vertices
        )

    n = len(base_vertices)
    all_verts = np.vstack([base_vertices, top_vertices])
    # bottom (fan)
    faces: list[list[int]] = [[0, i + 1, i] for i in range(1, n - 1)]
    # top (fan)
    faces.extend([n, n + i, n + i + 1] for i in range(1, n - 1))
    # sides
    for i in range(n):
        ni = (i + 1) % n
        faces.append([i, ni, ni + n])
        faces.append([i, ni + n, i + n])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(all_verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    return mesh


def _create_prism_mesh_with_holes_o3d(
    shapely_polygon: Any,
    base_vertices: np.ndarray,
    top_vertices: np.ndarray,
) -> Any:
    """Create Open3D mesh from a Shapely polygon with holes via Delaunay."""
    try:
        import shapely.geometry as sg
        from scipy.spatial import Delaunay
    except ImportError:
        return _create_prism_mesh_o3d(base_vertices, top_vertices)

    all_points: list[tuple[float, float]] = []
    boundary_segments: list[list[int]] = []

    exterior_coords = list(shapely_polygon.exterior.coords[:-1])
    start_idx = 0
    all_points.extend(exterior_coords)
    boundary_segments = [
        [start_idx + i, start_idx + (i + 1) % len(exterior_coords)]
        for i in range(len(exterior_coords))
    ]

    for interior in shapely_polygon.interiors:
        interior_coords = list(interior.coords[:-1])
        start_idx = len(all_points)
        all_points.extend(interior_coords)
        boundary_segments.extend(
            [start_idx + i, start_idx + (i + 1) % len(interior_coords)]
            for i in range(len(interior_coords))
        )

    if len(all_points) < 3:
        return _create_prism_mesh_o3d(base_vertices, top_vertices)

    points_2d = np.array(all_points)
    tri = Delaunay(points_2d)

    valid_triangles = [
        simplex
        for simplex in tri.simplices
        if shapely_polygon.contains(sg.Point(*np.mean(points_2d[simplex], axis=0)))
    ]

    if not valid_triangles:
        return _create_prism_mesh_o3d(base_vertices, top_vertices)

    z_base = base_vertices[0, 2] if len(base_vertices) > 0 else 0
    z_top = top_vertices[0, 2] if len(top_vertices) > 0 else z_base + 1

    verts_3d: list[list[float]] = [[pt[0], pt[1], z_base] for pt in points_2d] + [
        [pt[0], pt[1], z_top] for pt in points_2d
    ]

    n_pts = len(points_2d)
    faces_3d: list[list[int]] = [
        [tri_idx[0], tri_idx[1], tri_idx[2]] for tri_idx in valid_triangles
    ]
    faces_3d.extend(
        [tri_idx[0] + n_pts, tri_idx[2] + n_pts, tri_idx[1] + n_pts]
        for tri_idx in valid_triangles
    )
    for seg in boundary_segments:
        i, j = seg
        faces_3d.append([i, j, j + n_pts])
        faces_3d.append([i, j + n_pts, i + n_pts])

    try:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts_3d)
        mesh.triangles = o3d.utility.Vector3iVector(faces_3d)
        mesh.compute_vertex_normals()
    except Exception:
        return _create_prism_mesh_o3d(base_vertices, top_vertices)
    else:
        return mesh


# ---------------------------------------------------------------------------
# Plotly helpers
# ---------------------------------------------------------------------------


def _mesh_to_mesh3d(
    mesh: Any,
    opacity: float = 1.0,
    name: str = "",
    color: list[float] | None = None,
) -> Any:
    """Convert Open3D mesh to Plotly Mesh3d."""
    import plotly.graph_objects as go

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)

    if color is not None:
        c = (np.array(color) * 255).astype(int)
        color_str = f"rgb({c[0]},{c[1]},{c[2]})"
    elif len(mesh.vertex_colors):
        c = (np.asarray(mesh.vertex_colors)[0] * 255).astype(int)
        color_str = f"rgb({c[0]},{c[1]},{c[2]})"
    else:
        color_str = "rgb(180,180,180)"

    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=tris[:, 0],
        j=tris[:, 1],
        k=tris[:, 2],
        color=color_str,
        opacity=opacity,
        name=name,
        showlegend=bool(name),
    )


def _wireframe_to_scatter3d(mesh: Any, name: str = "") -> Any:
    """Convert Open3D mesh edges to Plotly Scatter3d."""
    import plotly.graph_objects as go

    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    points = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)

    x_lines: list[float | None] = []
    y_lines: list[float | None] = []
    z_lines: list[float | None] = []

    for line in lines:
        p1, p2 = points[line[0]], points[line[1]]
        x_lines.extend([p1[0], p2[0], None])
        y_lines.extend([p1[1], p2[1], None])
        z_lines.extend([p1[2], p2[2], None])

    return go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        line=dict(color="black", width=2),
        name=name,
        showlegend=bool(name),
    )


def _create_simulation_box_plotly(geometry_model: GeometryModel) -> Any:
    """Create a Plotly Scatter3d wireframe box for the simulation bbox."""
    import plotly.graph_objects as go

    points, lines = simulation_box_corners(geometry_model)

    x_lines: list[float | None] = []
    y_lines: list[float | None] = []
    z_lines: list[float | None] = []

    for line in lines:
        p1, p2 = points[line[0]], points[line[1]]
        x_lines.extend([p1[0], p2[0], None])
        y_lines.extend([p1[1], p2[1], None])
        z_lines.extend([p1[2], p2[2], None])

    return go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        line=dict(color="black", width=2, dash="dot"),
        name="Simulation Box",
        showlegend=True,
    )
