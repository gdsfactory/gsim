"""Common visualization utilities for simulation meshes and geometry models.

The renderers do **not** depend on a particular solver. Geometry renderers
accept a ``GeometryModel``; the interactive Gmsh viewer accepts a ``.msh``
file plus solver-specific labels and unit scaling.

3D backends:
    - Three.js (interactive Gmsh mesh viewer)
    - PyVista  (desktop, ``plot_prisms_3d``)
    - Open3D + Plotly  (Jupyter, ``plot_prisms_3d_open3d``)

2D backends:
    - matplotlib  (``plot_prism_slices``)
"""

from gsim.common.viz.gmsh import MeshViewer, plot_mesh_interactive
from gsim.common.viz.render2d import plot_prism_slices
from gsim.common.viz.render2d_interactive import plot_prism_slices_interactive
from gsim.common.viz.render3d import (
    create_web_export,
    export_3d_mesh,
    plot_prisms_3d,
    plot_prisms_3d_open3d,
)

__all__ = [
    "MeshViewer",
    "create_web_export",
    "export_3d_mesh",
    "plot_mesh_interactive",
    "plot_prism_slices",
    "plot_prism_slices_interactive",
    "plot_prisms_3d",
    "plot_prisms_3d_open3d",
]
