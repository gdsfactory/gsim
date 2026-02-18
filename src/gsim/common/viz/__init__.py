"""Common visualization utilities for 3D and 2D geometry rendering.

All functions accept a ``GeometryModel`` (from ``gsim.common.geometry_model``)
and do **not** depend on any solver (meep, tidy3d, etc.).

3D backends:
    - PyVista  (desktop, ``plot_prisms_3d``)
    - Open3D + Plotly  (Jupyter, ``plot_prisms_3d_open3d``)

2D backends:
    - matplotlib  (``plot_prism_slices``)
"""

from gsim.common.viz.render2d import plot_prism_slices
from gsim.common.viz.render3d import (
    create_web_export,
    export_3d_mesh,
    plot_prisms_3d,
    plot_prisms_3d_open3d,
)

__all__ = [
    "create_web_export",
    "export_3d_mesh",
    "plot_prism_slices",
    "plot_prisms_3d",
    "plot_prisms_3d_open3d",
]
