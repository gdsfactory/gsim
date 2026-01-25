"""Geometry submodule for FDTD simulations.

This module provides 3D geometry modeling and visualization capabilities.
"""

from gsim.fdtd.geometry.core import Geometry
from gsim.fdtd.geometry.render2d import plot_prism_slices
from gsim.fdtd.geometry.render3d import (
    create_web_export,
    export_3d_mesh,
    plot_prisms_3d,
    plot_prisms_3d_open3d,
    serve_threejs_visualization,
)

__all__ = [
    "Geometry",
    "create_web_export",
    "export_3d_mesh",
    "plot_prism_slices",
    "plot_prisms_3d",
    "plot_prisms_3d_open3d",
    "serve_threejs_visualization",
]
