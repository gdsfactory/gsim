"""Unified 3D rendering API for GeometryModel.

Re-exports the public functions from the backend-specific modules so that
callers can simply do::

    from gsim.common.viz.render3d import plot_prisms_3d, plot_prisms_3d_open3d
"""

from gsim.common.viz.render3d_open3d import plot_prisms_3d_open3d
from gsim.common.viz.render3d_pyvista import (
    create_web_export,
    export_3d_mesh,
    plot_prisms_3d,
)

__all__ = [
    "create_web_export",
    "export_3d_mesh",
    "plot_prisms_3d",
    "plot_prisms_3d_open3d",
]
