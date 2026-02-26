"""Generic GMSH mesh generation for any LayerStack.

Provides solver-agnostic mesh generation from gdsfactory components
and LayerStack definitions. Used by Palace internally; also available
for standalone mesh inspection or other solvers.

Usage:
    from gsim.common.mesh import generate_mesh, GeometryData, MeshResult

    result = generate_mesh(component, stack, output_dir="./mesh_output")
"""

from __future__ import annotations

from gsim.common.mesh import gmsh_utils
from gsim.common.mesh.generator import collect_mesh_stats, generate_mesh
from gsim.common.mesh.geometry import (
    GeometryData,
    add_dielectrics,
    add_layer_volumes,
    extract_geometry,
    get_layer_info,
)
from gsim.common.mesh.types import MeshResult

__all__ = [
    "GeometryData",
    "MeshResult",
    "add_dielectrics",
    "add_layer_volumes",
    "collect_mesh_stats",
    "extract_geometry",
    "generate_mesh",
    "get_layer_info",
    "gmsh_utils",
]
