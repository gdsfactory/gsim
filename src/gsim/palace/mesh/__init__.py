"""3D mesh generation for Palace EM simulation.

This module provides mesh generation directly from gdsfactory components
and palace-api data structures.

Usage:
    from gsim.palace.mesh import generate_mesh, MeshConfig

    # Quick presets
    config = MeshConfig.coarse()   # Fast iteration
    config = MeshConfig.default()  # Balanced
    config = MeshConfig.fine()     # High accuracy

    # Or customize with overrides
    config = MeshConfig.coarse(margin=100.0, fmax=50e9)

    # Or full manual control
    config = MeshConfig(refined_mesh_size=3.0, max_mesh_size=200.0)

    result = generate_mesh(
        component=c,
        stack=stack,
        ports=ports,
        output_dir="./sim_output",
        refined_mesh_size=config.refined_mesh_size,
        max_mesh_size=config.max_mesh_size,
        margin_x=config.effective_margin_x,
        margin_y=config.effective_margin_y,
        fmax=config.fmax,
    )
"""

from __future__ import annotations

from gsim.palace.mesh.generator import (
    GeometryData,
    MeshResult,
    generate_mesh,
    write_config,
)
from gsim.palace.models.mesh import MeshConfig

from . import gmsh_utils

__all__ = [
    "GeometryData",
    "MeshConfig",
    "MeshResult",
    "generate_mesh",
    "gmsh_utils",
    "write_config",
]
