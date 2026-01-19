"""3D mesh generation for Palace EM simulation.

This module provides mesh generation directly from gdsfactory components
and palace-api data structures.

Usage:
    from gsim.palace.mesh import generate_mesh, MeshConfig

    # Quick presets (based on COMSOL guidelines)
    config = MeshConfig.coarse()   # Fast iteration
    config = MeshConfig.default()  # Balanced (COMSOL default)
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
        config=config,
    )
"""

from __future__ import annotations

from gsim.palace.mesh.generator import generate_mesh as generate_mesh_direct
from gsim.palace.mesh.pipeline import (
    GroundPlane,
    MeshConfig,
    MeshPreset,
    MeshResult,
    generate_mesh,
)

from . import gmsh_utils

__all__ = [
    "GroundPlane",
    "MeshConfig",
    "MeshPreset",
    "MeshResult",
    "generate_mesh",
    "generate_mesh_direct",
    "gmsh_utils",
]
