"""Generic mesh data types for GMSH mesh generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MeshResult:
    """Result from generic mesh generation.

    Attributes:
        mesh_path: Path to the generated .msh file.
        mesh_stats: Mesh statistics (nodes, elements, quality, etc.).
        groups: Physical group info mapping:
            {"volumes": {material: {"phys_group": int, "tags": [int]}},
             "outer_boundary": {"phys_group": int, "tags": [int]}}
    """

    mesh_path: Path
    mesh_stats: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)


__all__ = ["MeshResult"]
