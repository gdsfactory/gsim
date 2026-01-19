"""Mesh generation pipeline for Palace EM simulation.

This module provides the main entry point for generating meshes from
gdsfactory components. Uses the new generator module internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gsim.palace.ports.config import PalacePort
    from gsim.palace.stack.extractor import LayerStack

from gsim.palace.mesh.generator import generate_mesh as gen_mesh


class MeshPreset(Enum):
    """Mesh quality presets based on COMSOL guidelines.

    COMSOL uses 2nd order elements with ~5 elements per wavelength as default.
    Wavelength in dielectric: λ = c / (f * √εᵣ)
    At 100 GHz in SiO2 (εᵣ≈4): λ ≈ 1500 µm
    """

    COARSE = "coarse"  # ~2.5 elements/λ - fast iteration
    DEFAULT = "default"  # ~5 elements/λ - COMSOL default
    FINE = "fine"  # ~10 elements/λ - high accuracy


# Preset configurations: (refined_mesh_size, max_mesh_size, cells_per_wavelength)
_MESH_PRESETS = {
    MeshPreset.COARSE: (10.0, 600.0, 5),
    MeshPreset.DEFAULT: (5.0, 300.0, 10),
    MeshPreset.FINE: (2.0, 70.0, 20),
}


@dataclass
class GroundPlane:
    """Ground plane configuration for microstrip structures."""

    layer_name: str  # Name of metal layer for ground (e.g., "metal1")
    oversize: float = 50.0  # How much to extend beyond signal geometry (um)


@dataclass
class MeshConfig:
    """Configuration for mesh generation.

    Use class methods for quick presets:
        MeshConfig.coarse()   - Fast iteration (~2.5 elem/λ)
        MeshConfig.default()  - Balanced (COMSOL default, ~5 elem/λ)
        MeshConfig.fine()     - High accuracy (~10 elem/λ)

    Or customize directly:
        MeshConfig(refined_mesh_size=3.0, max_mesh_size=200.0)
    """

    # Mesh size control
    refined_mesh_size: float = 5.0  # Mesh size near conductors (um)
    max_mesh_size: float = 300.0  # Max mesh size in air/dielectric (um)
    cells_per_wavelength: int = 10  # Mesh cells per wavelength

    # Geometry margins
    margin: float = 50.0  # XY margin around design (um)
    air_above: float = 100.0  # Air above top metal (um)

    # Ground plane (optional - for microstrip structures)
    ground_plane: GroundPlane | None = None

    # Frequency for wavelength calculation (optional)
    fmax: float = 100e9  # Max frequency for mesh sizing (Hz)

    # Boundary conditions for 6 faces: [xmin, xmax, ymin, ymax, zmin, zmax]
    # Options: 'ABC' (absorbing), 'PEC' (perfect electric conductor), 'PMC'
    boundary_conditions: list[str] | None = None

    # GUI control
    show_gui: bool = False  # Show gmsh GUI during meshing
    preview_only: bool = False  # Show geometry without meshing

    def __post_init__(self) -> None:
        if self.boundary_conditions is None:
            # Default: ABC everywhere
            self.boundary_conditions = ["ABC", "ABC", "ABC", "ABC", "ABC", "ABC"]

    @classmethod
    def coarse(cls, **kwargs) -> MeshConfig:
        """Fast mesh for quick iteration (~2.5 elements per wavelength)."""
        refined, max_size, cpw = _MESH_PRESETS[MeshPreset.COARSE]
        return cls(
            refined_mesh_size=refined,
            max_mesh_size=max_size,
            cells_per_wavelength=cpw,
            **kwargs,
        )

    @classmethod
    def default(cls, **kwargs) -> MeshConfig:
        """Balanced mesh matching COMSOL defaults (~5 elements per wavelength)."""
        refined, max_size, cpw = _MESH_PRESETS[MeshPreset.DEFAULT]
        return cls(
            refined_mesh_size=refined,
            max_mesh_size=max_size,
            cells_per_wavelength=cpw,
            **kwargs,
        )

    @classmethod
    def fine(cls, **kwargs) -> MeshConfig:
        """High accuracy mesh (~10 elements per wavelength)."""
        refined, max_size, cpw = _MESH_PRESETS[MeshPreset.FINE]
        return cls(
            refined_mesh_size=refined,
            max_mesh_size=max_size,
            cells_per_wavelength=cpw,
            **kwargs,
        )


@dataclass
class MeshResult:
    """Result from mesh generation."""

    mesh_path: Path
    config_path: Path | None = None  # Palace config.json if generated

    # Physical group info for Palace
    conductor_groups: dict = field(default_factory=dict)
    dielectric_groups: dict = field(default_factory=dict)
    port_groups: dict = field(default_factory=dict)
    boundary_groups: dict = field(default_factory=dict)

    # Port metadata
    port_info: list = field(default_factory=list)


def generate_mesh(
    component,
    stack: LayerStack,
    ports: list[PalacePort],
    output_dir: str | Path,
    config: MeshConfig | None = None,
    model_name: str = "palace",
) -> MeshResult:
    """Generate mesh for Palace EM simulation.

    Args:
        component: gdsfactory Component
        stack: LayerStack from palace-api
        ports: List of PalacePort objects (single and multi-element)
        output_dir: Directory for output files
        config: MeshConfig with mesh parameters
        model_name: Base name for output files (default: "mesh" -> mesh.msh)

    Returns:
        MeshResult with mesh path and metadata
    """
    if config is None:
        config = MeshConfig()

    output_dir = Path(output_dir)

    # Use new generator
    result = gen_mesh(
        component=component,
        stack=stack,
        ports=ports,
        output_dir=output_dir,
        model_name=model_name,
        refined_mesh_size=config.refined_mesh_size,
        max_mesh_size=config.max_mesh_size,
        margin=config.margin,
        air_margin=config.margin,
        fmax=config.fmax,
        show_gui=config.show_gui,
    )

    # Convert to pipeline's MeshResult format
    return MeshResult(
        mesh_path=result.mesh_path,
        config_path=result.config_path,
        port_info=result.port_info,
    )
