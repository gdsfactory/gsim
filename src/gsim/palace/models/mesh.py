"""Mesh configuration models for Palace simulations.

This module contains Pydantic models for mesh generation configuration.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MeshConfig(BaseModel):
    """Configuration for mesh generation with COMSOL-style presets.

    Attributes:
        refined_mesh_size: Mesh size near conductors (um)
        max_mesh_size: Maximum mesh size in air/dielectric (um)
        cells_per_wavelength: Number of mesh cells per wavelength
        margin: XY margin around design (um)
        air_above: Air above top metal (um)
        fmax: Maximum frequency for mesh sizing (Hz)
        boundary_conditions: List of boundary conditions for each face
        show_gui: Show gmsh GUI during meshing
        preview_only: Generate preview only, don't save mesh
    """

    model_config = ConfigDict(validate_assignment=True)

    refined_mesh_size: float = Field(default=5.0, gt=0)
    max_mesh_size: float = Field(default=300.0, gt=0)
    cells_per_wavelength: int = Field(default=10, ge=1)
    margin: float = Field(default=50.0, ge=0)
    air_above: float = Field(default=100.0, ge=0)
    fmax: float = Field(default=100e9, gt=0)
    boundary_conditions: list[str] | None = None
    show_gui: bool = False
    preview_only: bool = False

    @model_validator(mode="after")
    def set_default_boundary_conditions(self) -> Self:
        """Set default boundary conditions if not provided."""
        if self.boundary_conditions is None:
            self.boundary_conditions = ["ABC", "ABC", "ABC", "ABC", "ABC", "ABC"]
        return self

    @classmethod
    def coarse(cls, **kwargs) -> Self:
        """Fast mesh for quick iteration (~2.5 elements per wavelength).

        This preset is suitable for initial debugging and quick checks.
        Not recommended for accurate results.
        """
        defaults = {
            "refined_mesh_size": 10.0,
            "max_mesh_size": 600.0,
            "cells_per_wavelength": 5,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def default(cls, **kwargs) -> Self:
        """Balanced mesh matching COMSOL defaults (~5 elements per wavelength).

        This preset provides a good balance between accuracy and computation time.
        Suitable for most simulations.
        """
        defaults = {
            "refined_mesh_size": 5.0,
            "max_mesh_size": 300.0,
            "cells_per_wavelength": 10,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def fine(cls, **kwargs) -> Self:
        """High accuracy mesh (~10 elements per wavelength).

        This preset provides higher accuracy at the cost of increased
        computation time. Use for final production simulations.
        """
        defaults = {
            "refined_mesh_size": 2.0,
            "max_mesh_size": 70.0,
            "cells_per_wavelength": 20,
        }
        defaults.update(kwargs)
        return cls(**defaults)


__all__ = [
    "MeshConfig",
]
