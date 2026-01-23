"""Result models for Palace simulations.

This module contains Pydantic models for simulation results and validation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ValidationResult(BaseModel):
    """Result of simulation configuration validation.

    Attributes:
        valid: Whether the configuration is valid
        errors: List of error messages
        warnings: List of warning messages
    """

    model_config = ConfigDict(validate_assignment=True)

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def __str__(self) -> str:
        lines = []
        if self.valid:
            lines.append("Validation: PASSED")
        else:
            lines.append("Validation: FAILED")
        if self.errors:
            lines.append("Errors:")
            lines.extend([f"  - {e}" for e in self.errors])
        if self.warnings:
            lines.append("Warnings:")
            lines.extend([f"  - {w}" for w in self.warnings])
        return "\n".join(lines)


class SimulationResult(BaseModel):
    """Result from running a Palace simulation.

    Attributes:
        mesh_path: Path to the generated mesh file
        output_dir: Output directory path
        config_path: Path to the Palace config file
        results: Dictionary mapping result filenames to paths
        conductor_groups: Physical group info for conductors
        dielectric_groups: Physical group info for dielectrics
        port_groups: Physical group info for ports
        boundary_groups: Physical group info for boundaries
        port_info: Port metadata
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    mesh_path: Path
    output_dir: Path
    config_path: Path | None = None
    results: dict[str, Path] = Field(default_factory=dict)

    # Physical group info for Palace
    conductor_groups: dict = Field(default_factory=dict)
    dielectric_groups: dict = Field(default_factory=dict)
    port_groups: dict = Field(default_factory=dict)
    boundary_groups: dict = Field(default_factory=dict)

    # Port metadata
    port_info: list = Field(default_factory=list)


__all__ = [
    "SimulationResult",
    "ValidationResult",
]
