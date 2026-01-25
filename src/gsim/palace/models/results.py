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

    # Mesh statistics
    mesh_stats: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["Mesh Summary"]
        lines.append("=" * 40)

        # Dimensions
        if self.mesh_stats:
            bbox = self.mesh_stats.get("bbox", {})
            if bbox:
                dx = bbox.get("xmax", 0) - bbox.get("xmin", 0)
                dy = bbox.get("ymax", 0) - bbox.get("ymin", 0)
                dz = bbox.get("zmax", 0) - bbox.get("zmin", 0)
                lines.append(f"Dimensions: {dx:.1f} x {dy:.1f} x {dz:.1f} µm")

            # Mesh info
            nodes = self.mesh_stats.get("nodes", 0)
            elements = self.mesh_stats.get("elements", 0)
            tets = self.mesh_stats.get("tetrahedra", 0)
            if nodes or elements:
                lines.append(f"Nodes:      {nodes:,}")
                lines.append(f"Elements:   {elements:,}")
                if tets:
                    lines.append(f"Tetrahedra: {tets:,}")

            # Edge lengths
            edge = self.mesh_stats.get("edge_length", {})
            if edge:
                lines.append(
                    f"Edge length: {edge.get('min', 0):.2f} - {edge.get('max', 0):.2f} µm"
                )

            # Mesh quality (gamma)
            quality = self.mesh_stats.get("quality", {})
            if quality:
                q_min = quality.get("min", 0)
                q_mean = quality.get("mean", 0)
                lines.append(f"Quality:    {q_mean:.3f} (min: {q_min:.3f})")

            # SICN quality (shows invalid elements)
            sicn = self.mesh_stats.get("sicn", {})
            if sicn:
                invalid = sicn.get("invalid", 0)
                if invalid > 0:
                    lines.append(f"SICN:       {sicn.get('mean', 0):.3f} ({invalid} invalid!)")
                else:
                    lines.append(f"SICN:       {sicn.get('mean', 0):.3f} (all valid)")

            # Physical groups
            groups = self.mesh_stats.get("groups", {})
            if groups:
                volumes = groups.get("volumes", [])
                surfaces = groups.get("surfaces", [])
                lines.append("-" * 40)
                if volumes:
                    lines.append(f"Volumes ({len(volumes)}):")
                    for v in volumes:
                        name = v["name"] if isinstance(v, dict) else v
                        tag = v.get("tag", "") if isinstance(v, dict) else ""
                        lines.append(f"  - {name}" + (f" [{tag}]" if tag else ""))
                if surfaces:
                    lines.append(f"Surfaces ({len(surfaces)}):")
                    for s in surfaces:
                        name = s["name"] if isinstance(s, dict) else s
                        tag = s.get("tag", "") if isinstance(s, dict) else ""
                        lines.append(f"  - {name}" + (f" [{tag}]" if tag else ""))

        lines.append("-" * 40)
        lines.append(f"Mesh:   {self.mesh_path}")
        if self.config_path:
            lines.append(f"Config: {self.config_path}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.__str__()


__all__ = [
    "SimulationResult",
    "ValidationResult",
]
