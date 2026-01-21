"""Pydantic models for Palace EM simulation configuration.

This module provides Pydantic v2 models for configuring Palace simulations,
offering validation, serialization, and a clean API for the fluent PalaceSim class.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from gsim.palace.stack.extractor import Layer, LayerStack
    from gsim.palace.stack.materials import MaterialProperties


# ============================================================================
# Problem Type Enum and Configs
# ============================================================================

ProblemType = Literal["driven", "eigenmode", "electrostatic", "magnetostatic", "transient"]


class DrivenConfig(BaseModel):
    """Configuration for driven (frequency sweep) simulation.

    This is used for S-parameter extraction and frequency response analysis.
    """

    model_config = ConfigDict(frozen=False)

    fmin: float = Field(default=1e9, gt=0, description="Min frequency in Hz")
    fmax: float = Field(default=100e9, gt=0, description="Max frequency in Hz")
    num_points: int = Field(default=40, ge=1, description="Number of frequency points")
    scale: Literal["linear", "log"] = "linear"

    # Adaptive options
    adaptive_tol: float = Field(
        default=0.02, ge=0, description="Adaptive tolerance (0 disables adaptive)"
    )
    adaptive_max_samples: int = Field(default=20, ge=1)

    # S-parameter options
    compute_s_params: bool = True
    reference_impedance: float = Field(default=50.0, gt=0)

    @model_validator(mode="after")
    def validate_frequency_range(self) -> Self:
        """Validate that fmin < fmax."""
        if self.fmin >= self.fmax:
            raise ValueError(f"fmin ({self.fmin}) must be less than fmax ({self.fmax})")
        return self

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        freq_step = (self.fmax - self.fmin) / max(1, self.num_points - 1) / 1e9
        return {
            "Samples": [
                {
                    "Type": "Linear" if self.scale == "linear" else "Log",
                    "MinFreq": self.fmin / 1e9,
                    "MaxFreq": self.fmax / 1e9,
                    "FreqStep": freq_step,
                    "SaveStep": 0,
                }
            ],
            "AdaptiveTol": self.adaptive_tol if self.adaptive_tol > 0 else 0,
        }


class EigenmodeConfig(BaseModel):
    """Configuration for eigenmode (resonance) simulation.

    This is used for finding resonant frequencies and mode shapes.
    """

    model_config = ConfigDict(frozen=False)

    num_modes: int = Field(default=10, ge=1, alias="N", description="Number of modes to find")
    target: float | None = Field(default=None, description="Target frequency in Hz")
    tolerance: float = Field(default=1e-6, gt=0, description="Eigenvalue solver tolerance")

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        config: dict = {
            "N": self.num_modes,
            "Tol": self.tolerance,
        }
        if self.target is not None:
            config["Target"] = self.target / 1e9  # Convert to GHz
        return config


class ElectrostaticConfig(BaseModel):
    """Configuration for electrostatic (capacitance matrix) simulation."""

    model_config = ConfigDict(frozen=False)

    save_fields: int = Field(default=0, ge=0, description="Number of fields to save")

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        return {
            "Save": self.save_fields,
        }


class MagnetostaticConfig(BaseModel):
    """Configuration for magnetostatic (inductance matrix) simulation."""

    model_config = ConfigDict(frozen=False)

    save_fields: int = Field(default=0, ge=0, description="Number of fields to save")

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        return {
            "Save": self.save_fields,
        }


class TransientConfig(BaseModel):
    """Configuration for transient (time-domain) simulation."""

    model_config = ConfigDict(frozen=False)

    excitation: Literal["sinusoidal", "gaussian", "ramp", "smoothstep"] = "sinusoidal"
    excitation_freq: float | None = Field(default=None, description="Excitation frequency in Hz")
    excitation_width: float | None = Field(
        default=None, description="Pulse width in ns (for gaussian)"
    )
    max_time: float = Field(description="Maximum simulation time in ns")
    time_step: float | None = Field(
        default=None, description="Time step in ns (None = adaptive)"
    )

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        config: dict = {
            "Type": self.excitation.capitalize(),
            "MaxTime": self.max_time,
        }
        if self.excitation_freq is not None:
            config["ExcitationFreq"] = self.excitation_freq / 1e9  # Convert to GHz
        if self.excitation_width is not None:
            config["ExcitationWidth"] = self.excitation_width
        if self.time_step is not None:
            config["TimeStep"] = self.time_step
        return config


class WavePortConfig(BaseModel):
    """Configuration for a wave port (domain boundary with mode solving)."""

    model_config = ConfigDict(frozen=False)

    name: str
    layer: str
    mode: int = Field(default=1, ge=1, description="Mode number to excite")
    excited: bool = True
    offset: float = Field(default=0.0, ge=0, description="De-embedding distance in um")


class TerminalConfig(BaseModel):
    """Configuration for a terminal (for electrostatic capacitance extraction)."""

    model_config = ConfigDict(frozen=False)

    name: str
    layer: str


class MaterialPropertiesModel(BaseModel):
    """EM properties for a material."""

    model_config = ConfigDict(frozen=False)

    type: Literal["conductor", "dielectric", "semiconductor"]
    conductivity: float | None = Field(default=None, ge=0)
    permittivity: float | None = Field(default=None, ge=1.0)
    loss_tangent: float | None = Field(default=None, ge=0, le=1)

    @classmethod
    def from_legacy(cls, props: MaterialProperties) -> Self:
        """Create from legacy dataclass."""
        return cls(
            type=props.type,
            conductivity=props.conductivity,
            permittivity=props.permittivity,
            loss_tangent=props.loss_tangent,
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for YAML output."""
        d: dict[str, object] = {"type": self.type}
        if self.conductivity is not None:
            d["conductivity"] = self.conductivity
        if self.permittivity is not None:
            d["permittivity"] = self.permittivity
        if self.loss_tangent is not None:
            d["loss_tangent"] = self.loss_tangent
        return d


class LayerModel(BaseModel):
    """Layer information for Palace simulation."""

    model_config = ConfigDict(frozen=False)

    name: str
    gds_layer: tuple[int, int]
    zmin: float
    zmax: float
    material: str
    layer_type: Literal["conductor", "via", "dielectric", "substrate"]
    mesh_resolution: str | float = "medium"

    @property
    def thickness(self) -> float:
        """Layer thickness in um."""
        return self.zmax - self.zmin

    @classmethod
    def from_legacy(cls, layer: Layer) -> Self:
        """Create from legacy dataclass."""
        return cls(
            name=layer.name,
            gds_layer=layer.gds_layer,
            zmin=layer.zmin,
            zmax=layer.zmax,
            material=layer.material,
            layer_type=layer.layer_type,
            mesh_resolution=layer.mesh_resolution,
        )

    def get_mesh_size(self, base_size: float = 1.0) -> float:
        """Get mesh size in um for this layer."""
        if isinstance(self.mesh_resolution, int | float):
            return float(self.mesh_resolution)

        resolution_map = {
            "fine": base_size * 0.5,
            "medium": base_size,
            "coarse": base_size * 2.0,
        }
        return resolution_map.get(self.mesh_resolution, base_size)

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML output."""
        return {
            "gds_layer": list(self.gds_layer),
            "zmin": round(self.zmin, 4),
            "zmax": round(self.zmax, 4),
            "thickness": round(self.thickness, 4),
            "material": self.material,
            "type": self.layer_type,
            "mesh_resolution": self.mesh_resolution,
        }


class LayerStackModel(BaseModel):
    """Complete layer stack for Palace simulation."""

    model_config = ConfigDict(frozen=False)

    pdk_name: str
    units: str = "um"
    layers: dict[str, LayerModel] = Field(default_factory=dict)
    materials: dict[str, dict] = Field(default_factory=dict)
    dielectrics: list[dict] = Field(default_factory=list)
    simulation: dict = Field(default_factory=dict)

    @classmethod
    def from_legacy(cls, stack: LayerStack) -> Self:
        """Create from legacy dataclass."""
        layers = {
            name: LayerModel.from_legacy(layer)
            for name, layer in stack.layers.items()
        }
        return cls(
            pdk_name=stack.pdk_name,
            units=stack.units,
            layers=layers,
            materials=stack.materials,
            dielectrics=stack.dielectrics,
            simulation=stack.simulation,
        )

    def get_z_range(self) -> tuple[float, float]:
        """Get the full z-range of the stack."""
        if not self.dielectrics:
            return (0.0, 0.0)
        z_min = min(d["zmin"] for d in self.dielectrics)
        z_max = max(d["zmax"] for d in self.dielectrics)
        return (z_min, z_max)

    def get_conductor_layers(self) -> dict[str, LayerModel]:
        """Get all conductor layers."""
        return {
            n: layer
            for n, layer in self.layers.items()
            if layer.layer_type == "conductor"
        }

    def get_via_layers(self) -> dict[str, LayerModel]:
        """Get all via layers."""
        return {
            n: layer
            for n, layer in self.layers.items()
            if layer.layer_type == "via"
        }


class MeshConfigModel(BaseModel):
    """Configuration for mesh generation with COMSOL-style presets."""

    model_config = ConfigDict(frozen=False)

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
        """Fast mesh for quick iteration (~2.5 elements per wavelength)."""
        defaults = {
            "refined_mesh_size": 10.0,
            "max_mesh_size": 600.0,
            "cells_per_wavelength": 5,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def default(cls, **kwargs) -> Self:
        """Balanced mesh matching COMSOL defaults (~5 elements per wavelength)."""
        defaults = {
            "refined_mesh_size": 5.0,
            "max_mesh_size": 300.0,
            "cells_per_wavelength": 10,
        }
        defaults.update(kwargs)
        return cls(**defaults)

    @classmethod
    def fine(cls, **kwargs) -> Self:
        """High accuracy mesh (~10 elements per wavelength)."""
        defaults = {
            "refined_mesh_size": 2.0,
            "max_mesh_size": 70.0,
            "cells_per_wavelength": 20,
        }
        defaults.update(kwargs)
        return cls(**defaults)


class PhysicsConfig(BaseModel):
    """Full Palace solver physics configuration.

    DEPRECATED: Use DrivenConfig, EigenmodeConfig, etc. instead.
    This class is kept for backward compatibility.
    """

    model_config = ConfigDict(frozen=False)

    # Problem type
    problem_type: Literal["driven", "eigenmode"] = "driven"

    # Frequency settings (driven)
    fmin: float = Field(default=1e9, gt=0)
    fmax: float = Field(default=100e9, gt=0)
    num_frequency_points: int = Field(default=40, ge=1)
    frequency_scale: Literal["linear", "log"] = "linear"

    # Eigenmode settings
    num_modes: int = Field(default=10, ge=1)
    target_frequency: float | None = None  # Hz, for eigenmode targeting

    # S-parameter options
    compute_s_params: bool = True
    reference_impedance: float = Field(default=50.0, gt=0)

    # Adaptive frequency refinement
    adaptive_refinement: bool = False
    refinement_tolerance: float = Field(default=0.01, gt=0)

    @model_validator(mode="after")
    def validate_frequency_range(self) -> Self:
        """Validate that fmin < fmax."""
        if self.fmin >= self.fmax:
            raise ValueError(f"fmin ({self.fmin}) must be less than fmax ({self.fmax})")
        return self

    def to_driven_config(self) -> DrivenConfig:
        """Convert to DrivenConfig for backward compatibility."""
        return DrivenConfig(
            fmin=self.fmin,
            fmax=self.fmax,
            num_points=self.num_frequency_points,
            scale=self.frequency_scale,
            adaptive_tol=self.refinement_tolerance if self.adaptive_refinement else 0.0,
            compute_s_params=self.compute_s_params,
            reference_impedance=self.reference_impedance,
        )

    def to_eigenmode_config(self) -> EigenmodeConfig:
        """Convert to EigenmodeConfig for backward compatibility."""
        return EigenmodeConfig(
            num_modes=self.num_modes,
            target=self.target_frequency,
        )


class NumericalConfig(BaseModel):
    """Numerical solver configuration."""

    model_config = ConfigDict(frozen=False)

    # Element order
    order: int = Field(default=2, ge=1, le=4)

    # Linear solver
    tolerance: float = Field(default=1e-6, gt=0)
    max_iterations: int = Field(default=400, ge=1)
    solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = "Default"

    # Preconditioner
    preconditioner: Literal["Default", "AMS", "BoomerAMG"] = "Default"

    # Device
    device: Literal["CPU", "GPU"] = "CPU"

    # Partitioning
    num_processors: int | None = None  # None = auto


class PortConfigModel(BaseModel):
    """Port configuration for Palace simulation."""

    model_config = ConfigDict(frozen=False)

    name: str | None = None
    port_type: Literal["lumped", "waveport"] = "lumped"
    geometry: Literal["inplane", "via"] = "inplane"
    layer: str | None = None
    from_layer: str | None = None
    to_layer: str | None = None
    length: float | None = None
    impedance: float = Field(default=50.0, gt=0)
    excited: bool = True

    @model_validator(mode="after")
    def validate_layer_config(self) -> Self:
        """Validate layer configuration based on geometry type."""
        if self.geometry == "inplane" and self.layer is None:
            raise ValueError("Inplane ports require 'layer' to be specified")
        if self.geometry == "via":
            if self.from_layer is None or self.to_layer is None:
                raise ValueError(
                    "Via ports require both 'from_layer' and 'to_layer'"
                )
        return self


class ValidationResultModel(BaseModel):
    """Result of simulation configuration validation."""

    model_config = ConfigDict(frozen=False)

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
    """Result from running a Palace simulation."""

    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)

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
