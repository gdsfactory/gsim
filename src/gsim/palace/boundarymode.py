"""Boundary mode simulation class for 2D propagation mode analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    BoundaryModeConfig,
    CPWPortConfig,
    CrossSectionPlaneConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PortConfig,
    TerminalConfig,
    WavePortConfig,
)
from gsim.palace.models.results import ValidationResult


class BoundaryModeSim(PalaceSimMixin, BaseModel):
    """Boundary mode simulation for 2D waveguide cross-section analysis.

    This class configures Palace ``BoundaryMode`` simulations used to compute
    propagation constants and mode profiles on a cross-section plane.
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    simulation_type: Literal["boundarymode"] = "boundarymode"

    # Composed objects
    geometry: Geometry | None = None
    stack: LayerStack | None = None

    # Boundary mode config
    boundary_mode: BoundaryModeConfig = Field(default_factory=BoundaryModeConfig)
    cross_section: CrossSectionPlaneConfig | None = None

    # Unused in boundary mode (kept for mixin compatibility)
    driven: None = None
    eigenmode: None = None
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)
    wave_ports: list[WavePortConfig] = Field(default_factory=list)
    terminals: list[TerminalConfig] = Field(default_factory=list)

    # Mesh and solver config
    mesh_config: MeshConfig = Field(default_factory=MeshConfig.default)
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    absorbing_boundary: bool = False

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _pec_blocks: list = PrivateAttr(default_factory=list)
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)
    _airbox_config: dict[str, float] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)

    # Cloud job state
    _job_id: str | None = PrivateAttr(default=None)

    def set_boundary_mode(
        self,
        *,
        freq: float = 5e9,
        num_modes: int = 1,
        save: int = 0,
        target: float = 0.0,
        tolerance: float = 1e-6,
        max_size: int = 0,
        solver_type: str = "Default",
    ) -> None:
        """Configure boundary mode solver parameters.

        Args:
            freq: Operating frequency in Hz.
            num_modes: Number of propagation modes to compute.
            save: Number of modes to save to disk.
            target: Target effective index for shift-and-invert.
            tolerance: Relative eigensolver tolerance.
            max_size: Eigensolver max subspace size.
            solver_type: Palace eigensolver type.
        """
        self.boundary_mode = BoundaryModeConfig(
            freq=freq,
            num_modes=num_modes,
            save=save,
            target=target,
            tolerance=tolerance,
            max_size=max_size,
            solver_type=solver_type,
        )

    def set_cross_section(self, plane: str | CrossSectionPlaneConfig) -> None:
        """Set the explicit cross-section plane for 2D mode extraction.

        Args:
            plane: Plane spec as ``"x=<value>"`` or ``"y=<value>"``, or
                a prebuilt CrossSectionPlaneConfig.
        """
        if isinstance(plane, CrossSectionPlaneConfig):
            self.cross_section = plane
        else:
            self.cross_section = CrossSectionPlaneConfig.from_spec(plane)

    def validate_config(self) -> ValidationResult:
        """Validate boundary mode simulation configuration."""
        base_result = super().validate_config()
        errors = list(base_result.errors)
        warnings = list(base_result.warnings)

        if self.cross_section is None:
            errors.append(
                "Boundary mode requires an explicit cross-section plane. "
                "Call set_cross_section('x=<value>') or set_cross_section('y=<value>')."
            )
        elif self.cross_section.axis == "z":
            errors.append(
                "Boundary mode native 2D currently supports only x/y cross sections. "
                "Use set_cross_section('x=<value>') or set_cross_section('y=<value>')."
            )

        if self.ports or self.cpw_ports or self.wave_ports:
            errors.append(
                "Boundary mode uses cross_section-only native 2D meshing. "
                "add_port(), add_cpw_port(), and add_wave_port() are not supported."
            )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )


__all__ = ["BoundaryModeSim"]
