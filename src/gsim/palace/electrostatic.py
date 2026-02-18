"""Electrostatic simulation class for capacitance extraction.

This module provides the ElectrostaticSim class for extracting
capacitance matrices between terminals.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.models import (
    ElectrostaticConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    SimulationResult,
    TerminalConfig,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ElectrostaticSim(PalaceSimMixin, BaseModel):
    """Electrostatic simulation for capacitance matrix extraction.

    This class configures and runs electrostatic simulations to extract
    the capacitance matrix between conductor terminals. Unlike driven
    and eigenmode simulations, this does not use ports.

    Example:
        >>> from gsim.palace import ElectrostaticSim
        >>>
        >>> sim = ElectrostaticSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack(air_above=300.0)
        >>> sim.add_terminal("T1", layer="topmetal2")
        >>> sim.add_terminal("T2", layer="topmetal2")
        >>> sim.set_electrostatic()
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> results = sim.run()

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        terminals: List of terminal configurations
        electrostatic: Electrostatic simulation configuration
        materials: Material property overrides
        numerical: Numerical solver configuration
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Composed objects (from common)
    geometry: Geometry | None = None
    stack: LayerStack | None = None

    # Terminal configurations (no ports in electrostatic)
    terminals: list[TerminalConfig] = Field(default_factory=list)

    # Electrostatic simulation config
    electrostatic: ElectrostaticConfig = Field(default_factory=ElectrostaticConfig)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_terminals: bool = PrivateAttr(default=False)

    # -------------------------------------------------------------------------
    # Terminal methods
    # -------------------------------------------------------------------------

    def add_terminal(
        self,
        name: str,
        *,
        layer: str,
    ) -> None:
        """Add a terminal for capacitance extraction.

        Terminals define conductor surfaces for capacitance matrix extraction.

        Args:
            name: Terminal name
            layer: Target conductor layer

        Example:
            >>> sim.add_terminal("T1", layer="topmetal2")
            >>> sim.add_terminal("T2", layer="topmetal2")
        """
        # Remove existing terminal with same name
        self.terminals = [t for t in self.terminals if t.name != name]
        self.terminals.append(
            TerminalConfig(
                name=name,
                layer=layer,
            )
        )

    # -------------------------------------------------------------------------
    # Electrostatic configuration
    # -------------------------------------------------------------------------

    def set_electrostatic(
        self,
        *,
        save_fields: int = 0,
    ) -> None:
        """Configure electrostatic simulation.

        Args:
            save_fields: Number of field solutions to save

        Example:
            >>> sim.set_electrostatic(save_fields=1)
        """
        self.electrostatic = ElectrostaticConfig(
            save_fields=save_fields,
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_config(self) -> ValidationResult:
        """Validate the simulation configuration.

        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings_list = []

        # Check geometry
        if self.geometry is None:
            errors.append("No component set. Call set_geometry(component) first.")

        # Check stack
        if self.stack is None and not self._stack_kwargs:
            warnings_list.append(
                "No stack configured. Will use active PDK with defaults."
            )

        # Electrostatic requires at least 2 terminals
        if len(self.terminals) < 2:
            errors.append(
                "Electrostatic simulation requires at least 2 terminals. "
                "Call add_terminal() to add terminals."
            )

        # Validate terminal configurations
        errors.extend(
            f"Terminal '{terminal.name}': 'layer' is required"
            for terminal in self.terminals
            if not terminal.layer
        )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _generate_mesh_internal(
        self,
        output_dir: Path,
        mesh_config: MeshConfig,
        model_name: str,
        verbose: bool,
    ) -> SimulationResult:
        """Internal mesh generation."""
        from gsim.palace.mesh import MeshConfig as LegacyMeshConfig
        from gsim.palace.mesh import generate_mesh

        component = self.geometry.component if self.geometry else None

        legacy_mesh_config = LegacyMeshConfig(
            refined_mesh_size=mesh_config.refined_mesh_size,
            max_mesh_size=mesh_config.max_mesh_size,
            cells_per_wavelength=mesh_config.cells_per_wavelength,
            margin=mesh_config.margin,
            air_above=mesh_config.air_above,
            fmax=mesh_config.fmax,
            show_gui=mesh_config.show_gui,
            preview_only=mesh_config.preview_only,
        )

        stack = self._resolve_stack()

        if verbose:
            logger.info("Generating mesh in %s", output_dir)

        mesh_result = generate_mesh(
            component=component,
            stack=stack,
            ports=[],  # No ports for electrostatic
            output_dir=output_dir,
            config=legacy_mesh_config,
            model_name=model_name,
            driven_config=None,  # No driven config for electrostatic
        )

        return SimulationResult(
            mesh_path=mesh_result.mesh_path,
            output_dir=output_dir,
            config_path=mesh_result.config_path,
            port_info=mesh_result.port_info,
            mesh_stats=mesh_result.mesh_stats,
        )

    # -------------------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------------------

    def preview(
        self,
        *,
        preset: Literal["coarse", "default", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        air_above: float | None = None,
        fmax: float | None = None,
        show_gui: bool = True,
    ) -> None:
        """Preview the mesh without running simulation.

        Args:
            preset: Mesh quality preset ("coarse", "default", "fine")
            refined_mesh_size: Mesh size near conductors (um)
            max_mesh_size: Max mesh size in air/dielectric (um)
            margin: XY margin around design (um)
            air_above: Air above top metal (um)
            fmax: Max frequency for mesh sizing (Hz)
            show_gui: Show gmsh GUI for interactive preview

        Example:
            >>> sim.preview(preset="fine", show_gui=True)
        """
        from gsim.palace.mesh import MeshConfig as LegacyMeshConfig
        from gsim.palace.mesh import generate_mesh

        component = self.geometry.component if self.geometry else None

        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            air_above=air_above,
            fmax=fmax,
            show_gui=show_gui,
        )

        stack = self._resolve_stack()

        legacy_mesh_config = LegacyMeshConfig(
            refined_mesh_size=mesh_config.refined_mesh_size,
            max_mesh_size=mesh_config.max_mesh_size,
            cells_per_wavelength=mesh_config.cells_per_wavelength,
            margin=mesh_config.margin,
            air_above=mesh_config.air_above,
            fmax=mesh_config.fmax,
            show_gui=show_gui,
            preview_only=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_mesh(
                component=component,
                stack=stack,
                ports=[],
                output_dir=tmpdir,
                config=legacy_mesh_config,
            )

    # -------------------------------------------------------------------------
    # Mesh generation
    # -------------------------------------------------------------------------

    def mesh(
        self,
        *,
        preset: Literal["coarse", "default", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        air_above: float | None = None,
        fmax: float | None = None,
        show_gui: bool = False,
        model_name: str = "palace",
        verbose: bool = True,
    ) -> SimulationResult:
        """Generate the mesh for Palace simulation.

        Requires set_output_dir() to be called first.

        Args:
            preset: Mesh quality preset ("coarse", "default", "fine")
            refined_mesh_size: Mesh size near conductors (um), overrides preset
            max_mesh_size: Max mesh size in air/dielectric (um), overrides preset
            margin: XY margin around design (um), overrides preset
            air_above: Air above top metal (um), overrides preset
            fmax: Max frequency for mesh sizing (Hz) - less relevant for electrostatic
            show_gui: Show gmsh GUI during meshing
            model_name: Base name for output files
            verbose: Print progress messages

        Returns:
            SimulationResult with mesh path

        Raises:
            ValueError: If output_dir not set or configuration is invalid

        Example:
            >>> sim.set_output_dir("./sim")
            >>> result = sim.mesh(preset="fine")
            >>> print(f"Mesh saved to: {result.mesh_path}")
        """
        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            air_above=air_above,
            fmax=fmax,
            show_gui=show_gui,
        )

        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        output_dir = self._output_dir

        self._resolve_stack()

        return self._generate_mesh_internal(
            output_dir=output_dir,
            mesh_config=mesh_config,
            model_name=model_name,
            verbose=verbose,
        )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def run(
        self,
        output_dir: str | Path | None = None,
        *,
        verbose: bool = True,
    ) -> dict[str, Path]:
        """Run electrostatic simulation on GDSFactory+ cloud.

        Args:
            output_dir: Directory containing mesh files
            verbose: Print progress messages

        Returns:
            Dict mapping result filenames to local paths

        Raises:
            NotImplementedError: Electrostatic is not yet fully implemented
        """
        raise NotImplementedError(
            "Electrostatic simulation is not yet fully implemented on cloud. "
            "Use DrivenSim for S-parameter extraction."
        )


__all__ = ["ElectrostaticSim"]
