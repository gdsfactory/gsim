"""Electrostatic simulation class for capacitance extraction.

This module provides the ElectrostaticSim class for extracting
capacitance matrices between terminals.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, PrivateAttr

from gsim.palace.base import SimBase
from gsim.palace.models import (
    ElectrostaticConfig,
    SimulationResult,
    TerminalConfig,
    ValidationResult,
)

if TYPE_CHECKING:
    from gsim.palace.stack.extractor import LayerStack


class ElectrostaticSim(SimBase):
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
        >>> sim.mesh("./sim", preset="default")
        >>> results = sim.simulate()

    Attributes:
        terminals: List of terminal configurations
        electrostatic: Electrostatic simulation configuration
    """

    # Terminal configurations (no ports in electrostatic)
    terminals: list[TerminalConfig] = Field(default_factory=list)

    # Electrostatic simulation config
    electrostatic: ElectrostaticConfig = Field(default_factory=ElectrostaticConfig)

    # Internal state
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

    def validate(self) -> ValidationResult:
        """Validate the simulation configuration.

        Returns:
            ValidationResult with validation status and messages
        """
        result = self._validate_base()
        errors = list(result.errors)
        warnings_list = list(result.warnings)

        # Electrostatic requires at least 2 terminals
        if len(self.terminals) < 2:
            errors.append(
                "Electrostatic simulation requires at least 2 terminals. "
                "Call add_terminal() to add terminals."
            )

        # Validate terminal configurations
        for terminal in self.terminals:
            if not terminal.layer:
                errors.append(f"Terminal '{terminal.name}': 'layer' is required")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Preview (no ports needed for electrostatic)
    # -------------------------------------------------------------------------

    def _get_ports_for_preview(self, stack: LayerStack) -> list:
        """Get ports for preview (none for electrostatic)."""
        return []

    # -------------------------------------------------------------------------
    # Mesh generation
    # -------------------------------------------------------------------------

    def mesh(
        self,
        output_dir: str | Path,
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
        """Generate the mesh and configuration files.

        Args:
            output_dir: Directory for output files
            preset: Mesh quality preset ("coarse", "default", "fine")
            refined_mesh_size: Mesh size near conductors (um)
            max_mesh_size: Max mesh size in air/dielectric (um)
            margin: XY margin around design (um)
            air_above: Air above top metal (um)
            fmax: Max frequency for mesh sizing (Hz) - less relevant for electrostatic
            show_gui: Show gmsh GUI during meshing
            model_name: Base name for output files
            verbose: Print progress messages

        Returns:
            SimulationResult with mesh and config paths
        """
        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            air_above=air_above,
            fmax=fmax,
            show_gui=show_gui,
        )

        validation = self.validate()
        if not validation.valid:
            raise ValueError(
                f"Invalid configuration:\n" + "\n".join(validation.errors)
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir = output_dir

        stack = self._resolve_stack()

        # Electrostatic doesn't use ports
        return self._generate_mesh_internal(
            output_dir=output_dir,
            mesh_config=mesh_config,
            ports=[],  # No ports for electrostatic
            driven_config=None,  # No driven config for electrostatic
            model_name=model_name,
            verbose=verbose,
        )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def simulate(
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
