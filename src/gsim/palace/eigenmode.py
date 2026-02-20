"""Eigenmode simulation class for resonance/mode finding.

This module provides the EigenmodeSim class for finding resonant
frequencies and mode shapes.
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
    CPWPortConfig,
    EigenmodeConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PortConfig,
    SimulationResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class EigenmodeSim(PalaceSimMixin, BaseModel):
    """Eigenmode simulation for finding resonant frequencies.

    This class configures and runs eigenmode simulations to find
    resonant frequencies and mode shapes of structures.

    Example:
        >>> from gsim.palace import EigenmodeSim
        >>>
        >>> sim = EigenmodeSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack(air_above=300.0)
        >>> sim.add_port("o1", layer="topmetal2", length=5.0)
        >>> sim.set_eigenmode(num_modes=10, target=50e9)
        >>> sim.set_output_dir("./sim")
        >>> sim.mesh(preset="default")
        >>> results = sim.run()

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        ports: List of single-element port configurations
        cpw_ports: List of CPW (two-element) port configurations
        eigenmode: Eigenmode simulation configuration
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

    # Port configurations (eigenmode can have ports for Q-factor calculation)
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)

    # Eigenmode simulation config
    eigenmode: EigenmodeConfig = Field(default_factory=EigenmodeConfig)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_ports: bool = PrivateAttr(default=False)

    # -------------------------------------------------------------------------
    # Port methods (Eigenmode can have ports for Q-factor calculation)
    # -------------------------------------------------------------------------

    def add_port(
        self,
        name: str,
        *,
        layer: str | None = None,
        from_layer: str | None = None,
        to_layer: str | None = None,
        length: float | None = None,
        impedance: float = 50.0,
        excited: bool = True,
        geometry: Literal["inplane", "via"] = "inplane",
    ) -> None:
        """Add a single-element lumped port.

        Args:
            name: Port name (must match component port name)
            layer: Target layer for inplane ports
            from_layer: Bottom layer for via ports
            to_layer: Top layer for via ports
            length: Port extent along direction (um)
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited
            geometry: Port geometry type ("inplane" or "via")

        Example:
            >>> sim.add_port("o1", layer="topmetal2", length=5.0)
        """
        self.ports = [p for p in self.ports if p.name != name]
        self.ports.append(
            PortConfig(
                name=name,
                layer=layer,
                from_layer=from_layer,
                to_layer=to_layer,
                length=length,
                impedance=impedance,
                excited=excited,
                geometry=geometry,
            )
        )

    def add_cpw_port(
        self,
        upper: str,
        lower: str,
        *,
        layer: str,
        length: float,
        impedance: float = 50.0,
        excited: bool = True,
        name: str | None = None,
    ) -> None:
        """Add a coplanar waveguide (CPW) port.

        Args:
            upper: Name of the upper gap port on the component
            lower: Name of the lower gap port on the component
            layer: Target conductor layer
            length: Port extent along direction (um)
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited
            name: Optional name for the CPW port

        Example:
            >>> sim.add_cpw_port("P2", "P1", layer="topmetal2", length=5.0)
        """
        self.cpw_ports = [
            p for p in self.cpw_ports if not (p.upper == upper and p.lower == lower)
        ]
        self.cpw_ports.append(
            CPWPortConfig(
                upper=upper,
                lower=lower,
                layer=layer,
                length=length,
                impedance=impedance,
                excited=excited,
                name=name,
            )
        )

    # -------------------------------------------------------------------------
    # Eigenmode configuration
    # -------------------------------------------------------------------------

    def set_eigenmode(
        self,
        *,
        num_modes: int = 10,
        target: float | None = None,
        tolerance: float = 1e-6,
    ) -> None:
        """Configure eigenmode simulation.

        Args:
            num_modes: Number of modes to find
            target: Target frequency in Hz for mode search
            tolerance: Eigenvalue solver tolerance

        Example:
            >>> sim.set_eigenmode(num_modes=10, target=50e9)
        """
        self.eigenmode = EigenmodeConfig(
            num_modes=num_modes,
            target=target,
            tolerance=tolerance,
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

        # Eigenmode simulations may not require ports
        if not self.ports and not self.cpw_ports:
            warnings_list.append(
                "No ports configured. Eigenmode finds all modes without port loading."
            )

        # Validate port configurations
        for port in self.ports:
            if port.geometry == "inplane" and port.layer is None:
                errors.append(f"Port '{port.name}': inplane ports require 'layer'")
            if port.geometry == "via" and (
                port.from_layer is None or port.to_layer is None
            ):
                errors.append(
                    f"Port '{port.name}': via ports require 'from_layer' and 'to_layer'"
                )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _configure_ports_on_component(self, stack: LayerStack) -> None:  # noqa: ARG002
        """Configure ports on the component."""
        from gsim.palace.ports import (
            configure_cpw_port,
            configure_inplane_port,
            configure_via_port,
        )

        component = self.geometry.component if self.geometry else None
        if component is None:
            raise ValueError("No component set")

        for port_config in self.ports:
            if port_config.name is None:
                continue

            gf_port = None
            for p in component.ports:
                if p.name == port_config.name:
                    gf_port = p
                    break

            if gf_port is None:
                raise ValueError(
                    f"Port '{port_config.name}' not found on component. "
                    f"Available: {[p.name for p in component.ports]}"
                )

            if port_config.geometry == "inplane" and port_config.layer is not None:
                configure_inplane_port(
                    gf_port,
                    layer=port_config.layer,
                    length=port_config.length or gf_port.width,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                )
            elif port_config.geometry == "via" and (
                port_config.from_layer is not None and port_config.to_layer is not None
            ):
                configure_via_port(
                    gf_port,
                    from_layer=port_config.from_layer,
                    to_layer=port_config.to_layer,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                )

        for cpw_config in self.cpw_ports:
            port_upper = None
            port_lower = None
            for p in component.ports:
                if p.name == cpw_config.upper:
                    port_upper = p
                if p.name == cpw_config.lower:
                    port_lower = p

            if port_upper is None:
                raise ValueError(f"CPW upper port '{cpw_config.upper}' not found.")
            if port_lower is None:
                raise ValueError(f"CPW lower port '{cpw_config.lower}' not found.")

            configure_cpw_port(
                port_upper=port_upper,
                port_lower=port_lower,
                layer=cpw_config.layer,
                length=cpw_config.length,
                impedance=cpw_config.impedance,
                excited=cpw_config.excited,
                cpw_name=cpw_config.name,
            )

        self._configured_ports = True

    def _generate_mesh_internal(
        self,
        output_dir: Path,
        mesh_config: MeshConfig,
        ports: list,
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
            planar_conductors=mesh_config.planar_conductors,
        )

        stack = self._resolve_stack()

        if verbose:
            logger.info("Generating mesh in %s", output_dir)

        mesh_result = generate_mesh(
            component=component,
            stack=stack,
            ports=ports,
            output_dir=output_dir,
            config=legacy_mesh_config,
            model_name=model_name,
            driven_config=None,  # Eigenmode doesn't use driven config
        )

        return SimulationResult(
            mesh_path=mesh_result.mesh_path,
            output_dir=output_dir,
            config_path=mesh_result.config_path,
            port_info=mesh_result.port_info,
            mesh_stats=mesh_result.mesh_stats,
        )

    def _get_ports_for_preview(self, stack: LayerStack) -> list:
        """Get ports for preview."""
        from gsim.palace.ports import extract_ports

        component = self.geometry.component if self.geometry else None
        if self.ports or self.cpw_ports:
            self._configure_ports_on_component(stack)
            return extract_ports(component, stack)
        return []

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
        planar_conductors: bool | None = None,
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
            planar_conductors: Treat conductors as 2D PEC surfaces
            show_gui: Show gmsh GUI for interactive preview

        Example:
            >>> sim.preview(preset="fine", planar_conductors=True, show_gui=True)
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
            planar_conductors=planar_conductors,
            show_gui=show_gui,
        )

        stack = self._resolve_stack()
        ports = self._get_ports_for_preview(stack)

        legacy_mesh_config = LegacyMeshConfig(
            refined_mesh_size=mesh_config.refined_mesh_size,
            max_mesh_size=mesh_config.max_mesh_size,
            cells_per_wavelength=mesh_config.cells_per_wavelength,
            margin=mesh_config.margin,
            air_above=mesh_config.air_above,
            fmax=mesh_config.fmax,
            show_gui=show_gui,
            preview_only=True,
            planar_conductors=mesh_config.planar_conductors,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_mesh(
                component=component,
                stack=stack,
                ports=ports,
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
        planar_conductors: bool | None = None,
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
            fmax: Max frequency for mesh sizing (Hz), overrides preset
            planar_conductors: Treat conductors as 2D PEC surfaces, overrides mesh_config
            show_gui: Show gmsh GUI during meshing
            model_name: Base name for output files
            verbose: Print progress messages

        Returns:
            SimulationResult with mesh path

        Raises:
            ValueError: If output_dir not set or configuration is invalid

        Example:
            >>> sim.set_output_dir("./sim")
            >>> result = sim.mesh(preset="fine", planar_conductors=True)
            >>> print(f"Mesh saved to: {result.mesh_path}")
        """
        from gsim.palace.ports import extract_ports

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        component = self.geometry.component if self.geometry else None

        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            air_above=air_above,
            fmax=fmax,
            planar_conductors=planar_conductors,
            show_gui=show_gui,
        )

        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        output_dir = self._output_dir

        stack = self._resolve_stack()

        palace_ports = []
        if self.ports or self.cpw_ports:
            self._configure_ports_on_component(stack)
            palace_ports = extract_ports(component, stack)

        return self._generate_mesh_internal(
            output_dir=output_dir,
            mesh_config=mesh_config,
            ports=palace_ports,
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
        """Run eigenmode simulation on GDSFactory+ cloud.

        Args:
            output_dir: Directory containing mesh files
            verbose: Print progress messages

        Returns:
            Dict mapping result filenames to local paths

        Raises:
            NotImplementedError: Eigenmode is not yet fully implemented
        """
        raise NotImplementedError(
            "Eigenmode simulation is not yet fully implemented on cloud. "
            "Use DrivenSim for S-parameter extraction."
        )


__all__ = ["EigenmodeSim"]
