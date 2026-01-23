"""Eigenmode simulation class for resonance/mode finding.

This module provides the EigenmodeSim class for finding resonant
frequencies and mode shapes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, PrivateAttr

from gsim.palace.base import SimBase
from gsim.palace.models import (
    CPWPortConfig,
    EigenmodeConfig,
    PortConfig,
    SimulationResult,
    ValidationResult,
)

if TYPE_CHECKING:
    from gsim.palace.stack.extractor import LayerStack


class EigenmodeSim(SimBase):
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
        >>> sim.mesh("./sim", preset="default")
        >>> results = sim.simulate()

    Attributes:
        ports: List of single-element port configurations
        cpw_ports: List of CPW (two-element) port configurations
        eigenmode: Eigenmode simulation configuration
    """

    # Port configurations (eigenmode can have ports for Q-factor calculation)
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)

    # Eigenmode simulation config
    eigenmode: EigenmodeConfig = Field(default_factory=EigenmodeConfig)

    # Internal port state
    _configured_ports: bool = PrivateAttr(default=False)

    # -------------------------------------------------------------------------
    # Port methods (same as DrivenSim)
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

    def validate(self) -> ValidationResult:
        """Validate the simulation configuration.

        Returns:
            ValidationResult with validation status and messages
        """
        result = self._validate_base()
        errors = list(result.errors)
        warnings_list = list(result.warnings)

        # Eigenmode simulations may not require ports
        if not self.ports and not self.cpw_ports:
            warnings_list.append(
                "No ports configured. Eigenmode will find all modes without port loading."
            )

        # Validate port configurations
        for port in self.ports:
            if port.geometry == "inplane" and port.layer is None:
                errors.append(f"Port '{port.name}': inplane ports require 'layer'")
            if port.geometry == "via":
                if port.from_layer is None or port.to_layer is None:
                    errors.append(
                        f"Port '{port.name}': via ports require 'from_layer' and 'to_layer'"
                    )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Port configuration
    # -------------------------------------------------------------------------

    def _configure_ports_on_component(self, stack: LayerStack) -> None:
        """Configure ports on the component."""
        from gsim.palace.ports import (
            configure_cpw_port,
            configure_inplane_port,
            configure_via_port,
        )

        for port_config in self.ports:
            if port_config.name is None:
                continue

            gf_port = None
            for p in self._component.ports:
                if p.name == port_config.name:
                    gf_port = p
                    break

            if gf_port is None:
                raise ValueError(
                    f"Port '{port_config.name}' not found on component. "
                    f"Available: {[p.name for p in self._component.ports]}"
                )

            if port_config.geometry == "inplane":
                configure_inplane_port(
                    gf_port,
                    layer=port_config.layer,
                    length=port_config.length or gf_port.width,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                )
            elif port_config.geometry == "via":
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
            for p in self._component.ports:
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

    def _get_ports_for_preview(self, stack: LayerStack) -> list:
        """Get ports for preview."""
        from gsim.palace.ports import extract_ports

        if self.ports or self.cpw_ports:
            self._configure_ports_on_component(stack)
            return extract_ports(self._component, stack)
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
            fmax: Max frequency for mesh sizing (Hz)
            show_gui: Show gmsh GUI during meshing
            model_name: Base name for output files
            verbose: Print progress messages

        Returns:
            SimulationResult with mesh and config paths
        """
        from gsim.palace.ports import extract_ports

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

        palace_ports = []
        if self.ports or self.cpw_ports:
            self._configure_ports_on_component(stack)
            palace_ports = extract_ports(self._component, stack)

        # Note: eigenmode uses eigenmode_config, not driven_config
        # The mesh generator needs to be updated to handle this
        return self._generate_mesh_internal(
            output_dir=output_dir,
            mesh_config=mesh_config,
            ports=palace_ports,
            driven_config=None,  # Eigenmode doesn't use driven config
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
