"""Driven simulation class for frequency-domain S-parameter extraction.

This module provides the DrivenSim class for running frequency-sweep
simulations to extract S-parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, PrivateAttr

from gsim.palace.base import SimBase
from gsim.palace.models import (
    CPWPortConfig,
    DrivenConfig,
    PortConfig,
    SimulationResult,
    ValidationResult,
)

if TYPE_CHECKING:
    from gsim.palace.stack.extractor import LayerStack


class DrivenSim(SimBase):
    """Frequency-domain driven simulation for S-parameter extraction.

    This class configures and runs driven simulations that sweep through
    frequencies to compute S-parameters.

    Example:
        >>> from gsim.palace import DrivenSim
        >>>
        >>> sim = DrivenSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack(air_above=300.0)
        >>> sim.add_cpw_port("P2", "P1", layer="topmetal2", length=5.0)
        >>> sim.add_cpw_port("P3", "P4", layer="topmetal2", length=5.0)
        >>> sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
        >>> sim.mesh("./sim", preset="default")
        >>> results = sim.simulate()

    Attributes:
        ports: List of single-element port configurations
        cpw_ports: List of CPW (two-element) port configurations
        driven: Driven simulation configuration (frequencies, etc.)
    """

    # Port configurations
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)

    # Driven simulation config
    driven: DrivenConfig = Field(default_factory=DrivenConfig)

    # Internal port state for legacy API compatibility
    _configured_ports: bool = PrivateAttr(default=False)

    # -------------------------------------------------------------------------
    # Port methods
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
            >>> sim.add_port("feed", from_layer="metal1", to_layer="topmetal2", geometry="via")
        """
        # Remove existing config for this port if any
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

        CPW ports consist of two elements (upper and lower gaps) that are
        excited with opposite E-field directions to create the CPW mode.

        Args:
            upper: Name of the upper gap port on the component
            lower: Name of the lower gap port on the component
            layer: Target conductor layer (e.g., "topmetal2")
            length: Port extent along direction (um)
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited
            name: Optional name for the CPW port (default: "cpw_{lower}")

        Example:
            >>> sim.add_cpw_port("P2", "P1", layer="topmetal2", length=5.0)
        """
        # Remove existing CPW port with same elements if any
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
    # Driven configuration
    # -------------------------------------------------------------------------

    def set_driven(
        self,
        *,
        fmin: float = 1e9,
        fmax: float = 100e9,
        num_points: int = 40,
        scale: Literal["linear", "log"] = "linear",
        adaptive_tol: float = 0.02,
        adaptive_max_samples: int = 20,
        compute_s_params: bool = True,
        reference_impedance: float = 50.0,
        excitation_port: str | None = None,
    ) -> None:
        """Configure driven (frequency sweep) simulation.

        Args:
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            num_points: Number of frequency points
            scale: "linear" or "log" frequency spacing
            adaptive_tol: Adaptive frequency tolerance (0 disables adaptive)
            adaptive_max_samples: Max samples for adaptive refinement
            compute_s_params: Compute S-parameters
            reference_impedance: Reference impedance for S-params (Ohms)
            excitation_port: Port to excite (None = first port)

        Example:
            >>> sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
        """
        self.driven = DrivenConfig(
            fmin=fmin,
            fmax=fmax,
            num_points=num_points,
            scale=scale,
            adaptive_tol=adaptive_tol,
            adaptive_max_samples=adaptive_max_samples,
            compute_s_params=compute_s_params,
            reference_impedance=reference_impedance,
            excitation_port=excitation_port,
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> ValidationResult:
        """Validate the simulation configuration.

        Returns:
            ValidationResult with validation status and messages
        """
        # Start with base validation
        result = self._validate_base()
        errors = list(result.errors)
        warnings_list = list(result.warnings)

        # Check ports
        has_ports = bool(self.ports) or bool(self.cpw_ports)
        if not has_ports:
            warnings_list.append(
                "No ports configured. Call add_port() or add_cpw_port()."
            )
        else:
            # Validate port configurations
            for port in self.ports:
                if port.geometry == "inplane" and port.layer is None:
                    errors.append(
                        f"Port '{port.name}': inplane ports require 'layer'"
                    )
                if port.geometry == "via":
                    if port.from_layer is None or port.to_layer is None:
                        errors.append(
                            f"Port '{port.name}': via ports require "
                            "'from_layer' and 'to_layer'"
                        )

            # Validate CPW ports
            for cpw in self.cpw_ports:
                if not cpw.layer:
                    errors.append(
                        f"CPW port ({cpw.upper}, {cpw.lower}): 'layer' is required"
                    )

        # Validate excitation port if specified
        if self.driven.excitation_port is not None:
            port_names = [p.name for p in self.ports]
            cpw_names = [cpw.effective_name for cpw in self.cpw_ports]
            all_port_names = port_names + cpw_names
            if self.driven.excitation_port not in all_port_names:
                errors.append(
                    f"Excitation port '{self.driven.excitation_port}' not found. "
                    f"Available: {all_port_names}"
                )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Port configuration on component
    # -------------------------------------------------------------------------

    def _configure_ports_on_component(self, stack: LayerStack) -> None:
        """Configure ports on the component using legacy functions."""
        from gsim.palace.ports import (
            configure_cpw_port,
            configure_inplane_port,
            configure_via_port,
        )

        # Configure regular ports
        for port_config in self.ports:
            if port_config.name is None:
                continue

            # Find matching gdsfactory port
            gf_port = None
            for p in self._component.ports:
                if p.name == port_config.name:
                    gf_port = p
                    break

            if gf_port is None:
                raise ValueError(
                    f"Port '{port_config.name}' not found on component. "
                    f"Available ports: {[p.name for p in self._component.ports]}"
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

        # Configure CPW ports
        for cpw_config in self.cpw_ports:
            # Find upper port
            port_upper = None
            for p in self._component.ports:
                if p.name == cpw_config.upper:
                    port_upper = p
                    break
            if port_upper is None:
                raise ValueError(
                    f"CPW upper port '{cpw_config.upper}' not found. "
                    f"Available: {[p.name for p in self._component.ports]}"
                )

            # Find lower port
            port_lower = None
            for p in self._component.ports:
                if p.name == cpw_config.lower:
                    port_lower = p
                    break
            if port_lower is None:
                raise ValueError(
                    f"CPW lower port '{cpw_config.lower}' not found. "
                    f"Available: {[p.name for p in self._component.ports]}"
                )

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

        self._configure_ports_on_component(stack)
        return extract_ports(self._component, stack)

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
        """Generate the mesh and configuration files for Palace simulation.

        Args:
            output_dir: Directory for output files
            preset: Mesh quality preset ("coarse", "default", "fine")
            refined_mesh_size: Mesh size near conductors (um), overrides preset
            max_mesh_size: Max mesh size in air/dielectric (um), overrides preset
            margin: XY margin around design (um), overrides preset
            air_above: Air above top metal (um), overrides preset
            fmax: Max frequency for mesh sizing (Hz), overrides preset
            show_gui: Show gmsh GUI during meshing
            model_name: Base name for output files
            verbose: Print progress messages

        Returns:
            SimulationResult with mesh and config paths

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> result = sim.mesh("./sim", preset="fine")
            >>> print(f"Mesh saved to: {result.mesh_path}")
        """
        from gsim.palace.ports import extract_ports

        # Build mesh config
        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            air_above=air_above,
            fmax=fmax,
            show_gui=show_gui,
        )

        # Validate configuration
        validation = self.validate()
        if not validation.valid:
            raise ValueError(
                f"Invalid configuration:\n" + "\n".join(validation.errors)
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Store output_dir for simulate() calls
        self._output_dir = output_dir

        # Resolve stack and configure ports
        stack = self._resolve_stack()
        self._configure_ports_on_component(stack)

        # Extract ports
        palace_ports = extract_ports(self._component, stack)

        # Generate mesh
        return self._generate_mesh_internal(
            output_dir=output_dir,
            mesh_config=mesh_config,
            ports=palace_ports,
            driven_config=self.driven,
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
        """Run simulation on GDSFactory+ cloud.

        Requires mesh files to exist in output_dir (call mesh() first).

        Args:
            output_dir: Directory containing mesh files. If None, uses the
                directory from the last mesh() call.
            verbose: Print progress messages

        Returns:
            Dict mapping result filenames to local paths

        Raises:
            ValueError: If output_dir not provided and mesh() not called
            FileNotFoundError: If output directory doesn't exist
            RuntimeError: If simulation fails

        Example:
            >>> results = sim.simulate()
            >>> print(f"S-params saved to: {results['port-S.csv']}")
        """
        from gsim.gcloud import run_simulation

        if output_dir is None:
            if self._output_dir is None:
                raise ValueError(
                    "output_dir not provided and mesh() has not been called. "
                    "Either call mesh() first or provide output_dir explicitly."
                )
            output_dir = self._output_dir

        output_dir = Path(output_dir)

        if verbose:
            print(f"Running simulation on cloud from {output_dir}...")

        return run_simulation(
            output_dir,
            job_type="palace",
            verbose=verbose,
        )

    def simulate_local(
        self,
        output_dir: str | Path | None = None,
        *,
        verbose: bool = True,
    ) -> dict[str, Path]:
        """Run simulation locally using Palace.

        Requires mesh files to exist in output_dir (call mesh() first)
        and Palace to be installed locally.

        Args:
            output_dir: Directory containing mesh files. If None, uses the
                directory from the last mesh() call.
            verbose: Print progress messages

        Returns:
            Dict mapping result filenames to local paths

        Raises:
            NotImplementedError: Local simulation is not yet implemented
        """
        raise NotImplementedError(
            "Local simulation is not yet implemented. "
            "Use simulate() to run on GDSFactory+ cloud."
        )


__all__ = ["DrivenSim"]
