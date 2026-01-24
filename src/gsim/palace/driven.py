"""Driven simulation class for frequency-domain S-parameter extraction.

This module provides the DrivenSim class for running frequency-sweep
simulations to extract S-parameters.
"""

from __future__ import annotations

import logging
import tempfile
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.models import (
    CPWPortConfig,
    DrivenConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PortConfig,
    SimulationResult,
    ValidationResult,
)

if TYPE_CHECKING:
    from gdsfactory.component import Component


class DrivenSim(BaseModel):
    """Frequency-domain driven simulation for S-parameter extraction.

    This class configures and runs driven simulations that sweep through
    frequencies to compute S-parameters. Uses composition (no inheritance)
    with shared Geometry and Stack components from gsim.common.

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
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        ports: List of single-element port configurations
        cpw_ports: List of CPW (two-element) port configurations
        driven: Driven simulation configuration (frequencies, etc.)
        mesh: Mesh configuration
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

    # Port configurations
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)

    # Driven simulation config
    driven: DrivenConfig = Field(default_factory=DrivenConfig)

    # Mesh config
    mesh_config: MeshConfig = Field(default_factory=MeshConfig.default)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_ports: bool = PrivateAttr(default=False)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)

    # -------------------------------------------------------------------------
    # Output directory
    # -------------------------------------------------------------------------

    def set_output_dir(self, path: str | Path) -> None:
        """Set the output directory for mesh and config files.

        Args:
            path: Directory path for output files

        Example:
            >>> sim.set_output_dir("./palace-sim")
        """
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path | None:
        """Get the current output directory."""
        return self._output_dir

    # -------------------------------------------------------------------------
    # Geometry methods
    # -------------------------------------------------------------------------

    def set_geometry(self, component: Component) -> None:
        """Set the gdsfactory component for simulation.

        Args:
            component: gdsfactory Component to simulate

        Example:
            >>> sim.set_geometry(my_component)
        """
        self.geometry = Geometry(component=component)

    @property
    def component(self) -> Component | None:
        """Get the current component (for backward compatibility)."""
        return self.geometry.component if self.geometry else None

    # Backward compatibility alias
    @property
    def _component(self) -> Component | None:
        """Internal component access (backward compatibility)."""
        return self.component

    # -------------------------------------------------------------------------
    # Stack methods
    # -------------------------------------------------------------------------

    def set_stack(
        self,
        *,
        yaml_path: str | Path | None = None,
        air_above: float = 200.0,
        substrate_thickness: float = 2.0,
        include_substrate: bool = False,
        **kwargs,
    ) -> None:
        """Configure the layer stack.

        If yaml_path is provided, loads stack from YAML file.
        Otherwise, extracts from active PDK with given parameters.

        Args:
            yaml_path: Path to custom YAML stack file
            air_above: Air box height above top metal in um
            substrate_thickness: Thickness below z=0 in um
            include_substrate: Include lossy silicon substrate
            **kwargs: Additional args passed to extract_layer_stack

        Example:
            >>> sim.set_stack(air_above=300.0, substrate_thickness=2.0)
        """
        self._stack_kwargs = {
            "yaml_path": yaml_path,
            "air_above": air_above,
            "substrate_thickness": substrate_thickness,
            "include_substrate": include_substrate,
            **kwargs,
        }
        # Stack will be resolved lazily during mesh() or simulate()
        self.stack = None

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
    # Material methods
    # -------------------------------------------------------------------------

    def set_material(
        self,
        name: str,
        *,
        type: Literal["conductor", "dielectric", "semiconductor"] | None = None,
        conductivity: float | None = None,
        permittivity: float | None = None,
        loss_tangent: float | None = None,
    ) -> None:
        """Override or add material properties.

        Args:
            name: Material name
            type: Material type (conductor, dielectric, semiconductor)
            conductivity: Conductivity in S/m (for conductors)
            permittivity: Relative permittivity (for dielectrics)
            loss_tangent: Dielectric loss tangent

        Example:
            >>> sim.set_material("aluminum", type="conductor", conductivity=3.8e7)
            >>> sim.set_material("sio2", type="dielectric", permittivity=3.9)
        """
        # Determine type if not provided
        if type is None:
            if conductivity is not None and conductivity > 1e4:
                type = "conductor"
            elif permittivity is not None:
                type = "dielectric"
            else:
                type = "dielectric"

        self.materials[name] = MaterialConfig(
            type=type,
            conductivity=conductivity,
            permittivity=permittivity,
            loss_tangent=loss_tangent,
        )

    def set_numerical(
        self,
        *,
        order: int = 2,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = "Default",
        preconditioner: Literal["Default", "AMS", "BoomerAMG"] = "Default",
        device: Literal["CPU", "GPU"] = "CPU",
        num_processors: int | None = None,
    ) -> None:
        """Configure numerical solver parameters.

        Args:
            order: Finite element order (1-4)
            tolerance: Linear solver tolerance
            max_iterations: Maximum solver iterations
            solver_type: Linear solver type
            preconditioner: Preconditioner type
            device: Compute device (CPU or GPU)
            num_processors: Number of processors (None = auto)

        Example:
            >>> sim.set_numerical(order=3, tolerance=1e-8)
        """
        self.numerical = NumericalConfig(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
            num_processors=num_processors,
        )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate(self) -> ValidationResult:
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
    # Internal helpers
    # -------------------------------------------------------------------------

    def _resolve_stack(self) -> LayerStack:
        """Resolve the layer stack from PDK or YAML.

        Returns:
            Legacy LayerStack object for mesh generation
        """
        from gsim.common.stack import get_stack

        yaml_path = self._stack_kwargs.pop("yaml_path", None)
        legacy_stack = get_stack(yaml_path=yaml_path, **self._stack_kwargs)

        # Restore yaml_path for potential re-resolution
        self._stack_kwargs["yaml_path"] = yaml_path

        # Apply material overrides
        for name, props in self.materials.items():
            legacy_stack.materials[name] = props.to_dict()

        # Store the LayerStack
        self.stack = legacy_stack

        return legacy_stack

    def _configure_ports_on_component(self, stack: LayerStack) -> None:
        """Configure ports on the component using legacy functions."""
        from gsim.palace.ports import (
            configure_cpw_port,
            configure_inplane_port,
            configure_via_port,
        )

        component = self.geometry.component if self.geometry else None
        if component is None:
            raise ValueError("No component set")

        # Configure regular ports
        for port_config in self.ports:
            if port_config.name is None:
                continue

            # Find matching gdsfactory port
            gf_port = None
            for p in component.ports:
                if p.name == port_config.name:
                    gf_port = p
                    break

            if gf_port is None:
                raise ValueError(
                    f"Port '{port_config.name}' not found on component. "
                    f"Available ports: {[p.name for p in component.ports]}"
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
            for p in component.ports:
                if p.name == cpw_config.upper:
                    port_upper = p
                    break
            if port_upper is None:
                raise ValueError(
                    f"CPW upper port '{cpw_config.upper}' not found. "
                    f"Available: {[p.name for p in component.ports]}"
                )

            # Find lower port
            port_lower = None
            for p in component.ports:
                if p.name == cpw_config.lower:
                    port_lower = p
                    break
            if port_lower is None:
                raise ValueError(
                    f"CPW lower port '{cpw_config.lower}' not found. "
                    f"Available: {[p.name for p in component.ports]}"
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

    def _build_mesh_config(
        self,
        preset: Literal["coarse", "default", "fine"] | None,
        refined_mesh_size: float | None,
        max_mesh_size: float | None,
        margin: float | None,
        air_above: float | None,
        fmax: float | None,
        show_gui: bool,
    ) -> MeshConfig:
        """Build mesh config from preset with optional overrides."""
        # Build mesh config from preset
        if preset == "coarse":
            mesh_config = MeshConfig.coarse()
        elif preset == "fine":
            mesh_config = MeshConfig.fine()
        else:
            mesh_config = MeshConfig.default()

        # Track overrides for warning
        overrides = []
        if preset is not None:
            if refined_mesh_size is not None:
                overrides.append(f"refined_mesh_size={refined_mesh_size}")
            if max_mesh_size is not None:
                overrides.append(f"max_mesh_size={max_mesh_size}")
            if margin is not None:
                overrides.append(f"margin={margin}")
            if air_above is not None:
                overrides.append(f"air_above={air_above}")
            if fmax is not None:
                overrides.append(f"fmax={fmax}")

            if overrides:
                warnings.warn(
                    f"Preset '{preset}' values overridden by: {', '.join(overrides)}",
                    stacklevel=4,
                )

        # Apply overrides
        if refined_mesh_size is not None:
            mesh_config.refined_mesh_size = refined_mesh_size
        if max_mesh_size is not None:
            mesh_config.max_mesh_size = max_mesh_size
        if margin is not None:
            mesh_config.margin = margin
        if air_above is not None:
            mesh_config.air_above = air_above
        if fmax is not None:
            mesh_config.fmax = fmax
        mesh_config.show_gui = show_gui

        return mesh_config

    def _generate_mesh_internal(
        self,
        output_dir: Path,
        mesh_config: MeshConfig,
        ports: list,
        driven_config: Any,
        model_name: str,
        verbose: bool,
        write_config: bool = True,
    ) -> SimulationResult:
        """Internal mesh generation."""
        from gsim.palace.mesh import MeshConfig as LegacyMeshConfig
        from gsim.palace.mesh import generate_mesh

        component = self.geometry.component if self.geometry else None

        # Get effective fmax from driven config if mesh doesn't specify
        effective_fmax = mesh_config.fmax
        if driven_config is not None and mesh_config.fmax == 100e9:
            effective_fmax = driven_config.fmax

        legacy_mesh_config = LegacyMeshConfig(
            refined_mesh_size=mesh_config.refined_mesh_size,
            max_mesh_size=mesh_config.max_mesh_size,
            cells_per_wavelength=mesh_config.cells_per_wavelength,
            margin=mesh_config.margin,
            air_above=mesh_config.air_above,
            fmax=effective_fmax,
            show_gui=mesh_config.show_gui,
            preview_only=mesh_config.preview_only,
        )

        # Resolve stack
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
            driven_config=driven_config,
            write_config=write_config,
        )

        # Store mesh_result for deferred config generation
        self._last_mesh_result = mesh_result
        self._last_ports = ports

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
        self._configure_ports_on_component(stack)
        return extract_ports(component, stack)

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

        Opens the gmsh GUI to visualize the mesh interactively.

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

        # Validate configuration
        validation = self.validate()
        if not validation.valid:
            raise ValueError(
                f"Invalid configuration:\n" + "\n".join(validation.errors)
            )

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

        # Resolve stack
        stack = self._resolve_stack()

        # Get ports
        ports = self._get_ports_for_preview(stack)

        # Build legacy mesh config with preview mode
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

        # Generate mesh in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_mesh(
                component=component,
                stack=stack,
                ports=ports,
                output_dir=tmpdir,
                config=legacy_mesh_config,
            )

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def show_stack(self) -> None:
        """Print the layer stack table.

        Example:
            >>> sim.show_stack()
        """
        from gsim.common.stack import print_stack_table

        if self.stack is None:
            self._resolve_stack()

        if self.stack is not None:
            print_stack_table(self.stack)

    def plot_stack(self) -> None:
        """Plot the layer stack visualization.

        Example:
            >>> sim.plot_stack()
        """
        from gsim.common.stack import plot_stack

        if self.stack is None:
            self._resolve_stack()

        if self.stack is not None:
            plot_stack(self.stack)

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

        Only generates the mesh file (palace.msh). Config is generated
        separately with write_config().

        Requires set_output_dir() to be called first.

        Args:
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
            SimulationResult with mesh path

        Raises:
            ValueError: If output_dir not set or configuration is invalid

        Example:
            >>> sim.set_output_dir("./sim")
            >>> result = sim.mesh(preset="fine")
            >>> print(f"Mesh saved to: {result.mesh_path}")
        """
        from gsim.palace.ports import extract_ports

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        component = self.geometry.component if self.geometry else None

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

        output_dir = self._output_dir

        # Resolve stack and configure ports
        stack = self._resolve_stack()
        self._configure_ports_on_component(stack)

        # Extract ports
        palace_ports = extract_ports(component, stack)

        # Generate mesh (config is written separately by simulate() or write_config())
        return self._generate_mesh_internal(
            output_dir=output_dir,
            mesh_config=mesh_config,
            ports=palace_ports,
            driven_config=self.driven,
            model_name=model_name,
            verbose=verbose,
            write_config=False,
        )

    def write_config(self) -> Path:
        """Write Palace config.json after mesh generation.

        Use this when mesh() was called with write_config=False.

        Returns:
            Path to the generated config.json

        Raises:
            ValueError: If mesh() hasn't been called yet

        Example:
            >>> result = sim.mesh("./sim", write_config=False)
            >>> config_path = sim.write_config()
        """
        from gsim.palace.mesh.generator import write_config as gen_write_config

        if self._last_mesh_result is None:
            raise ValueError("No mesh result. Call mesh() first.")

        if not self._last_mesh_result.groups:
            raise ValueError(
                "Mesh result has no groups data. "
                "Was mesh() called with write_config=True already?"
            )

        stack = self._resolve_stack()
        config_path = gen_write_config(
            mesh_result=self._last_mesh_result,
            stack=stack,
            ports=self._last_ports,
            driven_config=self.driven,
        )

        return config_path

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def simulate(
        self,
        *,
        verbose: bool = True,
    ) -> dict[str, Path]:
        """Run simulation on GDSFactory+ cloud.

        Requires mesh() and write_config() to be called first.

        Args:
            verbose: Print progress messages

        Returns:
            Dict mapping result filenames to local paths

        Raises:
            ValueError: If output_dir not set
            FileNotFoundError: If mesh or config files don't exist
            RuntimeError: If simulation fails

        Example:
            >>> results = sim.simulate()
            >>> print(f"S-params saved to: {results['port-S.csv']}")
        """
        from gsim.gcloud import run_simulation

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        output_dir = self._output_dir

        return run_simulation(
            output_dir,
            job_type="palace",
            verbose=verbose,
        )

    def simulate_local(
        self,
        *,
        verbose: bool = True,
    ) -> dict[str, Path]:
        """Run simulation locally using Palace.

        Requires mesh() and write_config() to be called first,
        and Palace to be installed locally.

        Args:
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
