"""Central PalaceSim class with fluent API for Palace EM simulation.

This module provides the main entry point for configuring and running
Palace simulations using a fluent, chainable API.

Usage:
    from gsim.palace import PalaceSim

    # Minimal usage with defaults
    result = (
        PalaceSim()
        .geometry(component)
        .stack()  # Uses active PDK
        .ports(layer="topmetal2", length=5.0)
        .mesh(preset="default")
        .run("./sim")
    )

    # Full customization
    result = (
        PalaceSim()
        .geometry(component)
        .stack(air_above=300.0, include_substrate=True)
        .material("aluminum", conductivity=4.0e7)
        .mesh(preset="fine", margin=100.0)
        .physics(fmax=200e9, problem_type="driven")
        .numerical(order=3, tolerance=1e-8)
        .port("o1", layer="topmetal2", length=5.0, excited=True)
        .port("o2", layer="topmetal2", length=5.0, excited=False)
        .run("./sim", cloud=True)
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, PrivateAttr

from gsim.palace.models import (
    DrivenConfig,
    EigenmodeConfig,
    ElectrostaticConfig,
    LayerStackModel,
    MagnetostaticConfig,
    MaterialPropertiesModel,
    MeshConfigModel,
    NumericalConfig,
    PhysicsConfig,
    PortConfigModel,
    ProblemType,
    SimulationResult,
    TerminalConfig,
    TransientConfig,
    ValidationResultModel,
    WavePortConfig,
)

if TYPE_CHECKING:
    from gdsfactory.component import Component


class PalaceSim(BaseModel):
    """Central class for configuring and running Palace EM simulations.

    This class provides a fluent API where methods mutate internal state
    and return `self` for method chaining.

    Example:
        >>> result = (
        ...     PalaceSim()
        ...     .geometry(component)
        ...     .stack()
        ...     .ports(layer="topmetal2", length=5.0)
        ...     .mesh(preset="default")
        ...     .run("./sim")
        ... )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Private state (using PrivateAttr for non-Pydantic fields)
    _component: Component | None = PrivateAttr(default=None)
    _stack: LayerStackModel | None = PrivateAttr(default=None)
    _mesh_config: MeshConfigModel | None = PrivateAttr(default=None)
    _physics: PhysicsConfig | None = PrivateAttr(default=None)
    _numerical: NumericalConfig | None = PrivateAttr(default=None)
    _port_configs: list[PortConfigModel] = PrivateAttr(default_factory=list)
    _cpw_port_configs: list[dict] = PrivateAttr(default_factory=list)
    _material_overrides: dict[str, MaterialPropertiesModel] = PrivateAttr(
        default_factory=dict
    )

    # Stack extraction settings
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Problem type configuration (new API)
    _problem_type: ProblemType = PrivateAttr(default="driven")
    _driven_config: DrivenConfig | None = PrivateAttr(default=None)
    _eigenmode_config: EigenmodeConfig | None = PrivateAttr(default=None)
    _electrostatic_config: ElectrostaticConfig | None = PrivateAttr(default=None)
    _magnetostatic_config: MagnetostaticConfig | None = PrivateAttr(default=None)
    _transient_config: TransientConfig | None = PrivateAttr(default=None)

    # Additional port types (new API)
    _waveport_configs: list[WavePortConfig] = PrivateAttr(default_factory=list)
    _terminal_configs: list[TerminalConfig] = PrivateAttr(default_factory=list)

    # -------------------------------------------------------------------------
    # Fluent configuration methods
    # -------------------------------------------------------------------------

    def geometry(self, component: Component) -> Self:
        """Set the gdsfactory component for simulation.

        Args:
            component: gdsfactory Component to simulate

        Returns:
            Self for method chaining
        """
        self._component = component
        return self

    def stack(
        self,
        *,
        yaml_path: str | Path | None = None,
        air_above: float = 200.0,
        substrate_thickness: float = 2.0,
        include_substrate: bool = False,
        **kwargs,
    ) -> Self:
        """Configure the layer stack.

        If yaml_path is provided, loads stack from YAML file.
        Otherwise, extracts from active PDK with given parameters.

        Args:
            yaml_path: Path to custom YAML stack file
            air_above: Air box height above top metal in um
            substrate_thickness: Thickness below z=0 in um
            include_substrate: Include lossy silicon substrate
            **kwargs: Additional args passed to extract_layer_stack

        Returns:
            Self for method chaining
        """
        self._stack_kwargs = {
            "yaml_path": yaml_path,
            "air_above": air_above,
            "substrate_thickness": substrate_thickness,
            "include_substrate": include_substrate,
            **kwargs,
        }
        # Stack will be resolved lazily during run()
        self._stack = None
        return self

    def material(
        self,
        name: str,
        *,
        type: Literal["conductor", "dielectric", "semiconductor"] | None = None,
        conductivity: float | None = None,
        permittivity: float | None = None,
        loss_tangent: float | None = None,
    ) -> Self:
        """Override or add material properties.

        Args:
            name: Material name
            type: Material type (conductor, dielectric, semiconductor)
            conductivity: Conductivity in S/m (for conductors)
            permittivity: Relative permittivity (for dielectrics)
            loss_tangent: Dielectric loss tangent

        Returns:
            Self for method chaining
        """
        # Determine type if not provided
        if type is None:
            if conductivity is not None and conductivity > 1e4:
                type = "conductor"
            elif permittivity is not None:
                type = "dielectric"
            else:
                type = "dielectric"

        self._material_overrides[name] = MaterialPropertiesModel(
            type=type,
            conductivity=conductivity,
            permittivity=permittivity,
            loss_tangent=loss_tangent,
        )
        return self

    def materials(self, overrides: dict[str, dict]) -> Self:
        """Override multiple material properties at once.

        Args:
            overrides: Dict mapping material name to properties dict

        Returns:
            Self for method chaining
        """
        for name, props in overrides.items():
            self.material(name, **props)
        return self

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
        **kwargs,
    ) -> Self:
        """Configure mesh generation parameters.

        Args:
            preset: Mesh quality preset ("coarse", "default", "fine")
            refined_mesh_size: Mesh size near conductors (um)
            max_mesh_size: Max mesh size in air/dielectric (um)
            margin: XY margin around design (um)
            air_above: Air above top metal (um)
            fmax: Max frequency for mesh sizing (Hz)
            show_gui: Show gmsh GUI during meshing
            **kwargs: Additional MeshConfigModel parameters

        Returns:
            Self for method chaining
        """
        # Start with preset or defaults
        if preset == "coarse":
            config = MeshConfigModel.coarse()
        elif preset == "fine":
            config = MeshConfigModel.fine()
        elif preset == "default" or preset is None:
            config = MeshConfigModel.default()
        else:
            config = MeshConfigModel()

        # Apply overrides
        if refined_mesh_size is not None:
            config.refined_mesh_size = refined_mesh_size
        if max_mesh_size is not None:
            config.max_mesh_size = max_mesh_size
        if margin is not None:
            config.margin = margin
        if air_above is not None:
            config.air_above = air_above
        if fmax is not None:
            config.fmax = fmax
        config.show_gui = show_gui

        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self._mesh_config = config
        return self

    def physics(
        self,
        *,
        problem_type: Literal["driven", "eigenmode"] = "driven",
        fmin: float = 1e9,
        fmax: float = 100e9,
        num_frequency_points: int = 40,
        frequency_scale: Literal["linear", "log"] = "linear",
        num_modes: int = 10,
        target_frequency: float | None = None,
        compute_s_params: bool = True,
        reference_impedance: float = 50.0,
        adaptive_refinement: bool = False,
        refinement_tolerance: float = 0.01,
    ) -> Self:
        """Configure physics/solver settings.

        DEPRECATED: Use .driven(), .eigenmode(), .electrostatic(),
        .magnetostatic(), or .transient() instead.

        Args:
            problem_type: "driven" for frequency sweep, "eigenmode" for resonance
            fmin: Minimum frequency in Hz (for driven)
            fmax: Maximum frequency in Hz (for driven)
            num_frequency_points: Number of frequency points (for driven)
            frequency_scale: "linear" or "log" frequency spacing
            num_modes: Number of modes to find (for eigenmode)
            target_frequency: Target frequency for eigenmode search (Hz)
            compute_s_params: Compute S-parameters (for driven)
            reference_impedance: Reference impedance for S-params (Ohms)
            adaptive_refinement: Use adaptive frequency refinement
            refinement_tolerance: Tolerance for adaptive refinement

        Returns:
            Self for method chaining
        """
        warnings.warn(
            "physics() is deprecated. Use driven(), eigenmode(), electrostatic(), "
            "magnetostatic(), or transient() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Map to new API internally
        self._physics = PhysicsConfig(
            problem_type=problem_type,
            fmin=fmin,
            fmax=fmax,
            num_frequency_points=num_frequency_points,
            frequency_scale=frequency_scale,
            num_modes=num_modes,
            target_frequency=target_frequency,
            compute_s_params=compute_s_params,
            reference_impedance=reference_impedance,
            adaptive_refinement=adaptive_refinement,
            refinement_tolerance=refinement_tolerance,
        )

        # Also set the new API configs for forward compatibility
        self._problem_type = problem_type
        if problem_type == "driven":
            self._driven_config = self._physics.to_driven_config()
        elif problem_type == "eigenmode":
            self._eigenmode_config = self._physics.to_eigenmode_config()

        return self

    def numerical(
        self,
        *,
        order: int = 2,
        tolerance: float = 1e-6,
        max_iterations: int = 400,
        solver_type: Literal["Default", "SuperLU", "STRUMPACK", "MUMPS"] = "Default",
        preconditioner: Literal["Default", "AMS", "BoomerAMG"] = "Default",
        device: Literal["CPU", "GPU"] = "CPU",
        num_processors: int | None = None,
    ) -> Self:
        """Configure numerical solver parameters.

        Args:
            order: Finite element order (1-4)
            tolerance: Linear solver tolerance
            max_iterations: Maximum solver iterations
            solver_type: Linear solver type
            preconditioner: Preconditioner type
            device: Compute device (CPU or GPU)
            num_processors: Number of processors (None = auto)

        Returns:
            Self for method chaining
        """
        self._numerical = NumericalConfig(
            order=order,
            tolerance=tolerance,
            max_iterations=max_iterations,
            solver_type=solver_type,
            preconditioner=preconditioner,
            device=device,
            num_processors=num_processors,
        )
        return self

    def port(
        self,
        name: str,
        *,
        port_type: Literal["lumped", "waveport"] = "lumped",
        geometry: Literal["inplane", "via"] = "inplane",
        layer: str | None = None,
        from_layer: str | None = None,
        to_layer: str | None = None,
        length: float | None = None,
        impedance: float = 50.0,
        excited: bool = True,
    ) -> Self:
        """Add or configure a single port.

        Args:
            name: Port name (must match component port name)
            port_type: "lumped" or "waveport"
            geometry: "inplane" (horizontal) or "via" (vertical)
            layer: Target layer for inplane ports
            from_layer: Bottom layer for via ports
            to_layer: Top layer for via ports
            length: Port extent along direction (um)
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited

        Returns:
            Self for method chaining
        """
        # Remove existing config for this port if any
        self._port_configs = [p for p in self._port_configs if p.name != name]

        self._port_configs.append(
            PortConfigModel(
                name=name,
                port_type=port_type,
                geometry=geometry,
                layer=layer,
                from_layer=from_layer,
                to_layer=to_layer,
                length=length,
                impedance=impedance,
                excited=excited,
            )
        )
        return self

    def ports(
        self,
        *,
        layer: str | None = None,
        length: float | None = None,
        impedance: float = 50.0,
        port_type: Literal["lumped", "waveport"] = "lumped",
        excited: bool = True,
    ) -> Self:
        """Configure all component ports with the same settings.

        This method configures all optical ports on the component as
        inplane lumped ports. For more control, use the `port()` method
        for individual port configuration.

        Args:
            layer: Target conductor layer for all ports
            length: Port extent along direction (um)
            impedance: Port impedance (Ohms)
            port_type: Port type for all ports
            excited: Whether ports are excited

        Returns:
            Self for method chaining
        """
        if self._component is None:
            raise ValueError("Must call .geometry(component) before .ports()")

        # Clear existing port configs
        self._port_configs = []

        # Configure each port
        for gf_port in self._component.ports:
            self._port_configs.append(
                PortConfigModel(
                    name=gf_port.name,
                    port_type=port_type,
                    geometry="inplane",
                    layer=layer,
                    length=length,
                    impedance=impedance,
                    excited=excited,
                )
            )

        return self

    def cpw_port(
        self,
        port_upper: str,
        port_lower: str,
        *,
        layer: str,
        length: float,
        impedance: float = 50.0,
        excited: bool = True,
        name: str | None = None,
    ) -> Self:
        """Configure a CPW (coplanar waveguide) port from two gap ports.

        CPW ports consist of two elements (upper and lower gaps) that are
        excited with opposite E-field directions to create the CPW mode.

        Args:
            port_upper: Name of the upper gap port on the component
            port_lower: Name of the lower gap port on the component
            layer: Target conductor layer (e.g., "topmetal2")
            length: Port extent along direction (um)
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited
            name: Optional name for the CPW port (default: "cpw_{port_lower}")

        Returns:
            Self for method chaining

        Example:
            >>> sim.cpw_port("P2", "P1", layer="topmetal2", length=5.0)
        """
        self._cpw_port_configs.append({
            "port_upper": port_upper,
            "port_lower": port_lower,
            "layer": layer,
            "length": length,
            "impedance": impedance,
            "excited": excited,
            "name": name,
        })
        return self

    # -------------------------------------------------------------------------
    # Problem type methods (new API)
    # -------------------------------------------------------------------------

    def driven(
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
    ) -> Self:
        """Configure driven (frequency sweep) simulation.

        This is the most common simulation type for S-parameter extraction
        and frequency response analysis.

        Args:
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            num_points: Number of frequency points
            scale: "linear" or "log" frequency spacing
            adaptive_tol: Adaptive frequency tolerance (0 disables adaptive)
            adaptive_max_samples: Max samples for adaptive refinement
            compute_s_params: Compute S-parameters
            reference_impedance: Reference impedance for S-params (Ohms)

        Returns:
            Self for method chaining

        Example:
            >>> sim.driven(fmin=1e9, fmax=100e9, num_points=40)
        """
        self._problem_type = "driven"
        self._driven_config = DrivenConfig(
            fmin=fmin,
            fmax=fmax,
            num_points=num_points,
            scale=scale,
            adaptive_tol=adaptive_tol,
            adaptive_max_samples=adaptive_max_samples,
            compute_s_params=compute_s_params,
            reference_impedance=reference_impedance,
        )
        return self

    def eigenmode(
        self,
        *,
        num_modes: int = 10,
        target: float | None = None,
        tolerance: float = 1e-6,
    ) -> Self:
        """Configure eigenmode (resonance) simulation.

        This simulation type finds resonant frequencies and mode shapes.

        Args:
            num_modes: Number of modes to find
            target: Target frequency in Hz for mode search
            tolerance: Eigenvalue solver tolerance

        Returns:
            Self for method chaining

        Example:
            >>> sim.eigenmode(num_modes=10, target=50e9)

        Note:
            This problem type is not yet fully implemented. Currently raises
            NotImplementedError when run() is called.
        """
        self._problem_type = "eigenmode"
        self._eigenmode_config = EigenmodeConfig(
            num_modes=num_modes,
            target=target,
            tolerance=tolerance,
        )
        return self

    def electrostatic(
        self,
        *,
        save_fields: int = 0,
    ) -> Self:
        """Configure electrostatic (capacitance matrix) simulation.

        This simulation type extracts the capacitance matrix between terminals.

        Args:
            save_fields: Number of field solutions to save

        Returns:
            Self for method chaining

        Example:
            >>> sim.terminal("T1", layer="topmetal2")
            >>> sim.terminal("T2", layer="topmetal2")
            >>> sim.electrostatic()

        Note:
            This problem type is not yet fully implemented. Currently raises
            NotImplementedError when run() is called.
        """
        self._problem_type = "electrostatic"
        self._electrostatic_config = ElectrostaticConfig(
            save_fields=save_fields,
        )
        return self

    def magnetostatic(
        self,
        *,
        save_fields: int = 0,
    ) -> Self:
        """Configure magnetostatic (inductance matrix) simulation.

        This simulation type extracts the inductance matrix.

        Args:
            save_fields: Number of field solutions to save

        Returns:
            Self for method chaining

        Note:
            This problem type is not yet fully implemented. Currently raises
            NotImplementedError when run() is called.
        """
        self._problem_type = "magnetostatic"
        self._magnetostatic_config = MagnetostaticConfig(
            save_fields=save_fields,
        )
        return self

    def transient(
        self,
        *,
        max_time: float,
        excitation: Literal["sinusoidal", "gaussian", "ramp", "smoothstep"] = "sinusoidal",
        excitation_freq: float | None = None,
        excitation_width: float | None = None,
        time_step: float | None = None,
    ) -> Self:
        """Configure transient (time-domain) simulation.

        This simulation type performs time-domain analysis with various
        excitation waveforms.

        Args:
            max_time: Maximum simulation time in ns
            excitation: Excitation waveform type
            excitation_freq: Excitation frequency in Hz (for sinusoidal)
            excitation_width: Pulse width in ns (for gaussian)
            time_step: Time step in ns (None = adaptive)

        Returns:
            Self for method chaining

        Note:
            This problem type is not yet fully implemented. Currently raises
            NotImplementedError when run() is called.
        """
        self._problem_type = "transient"
        self._transient_config = TransientConfig(
            max_time=max_time,
            excitation=excitation,
            excitation_freq=excitation_freq,
            excitation_width=excitation_width,
            time_step=time_step,
        )
        return self

    def waveport(
        self,
        name: str,
        *,
        layer: str,
        mode: int = 1,
        excited: bool = True,
        offset: float = 0.0,
    ) -> Self:
        """Add a wave port (domain boundary with mode solving).

        Wave ports are used for domain-boundary ports where mode solving
        is needed. This is an alternative to lumped ports for more accurate
        S-parameter extraction.

        Args:
            name: Port name (must match component port name)
            layer: Target conductor layer
            mode: Mode number to excite
            excited: Whether this port is excited
            offset: De-embedding distance in um

        Returns:
            Self for method chaining

        Note:
            Wave ports are not yet fully implemented. Currently raises
            NotImplementedError when run() is called with wave ports.
        """
        self._waveport_configs.append(
            WavePortConfig(
                name=name,
                layer=layer,
                mode=mode,
                excited=excited,
                offset=offset,
            )
        )
        return self

    def terminal(
        self,
        name: str,
        *,
        layer: str,
    ) -> Self:
        """Add a terminal for electrostatic capacitance extraction.

        Terminals define conductor surfaces for capacitance matrix extraction
        in electrostatic simulations.

        Args:
            name: Terminal name
            layer: Target conductor layer

        Returns:
            Self for method chaining

        Example:
            >>> sim.terminal("T1", layer="topmetal2")
            >>> sim.terminal("T2", layer="topmetal2")
            >>> sim.electrostatic()

        Note:
            Terminals are not yet fully implemented. Currently raises
            NotImplementedError when run() is called with terminals.
        """
        self._terminal_configs.append(
            TerminalConfig(
                name=name,
                layer=layer,
            )
        )
        return self

    # -------------------------------------------------------------------------
    # Validation and execution
    # -------------------------------------------------------------------------

    def validate(self) -> ValidationResultModel:
        """Validate the current simulation configuration.

        Returns:
            ValidationResultModel with validation status and messages
        """
        errors = []
        warnings_list = []

        # Check component
        if self._component is None:
            errors.append("No component set. Call .geometry(component) first.")

        # Check stack
        if self._stack is None and not self._stack_kwargs:
            warnings_list.append(
                "No stack configured. Will use active PDK with defaults."
            )

        # Check problem type configuration
        if self._problem_type not in ("driven", "eigenmode"):
            warnings_list.append(
                f"Problem type '{self._problem_type}' is not yet fully implemented."
            )

        # Check ports
        all_ports = (
            bool(self._port_configs)
            or bool(self._cpw_port_configs)
            or bool(self._waveport_configs)
            or bool(self._terminal_configs)
        )
        if not all_ports:
            warnings_list.append(
                "No ports configured. Call .ports(), .port(), .cpw_port(), "
                ".waveport(), or .terminal()."
            )
        else:
            for port_config in self._port_configs:
                if port_config.geometry == "inplane" and port_config.layer is None:
                    errors.append(
                        f"Port '{port_config.name}': inplane ports require 'layer'"
                    )
                if port_config.geometry == "via":
                    if port_config.from_layer is None or port_config.to_layer is None:
                        errors.append(
                            f"Port '{port_config.name}': via ports require "
                            "'from_layer' and 'to_layer'"
                        )

            # Validate CPW ports
            for cpw_config in self._cpw_port_configs:
                if not cpw_config.get("layer"):
                    errors.append(
                        f"CPW port ({cpw_config['port_upper']}, "
                        f"{cpw_config['port_lower']}): 'layer' is required"
                    )

            # Validate wave ports (not yet implemented)
            if self._waveport_configs:
                warnings_list.append("Wave ports are not yet fully implemented.")

            # Validate terminals (not yet implemented)
            if self._terminal_configs:
                if self._problem_type != "electrostatic":
                    warnings_list.append(
                        "Terminals are only used with electrostatic simulations."
                    )

        # Check mesh config
        if self._mesh_config is None:
            warnings_list.append("No mesh config set. Will use default preset.")

        valid = len(errors) == 0
        return ValidationResultModel(valid=valid, errors=errors, warnings=warnings_list)

    def _resolve_stack(self):
        """Resolve the layer stack from PDK or YAML."""
        from gsim.palace.stack import get_stack

        yaml_path = self._stack_kwargs.pop("yaml_path", None)
        stack = get_stack(yaml_path=yaml_path, **self._stack_kwargs)

        # Apply material overrides
        for name, props in self._material_overrides.items():
            stack.materials[name] = props.to_dict()

        # Convert to Pydantic model
        self._stack = LayerStackModel.from_legacy(stack)

        # Return legacy stack for mesh generation
        return stack

    def _configure_ports_on_component(self, stack):
        """Configure ports on the component using legacy functions."""
        from gsim.palace.ports import (
            configure_cpw_port,
            configure_inplane_port,
            configure_via_port,
        )

        # Configure regular ports
        for port_config in self._port_configs:
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
        for cpw_config in self._cpw_port_configs:
            # Find upper port
            port_upper = None
            for p in self._component.ports:
                if p.name == cpw_config["port_upper"]:
                    port_upper = p
                    break
            if port_upper is None:
                raise ValueError(
                    f"CPW upper port '{cpw_config['port_upper']}' not found. "
                    f"Available: {[p.name for p in self._component.ports]}"
                )

            # Find lower port
            port_lower = None
            for p in self._component.ports:
                if p.name == cpw_config["port_lower"]:
                    port_lower = p
                    break
            if port_lower is None:
                raise ValueError(
                    f"CPW lower port '{cpw_config['port_lower']}' not found. "
                    f"Available: {[p.name for p in self._component.ports]}"
                )

            configure_cpw_port(
                port_upper=port_upper,
                port_lower=port_lower,
                layer=cpw_config["layer"],
                length=cpw_config["length"],
                impedance=cpw_config["impedance"],
                excited=cpw_config["excited"],
                cpw_name=cpw_config["name"],
            )

    def _get_effective_driven_config(self) -> DrivenConfig:
        """Get the effective driven config from new or legacy API."""
        if self._driven_config is not None:
            return self._driven_config
        elif self._physics is not None:
            return self._physics.to_driven_config()
        else:
            # Default driven config
            return DrivenConfig()

    def run(
        self,
        output_dir: str | Path,
        *,
        model_name: str = "palace",
        cloud: bool = False,
        verbose: bool = True,
    ) -> SimulationResult:
        """Run the Palace simulation.

        Args:
            output_dir: Directory for output files
            model_name: Base name for output files
            cloud: Run on GDSFactory+ cloud infrastructure
            verbose: Print progress messages

        Returns:
            SimulationResult with paths and results

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If simulation fails
            NotImplementedError: If using unsupported problem types
        """
        from gsim.palace.mesh import MeshConfig, generate_mesh
        from gsim.palace.ports import extract_ports

        # Validate configuration
        validation = self.validate()
        if not validation.valid:
            raise ValueError(
                f"Invalid configuration:\n" + "\n".join(validation.errors)
            )

        # Check for unsupported problem types
        if self._problem_type not in ("driven",):
            raise NotImplementedError(
                f"Problem type '{self._problem_type}' is not yet fully implemented. "
                f"Currently only 'driven' simulations are supported."
            )

        # Check for unsupported port types
        if self._waveport_configs:
            raise NotImplementedError(
                "Wave ports are not yet fully implemented."
            )
        if self._terminal_configs:
            raise NotImplementedError(
                "Terminals are not yet fully implemented."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve stack
        stack = self._resolve_stack()

        # Configure ports on component
        self._configure_ports_on_component(stack)

        # Extract ports
        palace_ports = extract_ports(self._component, stack)

        # Get effective driven config
        driven_config = self._get_effective_driven_config()

        # Build mesh config - use fmax from driven config if mesh doesn't override
        mesh_config = self._mesh_config or MeshConfigModel.default()
        effective_fmax = mesh_config.fmax if mesh_config.fmax != 100e9 else driven_config.fmax

        legacy_mesh_config = MeshConfig(
            refined_mesh_size=mesh_config.refined_mesh_size,
            max_mesh_size=mesh_config.max_mesh_size,
            cells_per_wavelength=mesh_config.cells_per_wavelength,
            margin=mesh_config.margin,
            air_above=mesh_config.air_above,
            fmax=effective_fmax,
            show_gui=mesh_config.show_gui,
            preview_only=mesh_config.preview_only,
        )

        # Generate mesh
        if verbose:
            print(f"Generating mesh in {output_dir}...")

        mesh_result = generate_mesh(
            component=self._component,
            stack=stack,
            ports=palace_ports,
            output_dir=output_dir,
            config=legacy_mesh_config,
            model_name=model_name,
            driven_config=driven_config,
        )

        if verbose:
            print(f"Mesh saved: {mesh_result.mesh_path}")
            if mesh_result.config_path:
                print(f"Config saved: {mesh_result.config_path}")

        # Build result
        result = SimulationResult(
            mesh_path=mesh_result.mesh_path,
            output_dir=output_dir,
            config_path=mesh_result.config_path,
            port_info=mesh_result.port_info,
        )

        # Run on cloud if requested
        if cloud:
            from gsim.gcloud import run_simulation

            if verbose:
                print("Running simulation on cloud...")

            cloud_results = run_simulation(
                output_dir,
                job_type="palace",
                verbose=verbose,
            )
            result.results = cloud_results

        return result

    def preview(self, *, show_gui: bool = True) -> None:
        """Preview the mesh without running simulation.

        Args:
            show_gui: Show gmsh GUI for interactive preview
        """
        from gsim.palace.mesh import MeshConfig, generate_mesh
        from gsim.palace.ports import extract_ports
        import tempfile

        # Validate configuration
        validation = self.validate()
        if not validation.valid:
            raise ValueError(
                f"Invalid configuration:\n" + "\n".join(validation.errors)
            )

        # Resolve stack
        stack = self._resolve_stack()

        # Configure ports on component
        self._configure_ports_on_component(stack)

        # Extract ports
        palace_ports = extract_ports(self._component, stack)

        # Build mesh config with preview mode
        mesh_config = self._mesh_config or MeshConfigModel.default()
        legacy_mesh_config = MeshConfig(
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
                component=self._component,
                stack=stack,
                ports=palace_ports,
                output_dir=tmpdir,
                config=legacy_mesh_config,
            )

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def show_stack(self) -> None:
        """Print the layer stack table."""
        from gsim.palace.stack import print_stack_table

        if self._stack is None:
            # Resolve stack first
            self._resolve_stack()

        if self._stack is not None:
            # Need to print using legacy function
            from gsim.palace.stack import get_stack

            yaml_path = self._stack_kwargs.get("yaml_path")
            kwargs = {k: v for k, v in self._stack_kwargs.items() if k != "yaml_path"}
            stack = get_stack(yaml_path=yaml_path, **kwargs)
            print_stack_table(stack)

    def plot_stack(self) -> None:
        """Plot the layer stack visualization."""
        from gsim.palace.stack import plot_stack, get_stack

        yaml_path = self._stack_kwargs.get("yaml_path")
        kwargs = {k: v for k, v in self._stack_kwargs.items() if k != "yaml_path"}
        stack = get_stack(yaml_path=yaml_path, **kwargs)
        plot_stack(stack)
