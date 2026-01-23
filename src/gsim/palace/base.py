"""Base simulation class for Palace EM simulations.

This module provides the SimBase class with shared functionality
for all simulation types (driven, eigenmode, electrostatic).
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.palace.models import (
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    SimulationResult,
    StackConfig,
    ValidationResult,
)

if TYPE_CHECKING:
    from gdsfactory.component import Component

    from gsim.palace.stack.extractor import LayerStack


class SimBase(BaseModel):
    """Base class for Palace EM simulations.

    This class provides shared functionality for all simulation types:
    - Geometry (component) management
    - Layer stack configuration
    - Material overrides
    - Mesh generation
    - Preview functionality

    Subclasses add problem-specific functionality (ports, frequencies, etc.).

    Example:
        >>> # Subclasses are used directly:
        >>> from gsim.palace import DrivenSim
        >>> sim = DrivenSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack(air_above=300.0)
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Serializable configuration
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Non-serializable state (stored as private attributes)
    _component: Component | None = PrivateAttr(default=None)
    _stack: StackConfig | None = PrivateAttr(default=None)
    _output_dir: Path | None = PrivateAttr(default=None)

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
        self._component = component

    @property
    def component(self) -> Component | None:
        """Get the current component."""
        return self._component

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
        self._stack = None

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
    # Internal helpers
    # -------------------------------------------------------------------------

    def _resolve_stack(self) -> LayerStack:
        """Resolve the layer stack from PDK or YAML.

        Returns:
            Legacy LayerStack object for mesh generation
        """
        from gsim.palace.stack import get_stack

        yaml_path = self._stack_kwargs.pop("yaml_path", None)
        stack = get_stack(yaml_path=yaml_path, **self._stack_kwargs)

        # Restore yaml_path for potential re-resolution
        self._stack_kwargs["yaml_path"] = yaml_path

        # Apply material overrides
        for name, props in self.materials.items():
            stack.materials[name] = props.to_dict()

        # Convert to Pydantic model for storage
        self._stack = StackConfig.from_legacy(stack)

        # Return legacy stack for mesh generation
        return stack

    def _validate_base(self) -> ValidationResult:
        """Validate base configuration common to all simulation types.

        Returns:
            ValidationResult with validation status and messages
        """
        errors = []
        warnings_list = []

        # Check component
        if self._component is None:
            errors.append("No component set. Call set_geometry(component) first.")

        # Check stack
        if self._stack is None and not self._stack_kwargs:
            warnings_list.append(
                "No stack configured. Will use active PDK with defaults."
            )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Mesh generation
    # -------------------------------------------------------------------------

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
    ) -> SimulationResult:
        """Internal mesh generation (called by subclasses)."""
        from gsim.palace.mesh import MeshConfig as LegacyMeshConfig
        from gsim.palace.mesh import generate_mesh

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
            print(f"Generating mesh in {output_dir}...")

        mesh_result = generate_mesh(
            component=self._component,
            stack=stack,
            ports=ports,
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

        return result

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

        # Validate configuration
        validation = self._validate_base()
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

        # Get ports (subclasses may override _get_ports)
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
                component=self._component,
                stack=stack,
                ports=ports,
                output_dir=tmpdir,
                config=legacy_mesh_config,
            )

    def _get_ports_for_preview(self, stack: LayerStack) -> list:
        """Get ports for preview. Override in subclasses."""
        return []

    # -------------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------------

    def show_stack(self) -> None:
        """Print the layer stack table.

        Example:
            >>> sim.show_stack()
        """
        from gsim.palace.stack import get_stack, print_stack_table

        if self._stack is None:
            # Resolve stack first
            self._resolve_stack()

        if self._stack is not None:
            yaml_path = self._stack_kwargs.get("yaml_path")
            kwargs = {k: v for k, v in self._stack_kwargs.items() if k != "yaml_path"}
            stack = get_stack(yaml_path=yaml_path, **kwargs)
            print_stack_table(stack)

    def plot_stack(self) -> None:
        """Plot the layer stack visualization.

        Example:
            >>> sim.plot_stack()
        """
        from gsim.palace.stack import get_stack, plot_stack

        yaml_path = self._stack_kwargs.get("yaml_path")
        kwargs = {k: v for k, v in self._stack_kwargs.items() if k != "yaml_path"}
        stack = get_stack(yaml_path=yaml_path, **kwargs)
        plot_stack(stack)


__all__ = ["SimBase"]
