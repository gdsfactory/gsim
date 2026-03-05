"""Base mixin for Palace simulation classes.

Provides common methods shared across all simulation types:
DrivenSim, EigenmodeSim, ElectrostaticSim.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from gsim.palace.models.results import ValidationResult

if TYPE_CHECKING:
    from gdsfactory.component import Component

    from gsim.common import Geometry, LayerStack
    from gsim.palace.models import MaterialConfig, MeshConfig, NumericalConfig


class PalaceSimMixin:
    """Mixin providing common methods for all Palace simulation classes.

    Subclasses must define these attributes (typically via Pydantic fields):
        - geometry: Geometry | None
        - stack: LayerStack | None
        - materials: dict[str, MaterialConfig]
        - numerical: NumericalConfig
        - _output_dir: Path | None (private)
        - _stack_kwargs: dict[str, Any] (private)
    """

    # Type hints for required attributes (implemented by subclasses)
    geometry: Geometry | None
    stack: LayerStack | None
    materials: dict[str, MaterialConfig]
    numerical: NumericalConfig
    _output_dir: Path | None
    _stack_kwargs: dict[str, Any]

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
        from gsim.common import Geometry

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
        stack: LayerStack | None = None,
        *,
        yaml_path: str | Path | None = None,
        air_above: float = 200.0,
        substrate_thickness: float = 2.0,
        include_substrate: bool = False,
        **kwargs,
    ) -> None:
        """Configure the layer stack.

        Three modes of use:

        1. **Active PDK** (default — auto-detects IHP, QPDK, etc.)::

               sim.set_stack(air_above=300.0, substrate_thickness=2.0)

        2. **YAML file**::

               sim.set_stack(yaml_path="custom_stack.yaml")

        3. **Custom stack** (advanced — pass a hand-built LayerStack)::

               sim.set_stack(my_layer_stack)

        Args:
            stack: Custom gsim LayerStack (bypasses PDK extraction).
            yaml_path: Path to custom YAML stack file.
            air_above: Air box height above top metal in um.
            substrate_thickness: Thickness below z=0 in um.
            include_substrate: Include lossy silicon substrate.
            **kwargs: Additional args passed to extract_layer_stack.

        Example:
            >>> sim.set_stack(air_above=300.0, substrate_thickness=2.0)
        """
        if stack is not None:
            # Directly use a pre-built LayerStack — skip lazy resolution
            self.stack = stack
            self._stack_kwargs = {"_prebuilt": True}
            return

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
    # Material methods
    # -------------------------------------------------------------------------

    def set_material(
        self,
        name: str,
        *,
        material_type: Literal["conductor", "dielectric", "semiconductor"]
        | None = None,
        conductivity: float | None = None,
        permittivity: float | None = None,
        loss_tangent: float | None = None,
    ) -> None:
        """Override or add material properties.

        Args:
            name: Material name
            material_type: Material type (conductor, dielectric, semiconductor)
            conductivity: Conductivity in S/m (for conductors)
            permittivity: Relative permittivity (for dielectrics)
            loss_tangent: Dielectric loss tangent

        Example:
            >>> sim.set_material(
            ...     "aluminum", material_type="conductor", conductivity=3.8e7
            ... )
            >>> sim.set_material("sio2", material_type="dielectric", permittivity=3.9)
        """
        from gsim.palace.models import MaterialConfig

        # Determine type if not provided
        resolved_type = material_type
        if resolved_type is None:
            if conductivity is not None and conductivity > 1e4:
                resolved_type = "conductor"
            elif permittivity is not None:
                resolved_type = "dielectric"
            else:
                resolved_type = "dielectric"

        self.materials[name] = MaterialConfig(
            type=resolved_type,
            conductivity=conductivity,
            permittivity=permittivity,
            loss_tangent=loss_tangent,
        )

    def set_numerical(
        self,
        *,
        order: int = 1,
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
        from gsim.palace.models import NumericalConfig

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
        """Resolve the layer stack from PDK, YAML, or custom object.

        Returns:
            LayerStack object for mesh generation
        """
        # If a custom stack was given via set_stack(layer_stack), use it
        if self.stack is not None and self._stack_kwargs.get("_prebuilt"):
            # Apply material overrides
            for name, props in self.materials.items():
                self.stack.materials[name] = props.to_dict()
            return self.stack

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

    def _build_mesh_config(
        self,
        preset: Literal["coarse", "default", "graded", "fine"] | None,
        refined_mesh_size: float | None,
        max_mesh_size: float | None,
        margin: float | None,
        airbox_margin: float | None,
        fmax: float | None,
        planar_conductors: bool | None,
        show_gui: bool,
    ) -> MeshConfig:
        """Build mesh config from preset with optional overrides."""
        from gsim.palace.models import MeshConfig

        # Build mesh config from preset
        if preset == "coarse":
            mesh_config = MeshConfig.coarse()
        elif preset == "graded":
            mesh_config = MeshConfig.graded()
        elif preset == "fine":
            mesh_config = MeshConfig.fine()
        else:
            mesh_config = MeshConfig.default()

        # Preserve planar_conductors from sim.mesh_config if not
        # explicitly provided via sim.mesh(planar_conductors=...)
        if planar_conductors is None:
            existing_config = getattr(self, "mesh_config", None)
            if existing_config is not None:
                mesh_config.planar_conductors = existing_config.planar_conductors
        else:
            mesh_config.planar_conductors = planar_conductors

        # Apply overrides
        if refined_mesh_size is not None:
            mesh_config.refined_mesh_size = refined_mesh_size
        if max_mesh_size is not None:
            mesh_config.max_mesh_size = max_mesh_size
        if margin is not None:
            mesh_config.margin = margin
        if airbox_margin is not None:
            mesh_config.airbox_margin = airbox_margin
        if fmax is not None:
            mesh_config.fmax = fmax
        mesh_config.show_gui = show_gui

        return mesh_config

    # -------------------------------------------------------------------------
    # Post-mesh validation
    # -------------------------------------------------------------------------

    def validate_mesh(self) -> ValidationResult:
        """Validate the generated mesh and config before cloud submission.

        Checks that physical groups are correctly assigned after meshing:
        conductor surfaces, dielectric volumes, ports, and absorbing boundary.
        Also verifies the generated config.json structure.

        Call after mesh() and before run().

        Returns:
            ValidationResult with validation status and messages

        Example:
            >>> sim.mesh(preset="coarse")
            >>> result = sim.validate_mesh()
            >>> print(result)
        """
        errors = []
        warnings_list = []

        mesh_result = getattr(self, "_mesh_result", None) or getattr(
            self, "_last_mesh_result", None
        )
        if mesh_result is None:
            errors.append("No mesh generated. Call mesh() first.")
            return ValidationResult(valid=False, errors=errors, warnings=warnings_list)

        groups = mesh_result.groups

        # Check dielectric volumes
        if not groups.get("volumes"):
            errors.append("No dielectric volumes in mesh.")
        else:
            vol_names = list(groups["volumes"].keys())
            warnings_list.append(f"Volumes: {vol_names}")

        # Check conductor surfaces (volumetric or PEC)
        has_conductors = bool(groups.get("conductor_surfaces"))
        has_pec = bool(groups.get("pec_surfaces"))
        if not has_conductors and not has_pec:
            errors.append(
                "No conductor surfaces in mesh. "
                "Check that conductor layers have polygons and correct layer_type."
            )
        else:
            if has_conductors:
                warnings_list.append(
                    f"Conductor surfaces: {list(groups['conductor_surfaces'].keys())}"
                )
            if has_pec:
                warnings_list.append(
                    f"PEC surfaces: {list(groups['pec_surfaces'].keys())}"
                )

        # Check ports
        port_surfaces = groups.get("port_surfaces", {})
        if not port_surfaces:
            errors.append("No port surfaces in mesh.")
        else:
            for port_name, port_info in port_surfaces.items():
                if port_info.get("type") == "cpw":
                    n_elems = len(port_info.get("elements", []))
                    if n_elems < 2:
                        errors.append(
                            f"CPW port '{port_name}' has {n_elems} elements "
                            f"(expected >= 2)."
                        )

        # Check absorbing boundary
        if not groups.get("boundary_surfaces", {}).get("absorbing"):
            warnings_list.append(
                "No absorbing boundary found. This is expected if airbox_margin=0."
            )

        # Validate config.json if it exists
        output_dir = getattr(self, "_output_dir", None)
        if output_dir is not None:
            import json

            config_path = output_dir / "config.json"
            if config_path.exists():
                try:
                    config = json.loads(config_path.read_text())
                    boundaries = config.get("Boundaries", {})
                    if not boundaries.get("Conductivity") and not boundaries.get("PEC"):
                        errors.append(
                            "config.json has no Conductivity or PEC boundaries."
                        )
                    if not boundaries.get("LumpedPort") and not boundaries.get(
                        "WavePort"
                    ):
                        errors.append(
                            "config.json has no LumpedPort or WavePort entries."
                        )
                except json.JSONDecodeError as e:
                    errors.append(f"config.json is invalid JSON: {e}")

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

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
    # Visualization
    # -------------------------------------------------------------------------

    def plot_mesh(
        self,
        output: str | Path | None = None,
        show_groups: list[str] | None = None,
        interactive: bool = True,
        style: Literal["wireframe", "solid"] = "wireframe",
        transparent_groups: list[str] | None = None,
    ) -> None:
        """Plot the mesh using PyVista.

        Requires mesh() to be called first.

        Args:
            output: Output PNG path (only used if interactive=False)
            show_groups: List of group name patterns to show (None = all).
                Example: ["metal", "P"] to show metal layers and ports.
            interactive: If True, open interactive 3D viewer.
                If False, save static PNG to output path.
            style: ``"wireframe"`` (edges only) or ``"solid"`` (coloured
                surfaces per physical group).
            transparent_groups: Group names rendered at low opacity in
                *solid* mode.  Ignored in *wireframe* mode.

        Raises:
            ValueError: If output_dir not set or mesh file doesn't exist

        Example:
            >>> sim.mesh(preset="default")
            >>> sim.plot_mesh(show_groups=["metal", "P"])
            >>> sim.plot_mesh(style="solid", transparent_groups=["Absorbing_boundary"])
        """
        from gsim.viz import plot_mesh as _plot_mesh

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        mesh_path = self._output_dir / "palace.msh"
        if not mesh_path.exists():
            raise ValueError(f"Mesh file not found: {mesh_path}. Call mesh() first.")

        # Default output path if not interactive
        if output is None and not interactive:
            output = self._output_dir / "mesh.png"

        _plot_mesh(
            msh_path=mesh_path,
            output=output,
            show_groups=show_groups,
            interactive=interactive,
            style=style,
            transparent_groups=transparent_groups,
        )
