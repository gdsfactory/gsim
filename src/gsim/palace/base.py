"""Base mixin for Palace simulation classes.

Provides common methods shared across all simulation types:
DrivenSim, EigenmodeSim, ElectrostaticSim.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from gsim.palace.models import (
    CPWPortConfig,
    DrivenConfig,
    EigenmodeConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PortConfig,
    TerminalConfig,
    WavePortConfig,
)
from gsim.palace.models.results import SimulationResult, ValidationResult

if TYPE_CHECKING:
    from gdsfactory.component import Component

    from gsim.common import Geometry, LayerStack

logger = logging.getLogger(__name__)


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
    driven: DrivenConfig
    eigenmode: EigenmodeConfig
    ports: list[PortConfig]
    cpw_ports: list[CPWPortConfig]
    wave_ports: list[WavePortConfig]
    terminals: list[TerminalConfig]
    simulation_type: Literal["driven", "eigenmode", "electrostatic"]
    _output_dir: Path | None
    _stack_kwargs: dict[str, Any]
    _pec_blocks: list
    _hints: dict[str, Any]
    absorbing_boundary: bool

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
        material_type: (
            Literal["conductor", "dielectric", "semiconductor"] | None
        ) = None,
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

    def add_pec(
        self,
        *,
        gds_layer: tuple[int, int],
        from_layer: str,
        to_layer: str,
    ) -> None:
        """Add a PEC block between two stack layers.

        The PEC block is a user-drawn polygon on a GDS layer that gets
        extruded between ``from_layer.zmin`` and ``to_layer.zmax`` and
        treated as a PEC boundary. This is the standard HFSS practice for
        connecting ground planes across metal layers at port boundaries.

        Args:
            gds_layer: GDS layer tuple where the PEC polygon is drawn.
            from_layer: Stack layer name — extrusion starts at this layer's zmin.
            to_layer: Stack layer name — extrusion ends at this layer's zmax.

        Example:
            >>> sim.add_pec(
            ...     gds_layer=(65000, 0), from_layer="metal1", to_layer="topmetal2"
            ... )
        """
        from gsim.palace.models.pec import PECBlockConfig

        self._pec_blocks.append(
            PECBlockConfig(
                gds_layer=gds_layer,
                from_layer=from_layer,
                to_layer=to_layer,
            )
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
    ) -> None:
        """Configure numerical solver parameters.

        Args:
            order: Finite element order (1-4)
            tolerance: Linear solver tolerance
            max_iterations: Maximum solver iterations
            solver_type: Linear solver type
            preconditioner: Preconditioner type
            device: Compute device (CPU or GPU)

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
        margin_x: float | None = None,
        margin_y: float | None = None,
    ) -> MeshConfig:
        """Build mesh config from preset with optional overrides."""
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
        if margin_x is not None:
            mesh_config.margin_x = margin_x
        if margin_y is not None:
            mesh_config.margin_y = margin_y
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
        if not port_surfaces and self.simulation_type == "driven":
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
                    if (
                        not boundaries.get("LumpedPort")
                        and not boundaries.get("WavePort")
                    ) and (
                        self.simulation_type == "driven"
                        or self.simulation_type == "waveport"
                    ):
                        errors.append(
                            "config.json has no LumpedPort nor Waveport entries."
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

        # Check ports

        has_ports = bool(self.ports) or bool(self.cpw_ports) or bool(self.wave_ports)
        if not has_ports:
            if self.simulation_type == "driven":
                warnings_list.append(
                    "No ports configured. Call add_port(), add_cpw_port(),"
                    " or add_wave_port()."
                )
            elif self.simulation_type == "eigenmode":
                warnings_list.append(
                    "No ports configured. Eigenmode findsallmodes without port loading."
                )
        else:
            # Validate port configurations
            for port in self.ports:
                if port.geometry == "inplane" and port.layer is None:
                    errors.append(f"Port '{port.name}': inplane ports require 'layer'")
                if port.geometry == "via" and (
                    port.from_layer is None or port.to_layer is None
                ):
                    errors.append(
                        f"Port '{port.name}': via ports require "
                        "'from_layer' and 'to_layer'"
                    )

            # Validate CPW ports
            errors.extend(
                f"CPW port '{cpw.name}': 'layer' is required"
                for cpw in self.cpw_ports
                if not cpw.layer
            )
            # Validate wave ports
            errors.extend(
                f"Wave port '{wp.name}': 'layer' is required"
                for wp in self.wave_ports
                if not wp.layer
            )

        # Validate excitation port if specified
        if self.simulation_type == "driven" and self.driven.excitation_port is not None:
            port_names = [p.name for p in self.ports]
            cpw_names = [cpw.name for cpw in self.cpw_ports]
            all_port_names = port_names + cpw_names
            if self.driven.excitation_port not in all_port_names:
                errors.append(
                    f"Excitation port '{self.driven.excitation_port}' not found. "
                    f"Available: {all_port_names}"
                )

        if self.simulation_type == "electrostatic" and len(self.terminals) < 2:
            # Electrostatic requires at least 2 terminals
            errors.append(
                "Electrostatic simulation requires at least 2 terminals. "
                "Call add_terminal() to add terminals."
            )
        if self.simulation_type == "electrostatic":
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

    def _find_gf_port(self, port_name: str):
        """Find a gdsfactory port by name."""
        component = self.geometry.component if self.geometry else None
        if component is None:
            raise ValueError("No component set")

        for p in component.ports:
            if p.name == port_name:
                return p

        raise ValueError(
            f"Port '{port_name}' not found on component. "
            f"Available ports: {[p.name for p in component.ports]}"
        )

    def _configure_ports_on_component(self, stack: LayerStack) -> None:  # noqa: ARG002
        """Configure ports on the component using legacy functions."""
        from gsim.palace.ports import (
            configure_cpw_port,
            configure_inplane_port,
            configure_via_port,
            configure_wave_port,
        )

        component = self.geometry.component if self.geometry else None
        if component is None:
            raise ValueError("No component set")

        # Configure regular ports
        for port_config in self.ports:
            if self.simulation_type != "driven":
                port_config.excited = False
            if port_config.name is None:
                continue

            # Find matching gdsfactory port
            gf_port = self._find_gf_port(port_config.name)

            if gf_port is None:
                raise ValueError(
                    f"Port '{port_config.name}' not found on component. "
                    f"Available ports: {[p.name for p in component.ports]}"
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

            # Attach RLC values to port info for downstream consumers
            if port_config.resistance is not None:
                gf_port.info["resistance"] = port_config.resistance
            if port_config.inductance is not None:
                gf_port.info["inductance"] = port_config.inductance
            if port_config.capacitance is not None:
                gf_port.info["capacitance"] = port_config.capacitance

        # Configure CPW ports
        for cpw_config in self.cpw_ports:
            # Find the single gdsfactory port at the signal center
            gf_port = self._find_gf_port(cpw_config.name)

            if gf_port is None:
                raise ValueError(
                    f"CPW port '{cpw_config.name}' not found on component. "
                    f"Available: {[p.name for p in component.ports]}"
                )

            configure_cpw_port(
                gf_port,
                layer=cpw_config.layer,
                s_width=cpw_config.s_width,
                gap_width=cpw_config.gap_width,
                length=cpw_config.length,
                impedance=cpw_config.impedance,
                excited=cpw_config.excited,
                offset=cpw_config.offset,
            )
        # Configure wave ports
        for port_config in self.wave_ports:
            if port_config.name is None:
                continue

            # Find matching gdsfactory port
            gf_port = self._find_gf_port(port_config.name)

            if port_config.layer is not None:
                configure_wave_port(
                    gf_port,
                    layer=port_config.layer,
                    z_margin=port_config.z_margin,
                    lateral_margin=port_config.lateral_margin,
                    max_size=port_config.max_size,
                    excited=port_config.excited,
                    mode=port_config.mode,
                    offset=port_config.offset,
                )

        self._configured_ports = True

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
            margin_x=mesh_config.margin_x,
            margin_y=mesh_config.margin_y,
            airbox_margin=mesh_config.airbox_margin,
            fmax=effective_fmax,
            show_gui=mesh_config.show_gui,
            preview_only=mesh_config.preview_only,
            planar_conductors=mesh_config.planar_conductors,
            refine_from_curves=mesh_config.refine_from_curves,
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
            eigenmode_config=self.eigenmode,
            simulation_type=self.simulation_type,
            write_config=write_config,
            absorbing_boundary=self.absorbing_boundary,
            pec_blocks=self._pec_blocks or None,
            verbosity=3,
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
        preset: Literal["coarse", "default", "graded", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        margin_x: float | None = None,
        margin_y: float | None = None,
        airbox_margin: float | None = None,
        fmax: float | None = None,
        planar_conductors: bool | None = None,
        show_gui: bool = True,
    ) -> None:
        """Preview the mesh without running simulation.

        Opens the gmsh GUI to visualize the mesh interactively.

        Args:
            preset: Mesh quality preset ("coarse", "default", "graded", "fine")
            refined_mesh_size: Mesh size near conductors (um)
            max_mesh_size: Max mesh size in air/dielectric (um)
            margin: XY margin around design (um)
            margin_x: X-axis margin (um). Overrides margin for X.
            margin_y: Y-axis margin (um). Overrides margin for Y.
            airbox_margin: Extra airbox around stack (um); 0 = disabled
            fmax: Max frequency for mesh sizing (Hz)
            planar_conductors: Treat conductors as 2D PEC surfaces
            show_gui: Show gmsh GUI for interactive preview

        Example:
            >>> sim.preview(preset="fine", planar_conductors=True, show_gui=True)
        """
        from gsim.palace.mesh import MeshConfig as LegacyMeshConfig
        from gsim.palace.mesh import generate_mesh

        component = self.geometry.component if self.geometry else None

        # Validate configuration
        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        # Build mesh config
        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            margin_x=margin_x,
            margin_y=margin_y,
            airbox_margin=airbox_margin,
            fmax=fmax,
            planar_conductors=planar_conductors,
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
            margin_x=mesh_config.margin_x,
            margin_y=mesh_config.margin_y,
            airbox_margin=mesh_config.airbox_margin,
            fmax=mesh_config.fmax,
            show_gui=show_gui,
            preview_only=True,
            planar_conductors=mesh_config.planar_conductors,
            refine_from_curves=mesh_config.refine_from_curves,
        )

        # Generate mesh in temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_mesh(
                component=component,
                stack=stack,
                ports=ports,
                output_dir=tmpdir,
                config=legacy_mesh_config,
                driven_config=self.driven,
                eigenmode_config=self.eigenmode,
                simulation_type=self.simulation_type,
                absorbing_boundary=self.absorbing_boundary,
                pec_blocks=self._pec_blocks or None,
            )

    # -------------------------------------------------------------------------
    # Mesh generation
    # -------------------------------------------------------------------------

    def mesh(
        self,
        *,
        preset: Literal["coarse", "default", "graded", "fine"] | None = None,
        refined_mesh_size: float | None = None,
        max_mesh_size: float | None = None,
        margin: float | None = None,
        margin_x: float | None = None,
        margin_y: float | None = None,
        airbox_margin: float | None = None,
        fmax: float | None = None,
        planar_conductors: bool | None = None,
        show_gui: bool = False,
        model_name: str = "palace",
        verbose: bool = True,
    ) -> SimulationResult:
        """Generate the mesh for Palace simulation.

        Only generates the mesh file (palace.msh). Config is generated
        separately with write_config().

        Requires set_output_dir() to be called first.

        Args:
            preset: Mesh quality preset ("coarse", "default", "graded", "fine")
            refined_mesh_size: Mesh size near conductors (um), overrides preset
            max_mesh_size: Max mesh size in air/dielectric (um), overrides preset
            margin: XY margin around design (um), overrides preset
            margin_x: X-axis margin (um). Overrides margin for X.
            margin_y: Y-axis margin (um). Overrides margin for Y.
            airbox_margin: Extra airbox around stack (um); 0 = disabled
            fmax: Max frequency for mesh sizing (Hz), overrides preset
            planar_conductors: Treat conductors as 2D PEC surfaces
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

        # Build mesh config
        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            margin_x=margin_x,
            margin_y=margin_y,
            airbox_margin=airbox_margin,
            fmax=fmax,
            planar_conductors=planar_conductors,
            show_gui=show_gui,
        )

        # Validate configuration
        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

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

        Returns:
            Path to the generated config.json

        Raises:
            ValueError: If mesh() hasn't been called yet

        Example:
            >>> result = sim.mesh("./sim")
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
            simulation_type=self.simulation_type,
            eigenmode_config=self.eigenmode,
            driven_config=self.driven,
            absorbing_boundary=self.absorbing_boundary,
            hints=self._hints,
        )

        # Validate mesh and config
        validation = self.validate_mesh()
        if not validation.valid:
            raise ValueError(f"Mesh validation failed:\n{validation}")

        return config_path

    # -------------------------------------------------------------------------
    # Cloud: fine-grained control
    # -------------------------------------------------------------------------

    def _prepare_upload_dir(self) -> Path:
        """Prepare a temp directory with all config/mesh files for upload.

        Ensures ``_output_dir`` is set, ``config.json`` exists, and copies
        everything to a fresh temp directory.

        Returns:
            Path to temp directory ready for upload.
        """
        import shutil

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        # Always (re)generate config.json to reflect current driven settings
        self.write_config()

        # Copy input files to a temp dir so we don't destroy the user's directory
        tmp = Path(tempfile.mkdtemp(prefix="palace_"))
        for item in self._output_dir.iterdir():
            dest = tmp / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        return tmp

    def upload(self, *, verbose: bool = True) -> str:
        """Prepare config, upload to the cloud. Does NOT start execution.

        Requires :meth:`set_output_dir` and :meth:`mesh` to have been
        called first.

        Args:
            verbose: Print progress messages.

        Returns:
            ``job_id`` string for use with :meth:`start`, :meth:`get_status`,
            or :func:`gsim.wait_for_results`.
        """
        from gsim import gcloud

        tmp = self._prepare_upload_dir()
        try:
            self._job_id = gcloud.upload(tmp, "palace", verbose=verbose)
        except Exception:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return self._job_id

    def start(self, *, verbose: bool = True) -> None:
        """Start cloud execution for this sim's uploaded job.

        Raises:
            ValueError: If :meth:`upload` has not been called.
        """
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("Call upload() first")
        gcloud.start(self._job_id, verbose=verbose)

    def get_status(self) -> str:
        """Get the current status of this sim's cloud job.

        Returns:
            Status string (``"created"``, ``"queued"``, ``"running"``,
            ``"completed"``, ``"failed"``).

        Raises:
            ValueError: If no job has been submitted yet.
        """
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("No job submitted yet")
        return gcloud.get_status(self._job_id)

    def wait_for_results(
        self,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        parent_dir: str | Path | None = None,
    ) -> Any:
        """Wait for this sim's cloud job, download and parse results.

        Args:
            verbose: Print progress messages.
            parent_dir: Where to create the sim-data directory.

        Returns:
            Parsed result (typically ``dict[str, Path]`` of output files).

        Raises:
            ValueError: If no job has been submitted yet.
        """
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("No job submitted yet")
        return gcloud.wait_for_results(
            self._job_id, verbose=verbose, parent_dir=parent_dir
        )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def run(
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        wait: bool = True,
    ) -> dict[str, Path] | str:
        """Run simulation on GDSFactory+ cloud.

        Requires mesh() to be called first. Automatically calls
        write_config() if config.json hasn't been written yet.

        Args:
            parent_dir: Where to create the sim directory.
                Defaults to the current working directory.
            verbose: ``"quiet"`` no output, ``"status"`` status line,
                ``"full"`` stream solver logs.
            wait: If ``True`` (default), block until results are ready.
                If ``False``, upload + start and return the ``job_id``.

        Returns:
            ``dict[str, Path]`` of output files when ``wait=True``,
            or ``job_id`` string when ``wait=False``.

        Raises:
            ValueError: If output_dir not set or mesh not generated
            RuntimeError: If simulation fails

        Example:
            >>> results = sim.run()
            >>> print(f"S-params saved to: {results['port-S.csv']}")
        """
        self.upload(verbose=False)
        self.start(verbose=verbose != "quiet")
        if not wait:
            if self._job_id is None:
                msg = "job_id not set — call upload() first"
                raise RuntimeError(msg)
            return self._job_id
        return self.wait_for_results(verbose=verbose, parent_dir=parent_dir)

    def run_local(
        self,
        *,
        palace_sif_path: str | Path | None = None,
        palace_executable: str | Path | None = None,
        use_apptainer: bool = True,
        num_processes: int = 1,
        num_threads: int | None = None,
        verbose: bool = True,
    ) -> dict[str, Path]:
        """Run simulation locally using Palace.

        Requires mesh() and write_config() to be called first.
        Supports both Apptainer and direct Palace installation.

        Args:
            palace_sif_path: Path to Palace Apptainer SIF file.
                Only used when ``use_apptainer=True``.
                If None, uses PALACE_SIF environment variable.
            palace_executable: Path to Palace executable.
                Only used when ``use_apptainer=False``.
                If None, uses PALACE_EXECUTABLE environment variable or "palace".
            use_apptainer: If True (default), run via Apptainer using SIF file.
                If False, run Palace executable directly.
            num_processes: Number of MPI processes (default: 1)
            num_threads: Number of OpenMP threads to use for OpenMP builds, default is 1
                or the value of OMP_NUM_THREADS in the environment
            verbose: Print progress messages

        Returns:
            Dict mapping result filenames to local paths

        Raises:
            ValueError: If output_dir not set or Palace not configured
            FileNotFoundError: If mesh, config, or Palace not found
            RuntimeError: If simulation fails

        Example:
            >>> # Using Apptainer (default)
            >>> import os
            >>> os.environ["PALACE_SIF"] = "/path/to/Palace.sif"
            >>> results = sim.run_local()
            >>>
            >>> # Using Apptainer with explicit path
            >>> results = sim.run_local(palace_sif_path="/path/to/Palace.sif")
            >>>
            >>> # Using direct Palace installation
            >>> results = sim.run_local(use_apptainer=False)
            >>>
            >>> # Using direct Palace with custom executable path
            >>> results = sim.run_local(
            ...     use_apptainer=False, palace_executable="/usr/local/bin/palace"
            ... )
            >>> print(f"S-params: {results['port-S.csv']}")
        """
        import os
        import shutil
        import subprocess

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        output_dir = Path(self._output_dir)
        config_path = output_dir / "config.json"
        mesh_path = output_dir / "palace.msh"

        # Check required files exist
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. Call write_config() first."
            )

        if not mesh_path.exists():
            raise FileNotFoundError(
                f"Mesh file not found: {mesh_path}. Call mesh() first."
            )

        # Determine Palace command based on use_apptainer flag
        if use_apptainer:
            # Determine Palace SIF path from environment variable or parameter
            if palace_sif_path is None:
                palace_sif_path = os.environ.get("PALACE_SIF")
                if palace_sif_path is None:
                    raise ValueError(
                        "Palace SIF path not specified. Either set PALACE_SIF "
                        "environment variable or pass palace_sif_path parameter."
                    )
                if verbose:
                    logger.info(
                        "Using PALACE_SIF from environment: %s", palace_sif_path
                    )

            sif_path = Path(palace_sif_path).expanduser().resolve()

            if not sif_path.exists():
                raise FileNotFoundError(
                    f"Palace SIF file not found: {sif_path}. "
                    "Install Palace via Apptainer or provide correct path."
                )

            # Check that apptainer is available
            if shutil.which("apptainer") is None:
                raise RuntimeError(
                    "Apptainer not found. Install Apptainer to run local simulations "
                    "with use_apptainer=True."
                )

            # Build Apptainer command
            cmd = [
                "apptainer",
                "run",
                str(sif_path),
                "-np",
                str(num_processes),
            ]

        else:
            # Direct Palace execution
            if palace_executable is None:
                palace_executable = os.environ.get("PALACE_EXECUTABLE", "palace")
                if verbose:
                    logger.info(
                        "Using Palace executable from environment/default: %s",
                        palace_executable,
                    )

            exe_path = Path(palace_executable).expanduser()

            # Check if executable exists
            if not exe_path.exists():
                # Try resolving to see if it's in PATH
                resolved = shutil.which(str(exe_path))
                if resolved is None:
                    raise FileNotFoundError(
                        f"Palace executable not found: {exe_path}. "
                        "Install Palace directly or provide correct path via "
                        "palace_executable parameter."
                    )
                exe_path = Path(resolved)

            cmd = [
                str(exe_path),
                "-np",
                str(num_processes),
            ]

        if num_threads is not None:
            cmd.extend(["-nt", str(num_threads)])
        cmd.extend(["config.json"])

        if verbose:
            if use_apptainer:
                logger.info("Running Palace simulation in %s via Apptainer", output_dir)
            else:
                logger.info("Running Palace simulation in %s directly", output_dir)
            logger.info("Command: %s", " ".join(cmd))
            logger.info("Processes: %d", num_processes)

        # Run simulation
        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                cwd=output_dir,
                check=True,
                capture_output=True,
                text=True,
            )

            # Log output if verbose
            if verbose and result.stdout:
                logger.info(result.stdout)
            if verbose and result.stderr:
                logger.warning(result.stderr)

        except subprocess.CalledProcessError as e:
            error_msg = f"Palace simulation failed with return code {e.returncode}"
            if e.stdout:
                error_msg += f"\n\nStdout:\n{e.stdout}"
            if e.stderr:
                error_msg += f"\n\nStderr:\n{e.stderr}"
            raise RuntimeError(error_msg) from e
        except FileNotFoundError as e:
            if use_apptainer:
                raise RuntimeError(
                    "Apptainer not found. Install Apptainer to run local simulations "
                    "with use_apptainer=True."
                ) from e
            raise RuntimeError(
                "Palace executable not found. Install Palace directly or provide "
                "correct path via palace_executable parameter, "
                "or set PALACE_EXECUTABLE environment variable."
            ) from e

        if verbose:
            logger.info("Simulation completed successfully")

        postpro_dir = output_dir / "output/palace/"

        if verbose:
            logger.info("Results saved to %s", postpro_dir)

        return {
            file.name: file
            for file in postpro_dir.iterdir()
            if file.is_file() and not file.name.startswith(".")
        }

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
        resistance: float | None = None,
        inductance: float | None = None,
        capacitance: float | None = None,
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
            resistance: Series resistance (Ohms)
            inductance: Series inductance (H)
            capacitance: Shunt capacitance (F)
            excited: Whether this port is excited
            geometry: Port geometry type ("inplane" or "via")

        Example:
            >>> sim.add_port("o1", layer="topmetal2", length=5.0)
            >>> sim.add_port(
            ...     "feed", from_layer="metal1", to_layer="topmetal2", geometry="via"
            ... )
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
                resistance=resistance,
                inductance=inductance,
                capacitance=capacitance,
                excited=excited,
                geometry=geometry,
            )
        )

    def add_cpw_port(
        self,
        name: str,
        *,
        layer: str,
        s_width: float,
        gap_width: float,
        length: float = 0.1,
        offset: float = 0.0,
        impedance: float = 50.0,
        excited: bool = True,
    ) -> None:
        """Add a coplanar waveguide (CPW) port.

        CPW ports consist of two elements (upper and lower gaps) that are
        excited with opposite E-field directions to create the CPW mode.

        Place a single gdsfactory port at the center of the signal conductor.
        The two gap element surfaces are computed from s_width and gap_width.

        Args:
            name: Port name (must match a component port at the signal center)
            layer: Target conductor layer (e.g., "topmetal2")
            s_width: Width of the signal (center) conductor (um)
            gap_width: Width of each gap between signal and ground (um)
            length: Port extent along direction (um)
            offset: Shift the port inward along the waveguide (um).
                Positive moves away from the boundary, into the conductor.
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited

        Example:
            >>> sim.add_cpw_port(
            ...     "left", layer="topmetal2", s_width=20, gap_width=15, length=5.0
            ... )
        """
        # Remove existing CPW port with same name if any
        self.cpw_ports = [p for p in self.cpw_ports if p.name != name]

        self.cpw_ports.append(
            CPWPortConfig(
                name=name,
                layer=layer,
                s_width=s_width,
                gap_width=gap_width,
                length=length,
                offset=offset,
                impedance=impedance,
                excited=excited,
            )
        )

    def add_wave_port(
        self,
        name: str,
        *,
        layer: str | None = None,
        z_margin: float = 0.0,
        lateral_margin: float = 0.0,
        max_size: bool = False,
        mode: int = 1,
        excited: bool = True,
        offset: float = 0.0,
    ) -> None:
        """Add a single element wave port.

        Args:
            name: Port name (must match a component port at the signal center)
            layer: Target conductor layer (e.g., "topmetal2")
            z_margin: Margin in z direction
            lateral_margin: Margin in x/y directions
              Ignores lateral margin and port_width.
            max_size: When True, automatically set z_margin and lateral_margin
                to fill the full simulation domain boundary on that side.
                Overrides z_margin and lateral_margin values.
            mode: Mode number to excite.
            excited: Whether this port is excited
            offset: Offset distance used for scattering parameter de-embedding.

        Example:
            >>> sim.add_wave_port(
            ...     "w1",
            ...     layer="topmetal2",
            ...     max_size=True,
            ...     mode=1,
            ...     excited=True,
            ...     offset=0.0,
            ... )
        """
        self.wave_ports = [p for p in self.wave_ports if p.name != name]

        self.wave_ports.append(
            WavePortConfig(
                name=name,
                layer=layer,
                z_margin=z_margin,
                lateral_margin=lateral_margin,
                max_size=max_size,
                mode=mode,
                excited=excited,
                offset=offset,
            )
        )
