"""Modal simulation class for 2-D cross-section mode analysis.

This module provides the ModalSim class for generating 2-D cross-section
meshes from 3-D Palace geometry.  The resulting mesh can be used with
external mode solvers such as *femwell*.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack
from gsim.palace.base import PalaceSimMixin
from gsim.palace.mesh.pipeline import CrossSection
from gsim.palace.models import (
    CPWPortConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PortConfig,
    SimulationResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class ModalSim(PalaceSimMixin, BaseModel):
    """2-D cross-section simulation for electromagnetic mode analysis.

    This class builds a 3-D geometry from a gdsfactory component and PDK
    layer stack, slices it with a cutting plane, and produces a triangulated
    2-D mesh suitable for eigenmode solvers such as *femwell*.

    Example:
        >>> from gsim.palace import ModalSim
        >>> from gsim.palace.mesh import CrossSection
        >>>
        >>> sim = ModalSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        >>> sim.set_cross_section(y=0.0)
        >>> sim.set_output_dir("./sim-modal")
        >>> sim.mesh(preset="default")

    Attributes:
        geometry: Wrapped gdsfactory Component (from common)
        stack: Layer stack configuration (from common)
        cross_section: Cutting-plane definition (required before meshing)
        ports: Optional port configurations (carried through for 3-D build)
        cpw_ports: Optional CPW port configurations
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

    # Cross-section definition
    cross_section: CrossSection | None = None

    # Port configurations (needed for 3-D geometry build before slicing)
    ports: list[PortConfig] = Field(default_factory=list)
    cpw_ports: list[CPWPortConfig] = Field(default_factory=list)

    # Material overrides and numerical config
    materials: dict[str, MaterialConfig] = Field(default_factory=dict)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)

    # Mesh config
    mesh_config: MeshConfig = Field(default_factory=MeshConfig.default)

    # Stack configuration (stored as kwargs until resolved)
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Internal state
    _output_dir: Path | None = PrivateAttr(default=None)
    _configured_ports: bool = PrivateAttr(default=False)
    _last_mesh_result: Any = PrivateAttr(default=None)
    _last_ports: list = PrivateAttr(default_factory=list)

    # -------------------------------------------------------------------------
    # Cross-section configuration
    # -------------------------------------------------------------------------

    def set_cross_section(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        """Define the cutting plane for the 2-D cross-section.

        Exactly one of *x* or *y* must be given.

        Args:
            x: Fix x to this value → slice in the YZ-plane.
            y: Fix y to this value → slice in the XZ-plane.

        Example:
            >>> sim.set_cross_section(y=0.0)   # XZ-plane at y = 0
            >>> sim.set_cross_section(x=50.0)  # YZ-plane at x = 50 µm
        """
        self.cross_section = CrossSection(x=x, y=y)

    # -------------------------------------------------------------------------
    # Port helpers (kept for geometry build)
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

        Ports are used during the 3-D geometry construction (before
        slicing).  They do not appear in the final 2-D mesh.

        Args:
            name: Port name (must match component port name)
            layer: Target layer for inplane ports
            from_layer: Bottom layer for via ports
            to_layer: Top layer for via ports
            length: Port extent along direction (um)
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited
            geometry: Port geometry type ("inplane" or "via")
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

        Ports are used during the 3-D geometry construction (before
        slicing).  They do not appear in the final 2-D mesh.

        Args:
            upper: Name of the upper gap port on the component
            lower: Name of the lower gap port on the component
            layer: Target conductor layer
            length: Port extent along direction (um)
            impedance: Port impedance (Ohms)
            excited: Whether this port is excited
            name: Optional name for the CPW port
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
    # Validation
    # -------------------------------------------------------------------------

    def validate_config(self) -> ValidationResult:
        """Validate the simulation configuration.

        Returns:
            ValidationResult with validation status and messages
        """
        errors: list[str] = []
        warnings_list: list[str] = []

        if self.geometry is None:
            errors.append("No component set. Call set_geometry(component) first.")

        if self.stack is None and not self._stack_kwargs:
            warnings_list.append(
                "No stack configured. Will use active PDK with defaults."
            )

        if self.cross_section is None:
            errors.append(
                "No cross-section defined. Call set_cross_section(x=...) or "
                "set_cross_section(y=...) first."
            )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _configure_ports_on_component(self, stack: LayerStack) -> None:  # noqa: ARG002
        """Configure ports on the component using legacy functions."""
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
            gf_port = next(
                (p for p in component.ports if p.name == port_config.name), None
            )
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
                port_config.from_layer is not None
                and port_config.to_layer is not None
            ):
                configure_via_port(
                    gf_port,
                    from_layer=port_config.from_layer,
                    to_layer=port_config.to_layer,
                    impedance=port_config.impedance,
                    excited=port_config.excited,
                )

        for cpw_config in self.cpw_ports:
            port_upper = next(
                (p for p in component.ports if p.name == cpw_config.upper), None
            )
            if port_upper is None:
                raise ValueError(
                    f"CPW upper port '{cpw_config.upper}' not found. "
                    f"Available: {[p.name for p in component.ports]}"
                )
            port_lower = next(
                (p for p in component.ports if p.name == cpw_config.lower), None
            )
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
        """Generate a 2-D cross-section mesh.

        The 3-D geometry is built exactly as for a driven simulation,
        then sliced by the configured cutting plane and meshed in 2-D.

        Requires ``set_output_dir()`` and ``set_cross_section()`` first.

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
            ValueError: If output_dir or cross_section not set
        """
        from gsim.palace.mesh import CrossSection as _CS
        from gsim.palace.mesh import MeshConfig as LegacyMeshConfig
        from gsim.palace.mesh import generate_mesh
        from gsim.palace.ports import extract_ports

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")
        if self.cross_section is None:
            raise ValueError(
                "No cross-section defined. Call set_cross_section() first."
            )

        component = self.geometry.component if self.geometry else None

        # Build mesh config from preset
        mesh_config = self._build_mesh_config(
            preset=preset,
            refined_mesh_size=refined_mesh_size,
            max_mesh_size=max_mesh_size,
            margin=margin,
            air_above=air_above,
            fmax=fmax,
            show_gui=show_gui,
        )

        # Validate
        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        output_dir = self._output_dir

        # Resolve stack and configure ports
        stack = self._resolve_stack()
        if self.ports or self.cpw_ports:
            self._configure_ports_on_component(stack)

        # Extract ports (may be empty for modal sim)
        palace_ports = extract_ports(component, stack) if (self.ports or self.cpw_ports) else []

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

        if verbose:
            logger.info("Generating 2-D cross-section mesh in %s", output_dir)

        # Pipeline cross-section object
        cs = _CS(x=self.cross_section.x, y=self.cross_section.y)

        mesh_result = generate_mesh(
            component=component,
            stack=stack,
            ports=palace_ports,
            output_dir=output_dir,
            config=legacy_mesh_config,
            model_name=model_name,
            write_config=False,
            cross_section=cs,
        )

        self._last_mesh_result = mesh_result
        self._last_ports = palace_ports

        return SimulationResult(
            mesh_path=mesh_result.mesh_path,
            output_dir=output_dir,
            config_path=None,
            port_info=mesh_result.port_info,
            mesh_stats=mesh_result.mesh_stats,
        )

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_mesh(
        self,
        output: str | Path | None = None,
        show_groups: list[str] | None = None,
        interactive: bool = True,
        style: str = "wireframe",
        transparent_groups: list[str] | None = None,
    ) -> None:
        """Plot the 2-D mesh using PyVista.

        Requires ``mesh()`` to be called first.

        Args:
            output: Output PNG path (only used if interactive=False)
            show_groups: List of group name patterns to show (None = all).
            interactive: If True, open interactive viewer.
                If False, save static PNG to output path.
            style: ``"wireframe"`` (default) or ``"solid"``.
                ``"solid"`` renders faces coloured by physical group
                with selectable transparent groups.
            transparent_groups: Physical-group names rendered at low opacity
                when *style="solid"*.  Defaults to
                ``["air_none", "air_plastic_enclosure"]``.

        Raises:
            ValueError: If output_dir not set or mesh file doesn't exist
        """
        from gsim.viz import plot_mesh as _plot_mesh

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        mesh_path = self._output_dir / "palace.msh"
        if not mesh_path.exists():
            raise ValueError(f"Mesh file not found: {mesh_path}. Call mesh() first.")

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


__all__ = ["ModalSim"]
