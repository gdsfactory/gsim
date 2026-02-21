"""Driven simulation class for frequency-domain S-parameter extraction.

This module provides the DrivenSim class for running frequency-sweep
simulations to extract S-parameters.
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
    DrivenConfig,
    MaterialConfig,
    MeshConfig,
    NumericalConfig,
    PortConfig,
    SimulationResult,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class DrivenSim(PalaceSimMixin, BaseModel):
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
        >>> results = sim.run()

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

    # Cloud job state (set by upload/run)
    _job_id: str | None = PrivateAttr(default=None)

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
        has_ports = bool(self.ports) or bool(self.cpw_ports)
        if not has_ports:
            warnings_list.append(
                "No ports configured. Call add_port() or add_cpw_port()."
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
                f"CPW port ({cpw.upper}, {cpw.lower}): 'layer' is required"
                for cpw in self.cpw_ports
                if not cpw.layer
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
            planar_conductors=mesh_config.planar_conductors,
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
        planar_conductors: bool | None = None,
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
            air_above=air_above,
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
            air_above=mesh_config.air_above,
            fmax=mesh_config.fmax,
            show_gui=show_gui,
            preview_only=True,
            planar_conductors=mesh_config.planar_conductors,
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
            air_above=air_above,
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
        verbose: bool = True,
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
        verbose: bool = True,
        wait: bool = True,
    ) -> dict[str, Path] | str:
        """Run simulation on GDSFactory+ cloud.

        Requires mesh() to be called first. Automatically calls
        write_config() if config.json hasn't been written yet.

        Args:
            parent_dir: Where to create the sim directory.
                Defaults to the current working directory.
            verbose: Print progress messages.
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
        self.start(verbose=verbose)
        if not wait:
            return self._job_id  # type: ignore[return-value]  # set by upload()
        return self.wait_for_results(verbose=verbose, parent_dir=parent_dir)

    def run_local(
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
