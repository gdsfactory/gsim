"""MEEP FDTD simulation class for S-parameter extraction.

Main entry point for photonic FDTD simulation using MEEP on the cloud.
No local MEEP dependency — MEEP runs only on the cloud.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from gsim.common import Geometry, LayerStack, ValidationResult
from gsim.common.stack.materials import MaterialProperties
from gsim.meep.base import MeepSimMixin
from gsim.meep.models import (
    AccuracyConfig,
    DiagnosticsConfig,
    FDTDConfig,
    DomainConfig,
    ResolutionConfig,
    SimConfig,
    SourceConfig,
    SParameterResult,
    StoppingConfig,
    SymmetryEntry,
)

logger = logging.getLogger(__name__)


class MeepSim(MeepSimMixin, BaseModel):
    """MEEP FDTD simulation for photonic S-parameter extraction.

    Uses a fluent API to collect configuration, then serializes to
    JSON + GDS + Python runner script for cloud execution.

    Example:
        >>> from gsim.meep import MeepSim
        >>>
        >>> sim = MeepSim()
        >>> sim.set_geometry(component)
        >>> sim.set_stack()
        >>> sim.set_material("si", refractive_index=3.47)
        >>> sim.set_wavelength(wavelength=1.55, bandwidth=0.1)
        >>> sim.set_resolution(pixels_per_um=32)
        >>> sim.set_output_dir("./meep-sim")
        >>> result = sim.simulate()
        >>> result.plot()
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    # Composed objects (from common)
    geometry: Geometry | None = None
    stack: LayerStack | None = None

    # MEEP-specific configs
    fdtd_config: FDTDConfig = Field(default_factory=FDTDConfig)
    source_config: SourceConfig = Field(default_factory=SourceConfig)
    stopping_config: StoppingConfig = Field(default_factory=StoppingConfig)
    resolution_config: ResolutionConfig = Field(default_factory=ResolutionConfig)
    domain_config: DomainConfig = Field(default_factory=DomainConfig)
    accuracy_config: AccuracyConfig = Field(default_factory=AccuracyConfig)
    diagnostics_config: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    verbose_interval: float = Field(default=0, ge=0)

    # Material overrides (optical properties)
    materials: dict[str, MaterialProperties] = Field(default_factory=dict)

    # Performance options
    symmetries: list[SymmetryEntry] = Field(default_factory=list)
    split_chunks_evenly: bool = False

    # Private state
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)
    _output_dir: Path | None = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Wavelength / frequency config
    # -------------------------------------------------------------------------

    def set_wavelength(
        self,
        *,
        wavelength: float = 1.55,
        bandwidth: float = 0.1,
        num_freqs: int = 11,
    ) -> None:
        """Configure wavelength range for simulation.

        Args:
            wavelength: Center wavelength in um
            bandwidth: Wavelength bandwidth in um
            num_freqs: Number of frequency points
        """
        self.fdtd_config = FDTDConfig(
            wavelength=wavelength,
            bandwidth=bandwidth,
            num_freqs=num_freqs,
        )

    # -------------------------------------------------------------------------
    # Source config
    # -------------------------------------------------------------------------

    def set_source(
        self,
        *,
        bandwidth: float | None = None,
        port: str | None = None,
    ) -> None:
        """Configure the excitation source.

        Args:
            bandwidth: Source Gaussian bandwidth in wavelength um.
                ``None`` = auto (~3x monitor bandwidth or ``0.2*fcen``).
            port: Source port name. ``None`` = auto-select first port.
        """
        self.source_config = SourceConfig(bandwidth=bandwidth, port=port)

    # -------------------------------------------------------------------------
    # Stopping config
    # -------------------------------------------------------------------------

    def set_stopping(
        self,
        *,
        mode: str = "fixed",
        max_time: float = 100.0,
        threshold: float = 1e-3,
        decay_dt: float = 50.0,
        decay_component: str = "Ey",
        decay_monitor_port: str | None = None,
        dft_min_run_time: float = 100,
    ) -> None:
        """Configure when the MEEP simulation stops.

        Args:
            mode: ``"fixed"`` (run for ``max_time``), ``"decay"``
                (field decay at a point), or ``"dft_decay"`` (DFT
                convergence — best for S-parameters).
            max_time: Time units to run after sources turn off. In
                ``fixed`` mode this is the run time; in ``decay``/
                ``dft_decay`` it is the maximum time cap.
            threshold: Stop when fields/DFT decay to this fraction.
            decay_dt: Time interval between decay checks (MEEP time
                units).  Only used in ``decay`` mode.
            decay_component: Field component to monitor (e.g. ``"Ey"``).
                Only used in ``decay`` mode.
            decay_monitor_port: Port name to monitor for decay. ``None``
                = auto-select first non-source port.  Only in ``decay``.
            dft_min_run_time: Minimum time after sources for ``dft_decay``
                mode (respects Fourier uncertainty principle).
        """
        self.stopping_config = StoppingConfig(
            mode=mode,
            run_after_sources=max_time,
            decay_dt=decay_dt,
            decay_component=decay_component,
            decay_by=threshold,
            decay_monitor_port=decay_monitor_port,
            dft_min_run_time=dft_min_run_time,
        )

    # -------------------------------------------------------------------------
    # Resolution config
    # -------------------------------------------------------------------------

    def set_resolution(
        self,
        *,
        pixels_per_um: int | None = None,
        preset: str | None = None,
    ) -> None:
        """Configure MEEP grid resolution.

        Args:
            pixels_per_um: Grid pixels per micrometer
            preset: Resolution preset ("coarse", "default", "fine")
        """
        if preset is not None:
            if preset == "coarse":
                self.resolution_config = ResolutionConfig.coarse()
            elif preset == "fine":
                self.resolution_config = ResolutionConfig.fine()
            else:
                self.resolution_config = ResolutionConfig.default()
        elif pixels_per_um is not None:
            self.resolution_config = ResolutionConfig(pixels_per_um=pixels_per_um)

    # -------------------------------------------------------------------------
    # Domain / PML config
    # -------------------------------------------------------------------------

    def set_domain(
        self,
        margin: float | None = None,
        *,
        margin_xy: float | None = None,
        margin_z: float | None = None,
        margin_z_above: float | None = None,
        margin_z_below: float | None = None,
        port_margin: float = 0.5,
        dpml: float = 1.0,
        extend_ports: float = 0.0,
    ) -> None:
        """Configure simulation domain margins and PML thickness.

        Margins control how much material is kept around the geometry.
        Along z, ``set_z_crop()`` uses ``margin_z_above`` / ``margin_z_below``
        to determine how much of the layer stack to keep around the core.
        Along XY, the margin is the gap between geometry bbox and PML.

        Resolution order for each axis:
            margin_z_above/margin_z_below > margin_z > margin > default (0.5)

        Args:
            margin: Uniform margin in all directions (um).
            margin_xy: XY margin between geometry and PML (um).
            margin_z: Z margin above and below core (um).
            margin_z_above: Z margin above core (um).
            margin_z_below: Z margin below core (um).
            port_margin: Margin on each side of waveguide width for mode
                monitors/sources (um). Default 0.5.
            dpml: PML absorber thickness in um.
            extend_ports: Length to extend waveguide ports into PML (um).
                0 = auto-calculate as margin_xy + dpml.
        """
        default = 0.5
        base = margin if margin is not None else default
        xy = margin_xy if margin_xy is not None else base
        z = margin_z if margin_z is not None else base
        z_above = margin_z_above if margin_z_above is not None else z
        z_below = margin_z_below if margin_z_below is not None else z

        self.domain_config = DomainConfig(
            dpml=dpml,
            margin_xy=xy,
            margin_z_above=z_above,
            margin_z_below=z_below,
            port_margin=port_margin,
            extend_ports=extend_ports,
        )

    # -------------------------------------------------------------------------
    # Accuracy / performance
    # -------------------------------------------------------------------------

    def set_accuracy(
        self,
        *,
        eps_averaging: bool = True,
        subpixel_maxeval: int = 0,
        subpixel_tol: float = 1e-4,
        simplify_tol: float = 0.0,
        verbose_interval: float = 0,
    ) -> None:
        """Configure accuracy and performance trade-offs.

        Args:
            eps_averaging: Enable MEEP subpixel averaging (expensive for
                complex polygons).
            subpixel_maxeval: Maximum integration evaluations for subpixel
                averaging (0 = unlimited).
            subpixel_tol: Convergence tolerance for subpixel integration.
            simplify_tol: Shapely polygon simplification tolerance in um.
                Reduces vertex count on dense GDS curves.  0 = no
                simplification.
            verbose_interval: Print progress every *interval* MEEP time
                units during FDTD stepping.  0 = silent.
        """
        self.accuracy_config = AccuracyConfig(
            eps_averaging=eps_averaging,
            subpixel_maxeval=subpixel_maxeval,
            subpixel_tol=subpixel_tol,
            simplify_tol=simplify_tol,
        )
        self.verbose_interval = verbose_interval

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def set_diagnostics(
        self,
        *,
        save_geometry: bool = True,
        save_fields: bool = True,
        save_epsilon_raw: bool = False,
        save_animation: bool = False,
        animation_interval: float = 0.5,
        preview_only: bool = False,
    ) -> None:
        """Configure diagnostic outputs from the MEEP runner.

        Args:
            save_geometry: Save pre-run geometry cross-section plots.
            save_fields: Save post-run field snapshot.
            save_epsilon_raw: Save raw epsilon numpy array.
            save_animation: Save field animation MP4 (heavy, needs ffmpeg).
            animation_interval: MEEP time units between animation frames.
            preview_only: If True, init sim and save geometry diagnostics
                but skip the FDTD run entirely. Fast geometry validation.
        """
        self.diagnostics_config = DiagnosticsConfig(
            save_geometry=save_geometry,
            save_fields=save_fields,
            save_epsilon_raw=save_epsilon_raw,
            save_animation=save_animation,
            animation_interval=animation_interval,
            preview_only=preview_only,
        )

    # -------------------------------------------------------------------------
    # Source port (deprecated)
    # -------------------------------------------------------------------------

    def set_source_port(self, name: str) -> None:
        """Set which port is the excitation source.

        .. deprecated::
            Use ``set_source(port=name)`` instead.

        Args:
            name: Port name (must match a gdsfactory component port)
        """
        warnings.warn(
            "set_source_port() is deprecated. Use set_source(port=name) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.source_config = SourceConfig(
            bandwidth=self.source_config.bandwidth,
            port=name,
        )

    # -------------------------------------------------------------------------
    # Symmetry
    # -------------------------------------------------------------------------

    def set_symmetry(
        self,
        *,
        x: int | None = None,
        y: int | None = None,
        z: int | None = None,
    ) -> None:
        """Set mirror symmetry planes for the simulation.

        Each axis with a non-None phase (+1 or -1) adds a mirror symmetry.

        .. warning::

            Symmetries are **ignored** during S-parameter extraction runs.
            MEEP's ``get_eigenmode_coefficients`` with ``add_mode_monitor``
            produces incorrect normalization when the source monitor
            straddles a mirror symmetry plane (~2x coefficient error).
            This matches gplugins, which also never uses ``mp.Mirror``
            for S-parameter extraction.

            Symmetries are only applied in **preview-only** mode
            (geometry validation, no FDTD run).

        Args:
            x: Phase (+1 or -1) for X mirror symmetry, or None to skip.
            y: Phase (+1 or -1) for Y mirror symmetry, or None to skip.
            z: Phase (+1 or -1) for Z mirror symmetry, or None to skip.
        """
        entries: list[SymmetryEntry] = []
        for direction, phase in [("X", x), ("Y", y), ("Z", z)]:
            if phase is not None:
                if phase not in (1, -1):
                    raise ValueError(
                        f"Phase for {direction} must be +1 or -1, got {phase}"
                    )
                entries.append(SymmetryEntry(direction=direction, phase=phase))
        self.symmetries = entries
        if entries:
            logger.warning(
                "Symmetries are ignored during S-parameter extraction "
                "(causes incorrect eigenmode normalization). They are only "
                "applied in preview-only mode."
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

        if self.geometry is not None:
            ports = list(self.geometry.component.ports)
            if not ports:
                errors.append("Component has no ports.")
            elif self.source_config.port is not None:
                port_names = [p.name for p in ports]
                if self.source_config.port not in port_names:
                    errors.append(
                        f"Source port '{self.source_config.port}' not found. "
                        f"Available: {port_names}"
                    )

        if self.fdtd_config.bandwidth <= 0:
            warnings_list.append(
                "Bandwidth is zero; simulation will use a single frequency."
            )

        valid = len(errors) == 0
        return ValidationResult(valid=valid, errors=errors, warnings=warnings_list)

    # -------------------------------------------------------------------------
    # Config writing
    # -------------------------------------------------------------------------

    def write_config(self) -> Path:
        """Serialize simulation config to output directory.

        Writes:
        - layout.gds: The component GDS file
        - sim_config.json: Simulation config (layer stack, ports, materials, fdtd)
        - run_meep.py: Self-contained MEEP runner script

        Returns:
            Path to the output directory

        Raises:
            ValueError: If output_dir not set or config is invalid
        """
        from gsim.meep.materials import resolve_materials
        from gsim.meep.ports import extract_port_info
        from gsim.meep.script import generate_meep_script

        if self._output_dir is None:
            raise ValueError("Output directory not set. Call set_output_dir() first.")

        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        if self.stack is None:
            self._resolve_stack()

        if self.stack is None:
            raise ValueError("Stack resolution failed.")
        if self.geometry is None:
            raise ValueError("No geometry set.")

        import gdsfactory as gf

        original_component = self.geometry.component.copy()

        # Compute port extension length
        extend_length = self.domain_config.extend_ports
        if extend_length == 0.0:
            extend_length = self.domain_config.margin_xy + self.domain_config.dpml

        # Extend waveguide ports into PML region
        original_bbox: list[float] | None = None
        if extend_length > 0:
            bbox = original_component.dbbox()
            original_bbox = [bbox.left, bbox.bottom, bbox.right, bbox.top]
            component = gf.components.extend_ports(
                original_component, length=extend_length
            )
        else:
            component = original_component

        # 1. Write EXTENDED component GDS
        gds_path = self._output_dir / "layout.gds"
        component.write_gds(gds_path)

        # 2. Build layer stack entries from our Layer objects
        layer_stack_entries = []
        used_materials: set[str] = set()
        for layer_name, layer in self.stack.layers.items():
            layer_stack_entries.append(
                {
                    "layer_name": layer_name,
                    "gds_layer": list(layer.gds_layer),
                    "zmin": layer.zmin,
                    "zmax": layer.zmax,
                    "material": layer.material,
                    "sidewall_angle": layer.sidewall_angle,
                }
            )
            used_materials.add(layer.material)

        # 2b. Build dielectric entries (background slabs from stack)
        dielectric_entries = []
        for diel in self.stack.dielectrics:
            dielectric_entries.append(
                {
                    "name": diel["name"],
                    "zmin": diel["zmin"],
                    "zmax": diel["zmax"],
                    "material": diel["material"],
                }
            )
            used_materials.add(diel["material"])

        # 3. Extract port info from ORIGINAL component (port centers must not change)
        port_infos = extract_port_info(
            original_component, self.stack, source_port=self.source_config.port
        )

        # 4. Resolve materials (layers + dielectrics)
        material_data = resolve_materials(used_materials, overrides=self.materials)

        # 5. Build SimConfig
        fdtd_dict = self.fdtd_config.to_dict()
        sim_config = SimConfig(
            gds_filename="layout.gds",
            component_bbox=original_bbox,
            layer_stack=layer_stack_entries,
            dielectrics=dielectric_entries,
            ports=[p.to_dict() for p in port_infos],
            materials={name: mat.to_dict() for name, mat in material_data.items()},
            fdtd=fdtd_dict,
            source=self.source_config.to_dict(
                self.fdtd_config.fcen, self.fdtd_config.df
            ),
            stopping=self.stopping_config.model_dump(),
            resolution=self.resolution_config.to_dict(),
            domain=self.domain_config.to_dict(),
            accuracy=self.accuracy_config.to_dict(),
            diagnostics=self.diagnostics_config.to_dict(),
            verbose_interval=self.verbose_interval,
            symmetries=[s.to_dict() for s in self.symmetries],
            split_chunks_evenly=self.split_chunks_evenly,
        )

        # 6. Write JSON config
        config_path = self._output_dir / "sim_config.json"
        sim_config.to_json(config_path)

        # 7. Write runner script
        script_path = self._output_dir / "run_meep.py"
        script_content = generate_meep_script(config_filename="sim_config.json")
        script_path.write_text(script_content)

        logger.info("Config written to %s", self._output_dir)
        return self._output_dir

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    def simulate(
        self,
        *,
        verbose: bool = True,
    ) -> SParameterResult:
        """Run MEEP simulation on GDSFactory+ cloud.

        Writes config, uploads to cloud, waits for completion,
        downloads results, and parses S-parameters.

        Args:
            verbose: Print progress messages

        Returns:
            SParameterResult with parsed S-parameters

        Raises:
            ValueError: If output_dir not set
            RuntimeError: If simulation fails
        """
        from gsim.gcloud import run_simulation

        self.write_config()

        if self._output_dir is None:
            raise ValueError("Output directory not set.")

        results = run_simulation(
            self._output_dir,
            job_type="meep",
            verbose=verbose,
        )

        csv_path = results.get("s_parameters.csv")
        if csv_path is not None:
            return SParameterResult.from_csv(csv_path)

        logger.warning("No s_parameters.csv found in results")
        return SParameterResult()


__all__ = ["MeepSim"]
