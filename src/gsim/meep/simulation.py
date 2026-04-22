"""Declarative Simulation container for MEEP photonic FDTD.

Translates the user-facing declarative API objects into the existing
``SimConfig`` JSON contract consumed by the cloud runner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from gsim.meep.models.api import (
    FDTD,
    Domain,
    FiberSource,
    Geometry,
    Material,
    ModeSource,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BuildResult
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    """Result of :meth:`Simulation.build_config` — single source of truth.

    Attributes:
        config: Full serializable SimConfig.
        component: Extended component (what meep actually simulates).
        original_component: Original component before port extension.
    """

    config: Any  # SimConfig
    component: Any  # gdsfactory Component (extended)
    original_component: Any  # gdsfactory Component (original)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


class Simulation(BaseModel):
    """Declarative MEEP FDTD simulation container.

    Assigns typed physics objects, then calls ``write_config()`` to
    produce the JSON + GDS + runner consumed by the cloud engine.

    Example::

        from gsim import meep

        sim = meep.Simulation()
        sim.geometry.component = ybranch
        sim.geometry.stack = stack
        sim.materials = {"si": 3.47, "sio2": 1.44}
        sim.source.port = "o1"
        sim.monitors = ["o1", "o2"]
        sim.solver.stopping = "dft_decay"
        sim.solver.max_time = 200
        result = sim.run()  # creates sim-data-{job_name}/ in CWD
    """

    model_config = ConfigDict(
        validate_assignment=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    geometry: Geometry = Field(default_factory=Geometry)
    materials: dict[str, float | Material] = Field(default_factory=dict)
    source: ModeSource = Field(default_factory=ModeSource)
    fiber_source: FiberSource | None = Field(
        default=None,
        description=(
            "Gaussian-beam fiber source for XZ 2D grating-coupler sims. "
            "When set, takes precedence over mode-source `source`."
        ),
    )
    monitors: list[str] = Field(default_factory=list)
    num_freqs: int = Field(
        default=11,
        ge=1,
        description=(
            "Number of frequency points sampled by flux/DFT monitors. "
            "Orthogonal to source choice — parameterizes the measurement "
            "grid, not the excitation."
        ),
    )
    domain: Domain = Field(default_factory=Domain)
    solver: FDTD = Field(default_factory=FDTD)

    # Private: kwargs captured from geometry.stack when it's a string/path
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Extra hints forwarded into the config JSON (not part of the schema).
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Cloud job state (set by upload/run)
    _job_id: str | None = PrivateAttr(default=None)
    _config_dir: Path | None = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("materials", mode="before")
    @classmethod
    def _normalize_materials(
        cls,
        v: dict[str, float | Material | dict],
    ) -> dict[str, float | Material]:
        """Accept float shorthand: ``{"si": 3.47}`` → ``Material(n=3.47)``."""
        out: dict[str, float | Material] = {}
        for name, val in v.items():
            if isinstance(val, (int, float)):
                out[name] = Material(n=float(val))
            elif isinstance(val, dict):
                out[name] = Material(**val)
            else:
                out[name] = val
        return out

    # -------------------------------------------------------------------------
    # Resolved materials helper
    # -------------------------------------------------------------------------

    def _resolved_materials(self) -> dict[str, Material]:
        """Return materials dict with all values normalized to Material."""
        out: dict[str, Material] = {}
        for name, val in self.materials.items():
            if isinstance(val, (int, float)):
                out[name] = Material(n=float(val))
            else:
                out[name] = val
        return out

    # -------------------------------------------------------------------------
    # Fiber source helper
    # -------------------------------------------------------------------------

    def source_fiber(self, **kwargs: Any) -> FiberSource:
        """Configure a tilted Gaussian-beam fiber source (XZ 2D only).

        Replaces any previous fiber source. Requires ``solver.is_3d=False``
        (and eventually ``solver.plane='xz'`` at ``build_config`` time).

        The ``waist`` kwarg is the 1/e² intensity *radius* (= MFD / 2),
        matching MEEP's ``beam_w0``. Typical SMF-28 values:
        ``waist ≈ 4.6 um`` at 1310 nm (MFD ≈ 9.2 um) and
        ``waist ≈ 5.2 um`` at 1550 nm (MFD ≈ 10.4 um).

        Args:
            **kwargs: Fields forwarded to :class:`FiberSource`.

        Returns:
            The newly created :class:`FiberSource` instance.
        """
        if self.solver.is_3d:
            raise ValueError(
                "fiber source requires is_3d=False (and plane='xz') — "
                "currently is_3d=True"
            )
        self.fiber_source = FiberSource(**kwargs)
        return self.fiber_source

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_config(self) -> Any:
        """Validate the simulation configuration.

        Returns:
            ValidationResult with errors/warnings.
        """
        from gsim.common import ValidationResult

        errors: list[str] = []
        warnings_list: list[str] = []

        if self.geometry.component is None:
            errors.append("No component set. Assign sim.geometry.component first.")

        if self.geometry.component is not None:
            ports = list(self.geometry.component.ports)
            if not ports and self.solver.plane != "xz":
                errors.append("Component has no ports.")
            elif self.source.port is not None:
                port_names = [p.name for p in ports]
                if self.source.port not in port_names:
                    errors.append(
                        f"Source port '{self.source.port}' not found. "
                        f"Available: {port_names}"
                    )

            # Validate monitor port names
            if ports and self.monitors:
                port_names = [p.name for p in ports]
                errors.extend(
                    f"Monitor port '{m}' not found. Available: {port_names}"
                    for m in self.monitors
                    if m not in port_names
                )

        # TODO: refactor source into a more coherent style. Today `Simulation.source`
        # is a default-factory ModeSource that's always present, so we detect
        # "user opted into mode source" via `source.port is not None` (Option A).
        # A cleaner design would make source selection explicit — e.g. a single
        # `Simulation.source` union field set by `sim.source(...)` or
        # `sim.source_fiber(...)`, so "exactly one source" is a type-level invariant.
        if self.fiber_source is not None and self.source.port is not None:
            errors.append(
                "Both `source.port` and `fiber_source` are set. Exactly one source "
                "drives the sim — unset source.port, or drop the fiber source."
            )

        if self.geometry.stack is None:
            warnings_list.append(
                "No stack configured. Will use active PDK with defaults."
            )

        # Inform about stopping mode
        s = self.solver
        if s.stopping == "energy_decay":
            warnings_list.append(
                f"Stopping: energy_decay (dt={s.stopping_dt}, "
                f"decay_by={s.stopping_threshold}, cap={s.max_time})"
            )
        elif s.stopping == "field_decay":
            warnings_list.append(
                f"Stopping: field_decay (component={s.stopping_component}, "
                f"dt={s.stopping_dt}, decay_by={s.stopping_threshold}, "
                f"cap={s.max_time})"
            )
        elif s.stopping == "dft_decay":
            warnings_list.append(
                f"Stopping: dft_decay (tol={s.stopping_threshold}, "
                f"min_time={s.stopping_min_time}, cap={s.max_time})"
            )
        elif s.stopping == "fixed":
            warnings_list.append(f"Stopping: fixed (time={s.max_time})")

        return ValidationResult(
            valid=len(errors) == 0, errors=errors, warnings=warnings_list
        )

    # -------------------------------------------------------------------------
    # Internal: stack resolution
    # -------------------------------------------------------------------------

    def _ensure_stack(self) -> None:
        """Lazily resolve the layer stack if not yet built."""
        if self.geometry.stack is not None:
            return

        from gsim.common.stack import get_stack

        if self._stack_kwargs:
            yaml_path = self._stack_kwargs.pop("yaml_path", None)
            self.geometry.stack = get_stack(yaml_path=yaml_path, **self._stack_kwargs)
            self._stack_kwargs["yaml_path"] = yaml_path
        else:
            # Fall back to active PDK defaults
            self.geometry.stack = get_stack()

    # -------------------------------------------------------------------------
    # Internal: fiber-aware z margin
    # -------------------------------------------------------------------------

    def _stack_material_extent(self) -> tuple[float, float] | None:
        """Return (zmin, zmax) spanning all non-air layers and dielectrics.

        This is the reference used by auto z-crop and by the fiber-aware
        margin expansion: it preserves the full fabricated stack (BOX,
        core, cladding, passive, ...) and trims only the synthetic air
        padding added by the extractor.
        """
        stack = self.geometry.stack
        if stack is None:
            return None
        zmins: list[float] = []
        zmaxs: list[float] = []
        for layer in stack.layers.values():
            if layer.material != "air":
                zmins.append(layer.zmin)
                zmaxs.append(layer.zmax)
        for diel in stack.dielectrics:
            if diel.get("material") != "air":
                zmins.append(diel["zmin"])
                zmaxs.append(diel["zmax"])
        if not zmins:
            return None
        return min(zmins), max(zmaxs)

    def _expand_margin_z_above_for_fiber(self) -> None:
        """Bump ``domain.margin_z_above`` to include the fiber source plane.

        When the user configures ``sim.source_fiber(...)`` the Gaussian beam
        sits at absolute z = ``fs.z``. The z-crop shrinks the stack around
        the physical-stack top, so ``margin_z_above`` must be large enough
        that ``stack_top + margin_z_above`` still sits above the beam plane
        plus a waist-sized buffer (otherwise the fiber ends up in PML).
        """
        if self.fiber_source is None:
            return
        fs = self.fiber_source
        extent = self._stack_material_extent()
        if extent is None:
            return
        _, stack_top = extent
        # Room for the beam plane + beam-half-waist so the Gaussian tail
        # is inside the sim cell before PML.
        needed = (fs.z - stack_top) + max(fs.waist / 2.0, 0.5)
        if self.domain.margin_z_above < needed:
            self.domain.margin_z_above = needed

    # -------------------------------------------------------------------------
    # Internal: z-crop
    # -------------------------------------------------------------------------

    def _apply_z_crop(self) -> None:
        """Apply z-crop to the stack if geometry.z_crop is set.

        Only applies once per stack — after cropping, sets z_crop to None
        to prevent double-cropping on subsequent calls.
        """
        if self.geometry.z_crop is None:
            return

        from gsim.common.stack.extractor import Layer, LayerStack

        stack = self.geometry.stack
        if stack is None:
            raise ValueError("No stack configured for z-crop.")

        z_crop_setting = self.geometry.z_crop

        # Determine the z-range to preserve ("ref window") before margins.
        # "auto" uses the full non-air stack extent (BOX through cladding),
        # so the fabricated stack stays intact and only synthetic air
        # padding above/below gets trimmed. A named layer restricts the
        # window to that single layer's z-extent.
        ref_name: str
        if z_crop_setting == "auto":
            extent = self._stack_material_extent()
            if extent is None:
                raise ValueError(
                    "Could not detect any non-air layers/dielectrics for "
                    "auto z-crop. Set geometry.z_crop to an explicit layer name."
                )
            ref_zmin, ref_zmax = extent
            ref_name = "stack"
        else:
            ref_name = z_crop_setting
            if ref_name not in stack.layers:
                raise ValueError(
                    f"Layer '{ref_name}' not found. "
                    f"Available: {list(stack.layers.keys())}"
                )
            ref: Layer = stack.layers[ref_name]
            ref_zmin, ref_zmax = ref.zmin, ref.zmax

        z_lo = ref_zmin - self.domain.margin_z_below
        z_hi = ref_zmax + self.domain.margin_z_above

        # Filter and clip layers
        cropped: dict[str, Layer] = {}
        trimmed_names: list[str] = []
        removed_names: list[str] = []
        for name, layer in stack.layers.items():
            if layer.zmax <= z_lo or layer.zmin >= z_hi:
                removed_names.append(name)
                continue
            new_zmin = max(layer.zmin, z_lo)
            new_zmax = min(layer.zmax, z_hi)
            if new_zmin != layer.zmin or new_zmax != layer.zmax:
                trimmed_names.append(name)
            cropped[name] = layer.model_copy(
                update={
                    "zmin": new_zmin,
                    "zmax": new_zmax,
                    "thickness": new_zmax - new_zmin,
                }
            )

        # Crop dielectrics
        cropped_dielectrics = []
        for diel in stack.dielectrics:
            if diel["zmax"] <= z_lo or diel["zmin"] >= z_hi:
                continue
            cropped_dielectrics.append(
                {
                    **diel,
                    "zmin": max(diel["zmin"], z_lo),
                    "zmax": min(diel["zmax"], z_hi),
                }
            )

        self.geometry.stack = LayerStack(
            pdk_name=stack.pdk_name,
            units=stack.units,
            layers=cropped,
            materials=stack.materials,
            dielectrics=cropped_dielectrics,
            simulation=stack.simulation,
        )
        logger.info(
            "z_crop=%r applied (ref=%r, z=[%.4g, %.4g]); trimmed %d layer(s): %s; "
            "removed %d layer(s) fully outside crop: %s",
            z_crop_setting,
            ref_name,
            z_lo,
            z_hi,
            len(trimmed_names),
            trimmed_names,
            len(removed_names),
            removed_names,
        )
        # Clear z_crop so repeat calls to build_config() (e.g. plot_2d then run)
        # don't re-crop an already-cropped stack. Invariant: after this method
        # runs successfully, self.geometry.stack is the cropped stack and
        # self.geometry.z_crop is None.
        self.geometry.z_crop = None

    # -------------------------------------------------------------------------
    # Internal: translate to config objects
    # -------------------------------------------------------------------------

    def _wavelength_config(self) -> Any:
        """Derive WavelengthConfig from the active source + sim-level num_freqs."""
        from gsim.meep.models.config import WavelengthConfig

        active = self.fiber_source if self.fiber_source is not None else self.source
        return WavelengthConfig(
            wavelength=active.wavelength,
            bandwidth=active.wavelength_span,
            num_freqs=self.num_freqs,
        )

    def _source_config(self) -> Any:
        """Translate ModeSource → SourceConfig."""
        from gsim.meep.models.config import SourceConfig

        return SourceConfig(
            bandwidth=None,
            port=self.source.port,
        )

    def _stopping_config(self) -> Any:
        """Translate FDTD stopping fields → StoppingConfig."""
        from gsim.meep.models.config import StoppingConfig

        s = self.solver
        return StoppingConfig(
            mode=s.stopping,
            max_time=s.max_time,
            threshold=s.stopping_threshold,
            dft_min_run_time=s.stopping_min_time,
            decay_component=s.stopping_component,
            decay_dt=s.stopping_dt,
            decay_monitor_port=s.stopping_monitor_port,
            wall_time_max=s.wall_time_max,
        )

    def _domain_config(self) -> Any:
        """Translate Domain → DomainConfig."""
        from gsim.meep.models.config import DomainConfig

        return DomainConfig(
            dpml=self.domain.pml,
            margin_xy=self.domain.margin,
            margin_z_above=self.domain.margin_z_above,
            margin_z_below=self.domain.margin_z_below,
            port_margin=self.domain.port_margin,
            extend_ports=self.domain.extend_ports,
            source_port_offset=self.domain.source_port_offset,
            distance_source_to_monitors=self.domain.distance_source_to_monitors,
        )

    def _resolution_config(self) -> Any:
        """Translate FDTD.resolution → ResolutionConfig."""
        from gsim.meep.models.config import ResolutionConfig

        return ResolutionConfig(pixels_per_um=self.solver.resolution)

    def _accuracy_config(self) -> Any:
        """Translate FDTD accuracy fields → AccuracyConfig."""
        from gsim.meep.models.config import AccuracyConfig

        return AccuracyConfig(
            eps_averaging=self.solver.subpixel,
            subpixel_maxeval=self.solver.subpixel_maxeval,
            subpixel_tol=self.solver.subpixel_tol,
            simplify_tol=self.solver.simplify_tol,
        )

    def _diagnostics_config(self) -> Any:
        """Translate FDTD diagnostic fields → DiagnosticsConfig."""
        from gsim.meep.models.config import DiagnosticsConfig

        return DiagnosticsConfig(
            save_geometry=self.solver.save_geometry,
            save_fields=self.solver.save_fields,
            save_epsilon_raw=self.solver.save_epsilon_raw,
            save_animation=self.solver.save_animation,
            animation_interval=self.solver.animation_interval,
            preview_only=self.solver.preview_only,
            verbose_interval=self.solver.verbose_interval,
        )

    def _material_overrides(self) -> dict[str, Any]:
        """Convert materials dict to MaterialProperties overrides."""
        from gsim.common.stack.materials import MaterialProperties

        overrides: dict[str, MaterialProperties] = {}
        for name, val in self._resolved_materials().items():
            overrides[name] = MaterialProperties(
                type="dielectric",
                refractive_index=val.n,
                extinction_coeff=val.k,
            )
        return overrides

    # -------------------------------------------------------------------------
    # build_config — single source of truth
    # -------------------------------------------------------------------------

    def build_config(self) -> BuildResult:
        """Build the complete simulation config (single source of truth).

        All computation — validation, stack resolution, z-crop, port
        extension, material resolution, MPI estimation — happens here.
        Both :meth:`write_config` and the viz methods consume this output.

        Returns:
            BuildResult with SimConfig, extended component, and original.

        Raises:
            ValueError: If config is invalid.
        """
        import math

        from gsim.meep.materials import resolve_materials
        from gsim.meep.models.config import (
            FiberSourceConfig,
            LayerStackEntry,
            SimConfig,
            SymmetryEntry,
        )
        from gsim.meep.ports import (
            _find_highest_n_layer,
            extract_port_info,
            filter_ports_for_xz,
        )

        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        is_3d = self.solver.is_3d
        plane = self.solver.plane

        # Resolve stack
        self._ensure_stack()
        if self.geometry.stack is None:
            raise ValueError("Stack resolution failed.")
        if self.geometry.component is None:
            raise ValueError("No geometry set.")

        # Apply z-crop for 3D and XZ 2D (both use the vertical dimension).
        # XY 2D collapses z entirely so cropping is meaningless.
        if is_3d or plane == "xz":
            # For XZ 2D, default z_crop to "auto" so users don't end up with
            # a huge auto-built stack (e.g. the 5 µm air_above) pinning the
            # cell height. This mirrors what 3D notebooks do explicitly.
            if plane == "xz" and self.geometry.z_crop is None:
                self.geometry.z_crop = "auto"

            # When a fiber source is configured in XZ mode, expand
            # margin_z_above so the cropped stack still contains the beam
            # plane (and a little PML headroom).
            self._expand_margin_z_above_for_fiber()

            self._apply_z_crop()

        import gdsfactory as gf

        original_component = self.geometry.component.copy()
        stack = self.geometry.stack

        # Resolve y_cut default for XZ 2D sims.
        if plane == "xz":
            if self.geometry.y_cut is None:
                bbox = original_component.dbbox()
                y_cut: float | None = (bbox.bottom + bbox.top) / 2.0
            else:
                y_cut = self.geometry.y_cut
        else:
            y_cut = self.geometry.y_cut

        # Build config objects
        domain_cfg = self._domain_config()
        wl_cfg = self._wavelength_config()
        source_cfg = self._source_config()
        stopping_cfg = self._stopping_config()
        resolution_cfg = self._resolution_config()
        accuracy_cfg = self._accuracy_config()
        diagnostics_cfg = self._diagnostics_config()

        # Compute port extension length
        extend_length = domain_cfg.extend_ports
        if extend_length == 0.0:
            extend_length = domain_cfg.margin_xy + domain_cfg.dpml

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

        # Build layer stack entries
        layer_stack_entries = []
        used_materials: set[str] = set()
        for layer_name, layer in stack.layers.items():
            layer_stack_entries.append(
                LayerStackEntry(
                    layer_name=layer_name,
                    gds_layer=list(layer.gds_layer),
                    zmin=layer.zmin,
                    zmax=layer.zmax,
                    material=layer.material,
                    sidewall_angle=layer.sidewall_angle,
                )
            )
            used_materials.add(layer.material)

        # Build dielectric entries
        dielectric_entries = []
        for diel in stack.dielectrics:
            dielectric_entries.append(
                {
                    "name": diel["name"],
                    "zmin": diel["zmin"],
                    "zmax": diel["zmax"],
                    "material": diel["material"],
                }
            )
            used_materials.add(diel["material"])

        # Extract port info from original component
        port_infos = extract_port_info(
            original_component, stack, source_port=source_cfg.port, is_3d=is_3d
        )

        # Drop ports that don't intersect the XZ cut.
        if plane == "xz":
            port_infos = filter_ports_for_xz(
                port_infos, y_cut=y_cut if y_cut is not None else 0.0
            )
            # When a fiber source drives the sim, no port is the excitation —
            # demote any auto-tagged source port to a monitor.
            if self.fiber_source is not None:
                port_infos = [
                    p.model_copy(update={"is_source": False}) for p in port_infos
                ]

        # Build FiberSourceConfig (XZ 2D only) with pre-computed k-direction.
        fiber_source_cfg: FiberSourceConfig | None = None
        if self.fiber_source is not None:
            if self.solver.is_3d:
                raise ValueError("fiber source requires is_3d=False (and plane='xz')")
            if plane != "xz":
                raise ValueError("fiber source requires plane='xz'")

            theta = math.radians(self.fiber_source.angle_deg)
            k_direction = [math.sin(theta), 0.0, -math.cos(theta)]

            fiber_source_cfg = FiberSourceConfig(
                x=self.fiber_source.x,
                z=self.fiber_source.z,
                angle_deg=self.fiber_source.angle_deg,
                waist=self.fiber_source.waist,
                wavelength=self.fiber_source.wavelength,
                wavelength_span=self.fiber_source.wavelength_span,
                polarization=self.fiber_source.polarization,
                k_direction=k_direction,
            )

        if plane == "xz" and not port_infos and fiber_source_cfg is None:
            raise ValueError(
                "XZ 2D sim has no valid monitors and no fiber source — "
                "nothing to observe. Either add a port intersecting y_cut, "
                "or call sim.source_fiber(...)."
            )

        # Resolve materials
        material_data = resolve_materials(
            used_materials, overrides=self._material_overrides()
        )

        fwidth = source_cfg.compute_fwidth(wl_cfg.fcen, wl_cfg.df)
        source_for_config = source_cfg.model_copy(update={"fwidth": fwidth})

        # Translate domain.symmetries → SymmetryEntry for config
        symmetry_entries = [
            SymmetryEntry(direction=s.direction, phase=s.phase)
            for s in self.domain.symmetries
        ]
        if symmetry_entries:
            import warnings

            warnings.warn(
                "Symmetries are not yet used in production S-parameter runs "
                "(only applied in preview-only mode).",
                stacklevel=2,
            )

        # Size waveguide port monitors around the core layer (core
        # thickness + 2·port_margin) rather than the full stack. For
        # XZ 2D sims the stack is inflated to hold the fiber beam plane;
        # using the full stack would make the port monitor unreasonably
        # tall.
        core_layer, _ = _find_highest_n_layer(stack)
        if core_layer is not None:
            monitor_z_span: float | None = (
                core_layer.zmax - core_layer.zmin
            ) + 2 * domain_cfg.port_margin
        else:
            monitor_z_span = None

        # Build SimConfig
        sim_config = SimConfig(
            is_3d=is_3d,
            plane=plane,
            y_cut=y_cut,
            fiber_source=fiber_source_cfg,
            gds_filename="layout.gds",
            component_bbox=original_bbox,
            layer_stack=layer_stack_entries,
            dielectrics=dielectric_entries,
            ports=port_infos,
            monitor_z_span=monitor_z_span,
            materials=material_data,
            wavelength=wl_cfg,
            source=source_for_config,
            stopping=stopping_cfg,
            resolution=resolution_cfg,
            domain=domain_cfg,
            accuracy=accuracy_cfg,
            diagnostics=diagnostics_cfg,
            verbose_interval=diagnostics_cfg.verbose_interval,
            symmetries=symmetry_entries,
        )
        # Forward any private hints into the config
        if self._hints:
            sim_config._hints.update(self._hints)  # noqa: SLF001

        return BuildResult(
            config=sim_config,
            component=component,
            original_component=original_component,
        )

    # -------------------------------------------------------------------------
    # write_config
    # -------------------------------------------------------------------------

    def write_config(self, output_dir: str | Path) -> Path:
        """Serialize simulation config to output directory.

        Thin wrapper around :meth:`build_config` — writes GDS, JSON, and
        the runner script.

        Args:
            output_dir: Directory to write layout.gds, sim_config.json, run_meep.py.

        Returns:
            Path to the output directory.

        Raises:
            ValueError: If config is invalid.
        """
        from gsim.meep.script import generate_meep_script

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.build_config()

        # Write extended component GDS
        result.component.write_gds(output_dir / "layout.gds")

        # Write JSON config
        result.config.to_json(output_dir / "sim_config.json")

        # Write runner script
        script_path = output_dir / "run_meep.py"
        script_content = generate_meep_script(config_filename="sim_config.json")
        script_path.write_text(script_content)

        logger.info("Config written to %s", output_dir)
        return output_dir

    # -------------------------------------------------------------------------
    # Cloud: fine-grained control
    # -------------------------------------------------------------------------

    def upload(self, *, verbose: bool = True) -> str:
        """Write config and upload to the cloud. Does NOT start execution.

        Args:
            verbose: Print progress messages.

        Returns:
            ``job_id`` string for use with :meth:`start`, :meth:`get_status`,
            or :func:`gsim.wait_for_results`.
        """
        import tempfile

        from gsim import gcloud

        tmp = Path(tempfile.mkdtemp(prefix="meep_"))
        self.write_config(tmp)
        self._config_dir = tmp
        self._job_id = gcloud.upload(tmp, "meep", verbose=verbose)
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
            verbose: ``"quiet"`` no output, ``"status"`` status line,
                ``"full"`` stream solver logs.
            parent_dir: Where to create the sim-data directory.

        Returns:
            Parsed result (typically ``SParameterResult``).

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
    # run
    # -------------------------------------------------------------------------

    def run(
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        wait: bool = True,
    ) -> Any:
        """Run MEEP simulation on the cloud.

        Args:
            parent_dir: Where to create the sim directory.
                Defaults to the current working directory.
            verbose: ``"quiet"`` no output, ``"status"`` status line,
                ``"full"`` stream solver logs.
            wait: If ``True`` (default), block until results are ready.
                If ``False``, upload + start and return the ``job_id``.

        Returns:
            ``SParameterResult`` when ``wait=True``, or ``job_id`` string
            when ``wait=False``.
        """
        self.upload(verbose=False)
        self.start(verbose=verbose != "quiet")
        if not wait:
            return self._job_id
        return self.wait_for_results(verbose=verbose, parent_dir=parent_dir)

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_2d(self, **kwargs: Any) -> Any:
        """Plot 2D cross-sections of the geometry.

        Uses :meth:`build_config` so the plot shows exactly what meep
        processes — including extended ports and PML boundaries.

        In XZ 2D mode (``solver.plane='xz'``), ``slices`` defaults to
        ``"y"`` and ``y`` defaults to the resolved ``y_cut``.

        Accepts the same keyword arguments as :func:`gsim.meep.viz.plot_2d`.
        """
        from gsim.meep.viz import plot_2d

        result = self.build_config()

        if self.solver.plane == "xz":
            kwargs.setdefault("slices", "y")
            if kwargs.get("slices") == "y":
                kwargs.setdefault("y", result.config.y_cut)

        return plot_2d(
            component=result.component,
            stack=self.geometry.stack,
            domain_config=result.config.domain,
            source_port=result.config.source.port,
            extend_ports_length=0,
            port_data=result.config.ports,
            component_bbox=result.config.component_bbox,
            fiber_source=result.config.fiber_source,
            monitor_z_span=result.config.monitor_z_span,
            **kwargs,
        )

    def plot_3d(self, **kwargs: Any) -> Any:
        """Plot 3D visualization of the geometry.

        Uses :meth:`build_config` so the plot shows exactly what meep
        processes — including extended ports.

        Accepts the same keyword arguments as :func:`gsim.meep.viz.plot_3d`.
        """
        from gsim.meep.viz import plot_3d

        result = self.build_config()

        return plot_3d(
            component=result.component,
            stack=self.geometry.stack,
            domain_config=result.config.domain,
            extend_ports_length=0,
            **kwargs,
        )
