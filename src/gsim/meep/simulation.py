"""Declarative Simulation container for MEEP photonic FDTD.

Translates the user-facing declarative API objects into the existing
``SimConfig`` JSON contract consumed by the cloud runner.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from gsim.meep.models.api import (
    FDTD,
    Domain,
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


def estimate_meep_np(
    cell_x: float,
    cell_y: float,
    cell_z: float,
    pixels_per_um: int,
    *,
    max_cores: int = 24,
    min_voxels_per_proc: int = 200_000,
) -> int:
    """Estimate optimal MPI process count from problem size.

    Args:
        cell_x: Cell size along X in um.
        cell_y: Cell size along Y in um.
        cell_z: Cell size along Z in um.
        pixels_per_um: Grid resolution (pixels per um).
        max_cores: Maximum physical cores available (default 24 for c6i.12xlarge).
        min_voxels_per_proc: Minimum voxels per process to keep MPI overhead < ~5%.

    Returns:
        Recommended number of MPI processes (1 to max_cores).
    """
    nx = math.ceil(cell_x * pixels_per_um)
    ny = math.ceil(cell_y * pixels_per_um)
    nz = max(1, math.ceil(cell_z * pixels_per_um))
    total_voxels = nx * ny * nz
    np_est = total_voxels // min_voxels_per_proc
    return max(1, min(np_est, max_cores))


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
    )

    geometry: Geometry = Field(default_factory=Geometry)
    materials: dict[str, float | Material] = Field(default_factory=dict)
    source: ModeSource = Field(default_factory=ModeSource)
    monitors: list[str] = Field(default_factory=list)
    domain: Domain = Field(default_factory=Domain)
    solver: FDTD = Field(default_factory=FDTD)

    # Private: kwargs captured from geometry.stack when it's a string/path
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

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
            if not ports:
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

        if self.geometry.stack is None:
            warnings_list.append(
                "No stack configured. Will use active PDK with defaults."
            )

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
        from gsim.meep.ports import _find_highest_n_layer

        stack = self.geometry.stack
        if stack is None:
            raise ValueError("No stack configured for z-crop.")

        # Find reference layer
        ref: Layer | None = None
        if self.geometry.z_crop == "auto":
            ref, best_n = _find_highest_n_layer(stack)
            if ref is None or best_n <= 1.5:
                raise ValueError(
                    "Could not auto-detect core layer (no layer with n > 1.5). "
                    "Set geometry.z_crop to an explicit layer name."
                )
        else:
            layer_name = self.geometry.z_crop
            if layer_name not in stack.layers:
                raise ValueError(
                    f"Layer '{layer_name}' not found. "
                    f"Available: {list(stack.layers.keys())}"
                )
            ref = stack.layers[layer_name]

        z_lo = ref.zmin - self.domain.margin_z_below
        z_hi = ref.zmax + self.domain.margin_z_above

        # Filter and clip layers
        cropped: dict[str, Layer] = {}
        for name, layer in stack.layers.items():
            if layer.zmax <= z_lo or layer.zmin >= z_hi:
                continue
            new_zmin = max(layer.zmin, z_lo)
            new_zmax = min(layer.zmax, z_hi)
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
        # Mark as applied by clearing the z_crop setting
        self.geometry.z_crop = None

    # -------------------------------------------------------------------------
    # Internal: translate to config objects
    # -------------------------------------------------------------------------

    def _wavelength_config(self) -> Any:
        """Derive WavelengthConfig from source."""
        from gsim.meep.models.config import WavelengthConfig

        return WavelengthConfig(
            wavelength=self.source.wavelength,
            bandwidth=self.source.wavelength_span,
            num_freqs=self.source.num_freqs,
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
        from gsim.meep.materials import resolve_materials
        from gsim.meep.models.config import LayerStackEntry, SimConfig, SymmetryEntry
        from gsim.meep.ports import extract_port_info

        validation = self.validate_config()
        if not validation.valid:
            raise ValueError("Invalid configuration:\n" + "\n".join(validation.errors))

        # Resolve stack
        self._ensure_stack()
        if self.geometry.stack is None:
            raise ValueError("Stack resolution failed.")
        if self.geometry.component is None:
            raise ValueError("No geometry set.")

        # Apply z-crop if requested
        self._apply_z_crop()

        import gdsfactory as gf

        original_component = self.geometry.component.copy()
        stack = self.geometry.stack

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
            original_component, stack, source_port=source_cfg.port
        )

        # Resolve materials
        material_data = resolve_materials(
            used_materials, overrides=self._material_overrides()
        )

        # Compute source fwidth
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

        # Build SimConfig
        sim_config = SimConfig(
            gds_filename="layout.gds",
            component_bbox=original_bbox,
            layer_stack=layer_stack_entries,
            dielectrics=dielectric_entries,
            ports=port_infos,
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

        # Estimate MPI process count
        dpml = domain_cfg.dpml
        margin_xy = domain_cfg.margin_xy
        if original_bbox is not None:
            bbox_w = original_bbox[2] - original_bbox[0]
            bbox_h = original_bbox[3] - original_bbox[1]
        else:
            bbox = component.dbbox()
            bbox_w = bbox.right - bbox.left
            bbox_h = bbox.top - bbox.bottom
        # Use both layer and dielectric z-ranges for the cell height.
        # PDKs without explicit box/clad layers (e.g. cspdk) would otherwise
        # produce a cell that's too short, with PML touching the core.
        z_vals = [e.zmin for e in layer_stack_entries] + [
            e.zmax for e in layer_stack_entries
        ]
        for d in dielectric_entries:
            z_vals.extend((d["zmin"], d["zmax"]))
        z_min = min(z_vals)
        z_max = max(z_vals)
        cell_x = bbox_w + 2 * (margin_xy + dpml)
        cell_y = bbox_h + 2 * (margin_xy + dpml)
        cell_z = (z_max - z_min) + 2 * dpml

        _meep_np_est = estimate_meep_np(
            cell_x, cell_y, cell_z, resolution_cfg.pixels_per_um
        )
        # TODO: use _meep_np_est once Batch vCPU allocation is passed to the
        # container (lscpu reports host cores, not container vCPUs → OOM).
        meep_np = 2
        sim_config.meep_np = meep_np
        logger.info(
            "meep_np=%d (estimated %d, cell %.1f x %.1f x %.1f um, res %d)",
            meep_np,
            _meep_np_est,
            cell_x,
            cell_y,
            cell_z,
            resolution_cfg.pixels_per_um,
        )

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
        verbose: bool = True,
        parent_dir: str | Path | None = None,
    ) -> Any:
        """Wait for this sim's cloud job, download and parse results.

        Args:
            verbose: Print progress messages.
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
        verbose: bool = True,
        wait: bool = True,
    ) -> Any:
        """Run MEEP simulation on the cloud.

        Args:
            parent_dir: Where to create the sim directory.
                Defaults to the current working directory.
            verbose: Print progress info.
            wait: If ``True`` (default), block until results are ready.
                If ``False``, upload + start and return the ``job_id``.

        Returns:
            ``SParameterResult`` when ``wait=True``, or ``job_id`` string
            when ``wait=False``.
        """
        self.upload(verbose=False)
        self.start(verbose=verbose)
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

        Accepts the same keyword arguments as :func:`gsim.meep.viz.plot_2d`.
        """
        from gsim.meep.viz import plot_2d

        result = self.build_config()

        return plot_2d(
            component=result.component,
            stack=self.geometry.stack,
            domain_config=result.config.domain,
            source_port=result.config.source.port,
            extend_ports_length=0,
            port_data=result.config.ports,
            component_bbox=result.config.component_bbox,
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
