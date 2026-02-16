"""Declarative Simulation container for MEEP photonic FDTD.

Translates the user-facing declarative API objects into the existing
``SimConfig`` JSON contract consumed by the cloud runner.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from gsim.meep.models.api import (
    DFTDecay,
    Domain,
    FDTD,
    FieldDecay,
    FixedTime,
    Geometry,
    Material,
    ModeSource,
)

logger = logging.getLogger(__name__)


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
        sim.solver.stopping = meep.DFTDecay(threshold=1e-3, min_time=100)
        result = sim.run("./meep-sim")
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

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("materials", mode="before")
    @classmethod
    def _normalize_materials(
        cls, v: dict[str, float | Material | dict],
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
            errors.append(
                "No component set. Assign sim.geometry.component first."
            )

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
                for mon_port in self.monitors:
                    if mon_port not in port_names:
                        errors.append(
                            f"Monitor port '{mon_port}' not found. "
                            f"Available: {port_names}"
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
        if self.geometry.stack is None and self._stack_kwargs:
            from gsim.common.stack import get_stack

            yaml_path = self._stack_kwargs.pop("yaml_path", None)
            self.geometry.stack = get_stack(
                yaml_path=yaml_path, **self._stack_kwargs
            )
            self._stack_kwargs["yaml_path"] = yaml_path

    # -------------------------------------------------------------------------
    # Internal: z-crop
    # -------------------------------------------------------------------------

    def _apply_z_crop(self) -> None:
        """Apply z-crop to the stack if geometry.z_crop is set."""
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
            cropped_dielectrics.append({
                **diel,
                "zmin": max(diel["zmin"], z_lo),
                "zmax": min(diel["zmax"], z_hi),
            })

        self.geometry.stack = LayerStack(
            pdk_name=stack.pdk_name,
            units=stack.units,
            layers=cropped,
            materials=stack.materials,
            dielectrics=cropped_dielectrics,
            simulation=stack.simulation,
        )

    # -------------------------------------------------------------------------
    # Internal: translate to legacy config objects
    # -------------------------------------------------------------------------

    def _wavelength_config(self) -> Any:
        """Derive WavelengthConfig from source."""
        from gsim.meep.models.config import WavelengthConfig

        return WavelengthConfig(
            wavelength=self.source.wavelength,
            bandwidth=self.source.bandwidth,
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
        """Translate stopping variant → StoppingConfig."""
        from gsim.meep.models.config import StoppingConfig

        s = self.solver.stopping
        if isinstance(s, FixedTime):
            return StoppingConfig(mode="fixed", max_time=s.max_time)
        if isinstance(s, FieldDecay):
            return StoppingConfig(
                mode="decay",
                max_time=s.max_time,
                threshold=s.threshold,
                decay_component=s.component,
                decay_dt=s.dt,
                decay_monitor_port=s.monitor_port,
            )
        if isinstance(s, DFTDecay):
            return StoppingConfig(
                mode="dft_decay",
                max_time=s.max_time,
                threshold=s.threshold,
                dft_min_run_time=s.min_time,
            )
        raise TypeError(f"Unknown stopping type: {type(s)}")

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
        """Convert materials dict → MaterialProperties overrides for resolve_materials."""
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
    # write_config
    # -------------------------------------------------------------------------

    def write_config(self, output_dir: str | Path) -> Path:
        """Serialize simulation config to output directory.

        Args:
            output_dir: Directory to write layout.gds, sim_config.json, run_meep.py.

        Returns:
            Path to the output directory.

        Raises:
            ValueError: If config is invalid.
        """
        from gsim.meep.materials import resolve_materials
        from gsim.meep.models.config import LayerStackEntry, SimConfig
        from gsim.meep.ports import extract_port_info
        from gsim.meep.script import generate_meep_script
        from gsim.meep.sim import estimate_meep_np

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        validation = self.validate_config()
        if not validation.valid:
            raise ValueError(
                "Invalid configuration:\n" + "\n".join(validation.errors)
            )

        # Resolve stack
        self._ensure_stack()
        if self.geometry.stack is None:
            # Try PDK default
            from gsim.common.stack import get_stack

            self.geometry.stack = get_stack()

        if self.geometry.stack is None:
            raise ValueError("Stack resolution failed.")
        if self.geometry.component is None:
            raise ValueError("No geometry set.")

        # Apply z-crop if requested
        self._apply_z_crop()

        import gdsfactory as gf

        original_component = self.geometry.component.copy()
        stack = self.geometry.stack

        # Build legacy config objects
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

        # Write extended component GDS
        gds_path = output_dir / "layout.gds"
        component.write_gds(gds_path)

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
            dielectric_entries.append({
                "name": diel["name"],
                "zmin": diel["zmin"],
                "zmax": diel["zmax"],
                "material": diel["material"],
            })
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

        # Translate domain.symmetries → SymmetryEntry for legacy config
        from gsim.meep.models.config import SymmetryEntry

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
        z_min = min(e.zmin for e in layer_stack_entries)
        z_max = max(e.zmax for e in layer_stack_entries)
        cell_x = bbox_w + 2 * (margin_xy + dpml)
        cell_y = bbox_h + 2 * (margin_xy + dpml)
        cell_z = (z_max - z_min) + 2 * dpml

        meep_np = estimate_meep_np(
            cell_x, cell_y, cell_z, resolution_cfg.pixels_per_um
        )
        sim_config.meep_np = meep_np
        logger.info(
            "Auto meep_np=%d (cell %.1f x %.1f x %.1f um, res %d)",
            meep_np, cell_x, cell_y, cell_z, resolution_cfg.pixels_per_um,
        )

        # Write JSON config
        config_path = output_dir / "sim_config.json"
        sim_config.to_json(config_path)

        # Write runner script
        script_path = output_dir / "run_meep.py"
        script_content = generate_meep_script(config_filename="sim_config.json")
        script_path.write_text(script_content)

        logger.info("Config written to %s", output_dir)
        return output_dir

    # -------------------------------------------------------------------------
    # run
    # -------------------------------------------------------------------------

    def run(self, output_dir: str | Path, *, verbose: bool = True) -> Any:
        """Run MEEP simulation on the cloud.

        Args:
            output_dir: Directory to write config and download results.
            verbose: Print progress info.

        Returns:
            SParameterResult with parsed S-parameters.
        """
        from gsim.gcloud import run_simulation
        from gsim.meep.models.results import SParameterResult

        output_dir = self.write_config(output_dir)

        results = run_simulation(
            output_dir,
            job_type="meep",
            verbose=verbose,
        )

        csv_path = results.get("s_parameters.csv")
        if csv_path is not None:
            return SParameterResult.from_csv(csv_path)

        logger.warning("No s_parameters.csv found in results")
        return SParameterResult()

    # -------------------------------------------------------------------------
    # Visualization delegation
    # -------------------------------------------------------------------------

    def _to_legacy_sim(self) -> Any:
        """Build a MeepSim equivalent for visualization delegation."""
        from gsim.meep.sim import MeepSim

        legacy = MeepSim()
        legacy.geometry = (
            __import__("gsim.common", fromlist=["Geometry"])
            .Geometry(component=self.geometry.component)
            if self.geometry.component is not None
            else None
        )
        legacy.stack = self.geometry.stack
        legacy.domain_config = self._domain_config()
        legacy.source_config = self._source_config()
        legacy.materials = self._material_overrides()
        legacy._output_dir = None
        legacy._stack_kwargs = self._stack_kwargs.copy()
        return legacy

    def plot_2d(self, **kwargs: Any) -> Any:
        """Plot 2D cross-sections (delegates to MeepSim visualization)."""
        return self._to_legacy_sim().plot_2d(**kwargs)

    def plot_3d(self, **kwargs: Any) -> Any:
        """Plot 3D visualization (delegates to MeepSim visualization)."""
        return self._to_legacy_sim().plot_3d(**kwargs)
