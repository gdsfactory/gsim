"""Declarative Simulation container for MEEP photonic FDTD.

Translates the user-facing declarative API objects into the existing
``SimConfig`` JSON contract consumed by the cloud runner.
"""

from __future__ import annotations

import logging
import warnings
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
    ModeSolver,
    ModeSource,
)

logger = logging.getLogger(__name__)

_AUTO_Z_PADDING = 0.5
_FIBER_FLUX_OFFSET = 0.3


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
        stack: Simulation stack remapped to the component's physical layers.
        gdsfactory_stack: Direct-layer stack used to visualize the component.
    """

    config: Any  # SimConfig
    component: Any  # gdsfactory Component (extended)
    original_component: Any  # gdsfactory Component (original)
    stack: Any  # gsim LayerStack (cropped and remapped)
    gdsfactory_stack: Any  # gdsfactory LayerStack (direct physical layers)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _precompute_cross_section_geometry(
    component: Any,
    stack: Any,
    *,
    port: str | None = None,
    position: tuple[float, float] | None = None,
    x_span: float | None = None,
    y_span: float | None = None,
    z_margin: float | tuple[float, float] = 0.0,
    pml_thickness: float = 0.0,
    resolution: float = 32,
    material_data: dict[str, Any] | None = None,
    background_material: str = "air",  # noqa: ARG001
) -> Any:
    """Pre-compute 2D cross-section cell geometry for cloud serialization.

    Reuses the same GDS-polygon interval logic as the local MEEP cell
    builders but produces a :class:`CrossSectionGeometry` without
    importing MEEP, so it can run on a client that has gdsfactory but
    not MEEP installed.
    """
    from gsim.meep.mode_solver import (
        _layer_has_any_polygon,
        _layer_x_intervals_at_y,
        _layer_y_intervals_at_x,
        _subtract_intervals,
    )
    from gsim.meep.models.config import CrossSectionBlock, CrossSectionGeometry

    if component is None:
        raise ValueError("component is required for cross-section mode")
    if stack is None:
        raise ValueError("stack is required for cross-section mode")

    _use_yz: bool
    _span: float

    if port is not None:
        port_info = None
        for p in component.ports:
            if p.name == port:
                port_info = p
                break
        if port_info is None:
            available = [p.name for p in component.ports]
            raise ValueError(
                f"Port '{port}' not found in component. Available: {available}"
            )
        port_ori = float(getattr(port_info, "orientation", 0))
        _use_yz = port_ori % 180 == 0
        if _use_yz:
            x_cut = float(port_info.center[0])
            if y_span is None:
                raise ValueError("y_span required for YZ cross-section with port")
            _span = y_span
        else:
            y_cut = float(port_info.center[1])
            if x_span is None:
                raise ValueError("x_span required for XZ cross-section with port")
            _span = x_span
    elif position is not None:
        _use_yz = False
        if _use_yz:
            if y_span is None:
                raise ValueError("y_span required for YZ cross-section with position")
            x_cut = float(position[0])
            _span = y_span
        else:
            if x_span is None:
                raise ValueError("x_span required for XZ cross-section with position")
            y_cut = float(position[1])
            _span = x_span
    else:
        raise ValueError("Either port or position must be specified")

    z_min = min(layer.zmin for layer in stack.layers.values())
    z_max = max(layer.zmax for layer in stack.layers.values())
    if isinstance(z_margin, (tuple, list)):
        z_margin_bottom, z_margin_top = z_margin
    else:
        z_margin_bottom = z_margin_top = z_margin
    z_center = (z_min + z_max) / 2.0 + (z_margin_top - z_margin_bottom) / 2.0

    horizontal_span = _span
    if pml_thickness > 0:
        horizontal_span += 2 * pml_thickness
    horizontal_span = round(horizontal_span * resolution) / resolution

    z_span = (z_max - z_min) + z_margin_bottom + z_margin_top
    if pml_thickness > 0:
        z_span += 2 * pml_thickness
    z_span = round(z_span * resolution) / resolution

    blocks: list[Any] = []

    layer_data: list[dict] = []
    for layer in stack.layers.values():
        if layer.material == "air":
            continue
        if material_data and layer.material not in material_data:
            continue
        layer_thickness = layer.zmax - layer.zmin
        if layer_thickness <= 0:
            continue
        if _use_yz:
            intervals = _layer_y_intervals_at_x(component, layer, x_cut)
            if not intervals and not _layer_has_any_polygon(component, layer):
                intervals = [(-horizontal_span / 2, horizontal_span / 2)]
        else:
            intervals = _layer_x_intervals_at_y(component, layer, y_cut)
            if not intervals and not _layer_has_any_polygon(component, layer):
                intervals = [(-horizontal_span / 2, horizontal_span / 2)]
        layer_data.append(
            {
                "z_lo": layer.zmin - z_center,
                "z_hi": layer.zmax - z_center,
                "intervals": intervals,
                "material": layer.material,
            }
        )

    for ld in layer_data:
        below_intervals: list[tuple[float, float]] = []
        for od in layer_data:
            if od is ld:
                continue
            if od["z_hi"] <= ld["z_lo"] + 1e-12:
                below_intervals.extend(od["intervals"])

        for h0, h1 in ld["intervals"]:
            h_center = (h0 + h1) / 2.0
            h_size = h1 - h0
            if h_size <= 0:
                continue
            z_lo = ld["z_lo"]
            z_hi = ld["z_hi"]
            block_z_size = z_hi - z_lo
            if block_z_size > 0:
                block_z_center = (z_lo + z_hi) / 2.0
                blocks.append(
                    CrossSectionBlock(
                        horizontal_center=h_center,
                        horizontal_size=h_size,
                        z_center=block_z_center,
                        z_size=block_z_size,
                        material=ld["material"],
                    )
                )

            bottom_free = _subtract_intervals((h0, h1), below_intervals)
            for bh0, bh1 in bottom_free:
                bh_center = (bh0 + bh1) / 2.0
                bh_size = bh1 - bh0
                if bh_size <= 0:
                    continue
                ext_z_lo = -z_span / 2.0
                ext_z_hi = ld["z_lo"]
                ext_z_size = ext_z_hi - ext_z_lo
                if ext_z_size <= 0:
                    continue
                blocks.append(
                    CrossSectionBlock(
                        horizontal_center=bh_center,
                        horizontal_size=bh_size,
                        z_center=(ext_z_lo + ext_z_hi) / 2.0,
                        z_size=ext_z_size,
                        material=ld["material"],
                    )
                )

    # Add dielectric slabs (uniform horizontal layers)
    for diel in stack.dielectrics:
        if diel["material"] == "air":
            continue
        z_lo = diel["zmin"] - z_center
        z_hi = diel["zmax"] - z_center
        z_size = z_hi - z_lo
        if z_size <= 0:
            continue
        if abs(diel["zmin"] - z_min) < 1e-12:
            z_lo = -z_span / 2.0
        z_size = z_hi - z_lo
        block_z_center = (z_lo + z_hi) / 2.0
        blocks.append(
            CrossSectionBlock(
                horizontal_center=0.0,
                horizontal_size=horizontal_span,
                z_center=block_z_center,
                z_size=z_size,
                material=diel["material"],
            )
        )

    return CrossSectionGeometry(
        plane="yz" if _use_yz else "xz",
        blocks=blocks,
        cell_horizontal_span=horizontal_span,
        cell_z_span=z_span,
        z_center=z_center,
    )


class Simulation(BaseModel):
    """Declarative MEEP FDTD simulation container.

    Assigns typed physics objects, then calls ``write_config()`` to
    produce the JSON + GDS + runner consumed by the cloud engine.

    Example::

        from gsim import meep

        sim = meep.Simulation()
        sim.geometry.component = ybranch
        sim.geometry.stack = stack
        sim.materials = {"si": Material(permittivity=12.0)}
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
    materials: dict[str, float | Material | dict[str, Any]] = Field(
        default_factory=dict
    )
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
    mode_solver: ModeSolver = Field(default_factory=ModeSolver)

    # Private: kwargs captured from geometry.stack when it's a string/path
    _stack_kwargs: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Private: guards against double vertical-cropping when build_config()
    # is called more than once (e.g. plot_2d then run) on the same sim.
    _z_cropped: bool = PrivateAttr(default=False)
    _z_crop_source_stack: Any | None = PrivateAttr(default=None)
    _last_cropped_stack: Any | None = PrivateAttr(default=None)
    _resolved_z_bounds: tuple[float, float] | None = PrivateAttr(default=None)

    # Extra hints forwarded into the config JSON (not part of the schema).
    _hints: dict[str, Any] = PrivateAttr(default_factory=dict)

    # Cloud job state (set by upload/run)
    _job_id: str | None = PrivateAttr(default=None)
    _config_dir: Path | None = PrivateAttr(default=None)
    _input_hash: str | None = PrivateAttr(default=None)

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @field_validator("materials", mode="before")
    @classmethod
    def _normalize_materials(
        cls,
        v: dict[str, float | Material | dict],
    ) -> dict[str, Material]:
        """Accept float/int shorthand: ``{"si": 12.0}`` -> ``Material(permittivity=12.0)``."""  # noqa: E501
        out: dict[str, Material] = {}
        for name, val in v.items():
            if isinstance(val, Material):
                out[name] = val
            elif isinstance(val, (int, float)):
                out[name] = Material(permittivity=float(val))
            elif isinstance(val, dict):
                out[name] = Material(**val)
            else:
                raise TypeError(
                    f"Material '{name}' must be a Material, number, or dict, "
                    f"got {type(val).__name__}."
                )
        return out

    # -------------------------------------------------------------------------
    # Resolved materials helper
    # -------------------------------------------------------------------------

    def _resolved_materials(self) -> dict[str, Material]:
        """Return materials dict with all values normalized to Material."""
        return dict(self.materials)  # ty: ignore[invalid-return-type]

    # -------------------------------------------------------------------------
    # Fiber source helper
    # -------------------------------------------------------------------------

    def source_fiber(self, **kwargs: Any) -> FiberSource:
        """Configure a tilted Gaussian-beam fiber source (XZ 2D only).

        Replaces any previous fiber source. Requires ``solver.mode='2d'``
        (and ``solver.y_cut`` set for the XZ plane at ``build_config`` time).

        The ``waist`` kwarg is the 1/e² intensity *radius* (= MFD / 2),
        matching MEEP's ``beam_w0``. Typical SMF-28 values:
        ``waist ~= 4.6 um`` at 1310 nm (MFD ~= 9.2 um) and
        ``waist ~= 5.2 um`` at 1550 nm (MFD ~= 10.4 um).

        Args:
            **kwargs: Fields forwarded to :class:`FiberSource`.

        Returns:
            The newly created :class:`FiberSource` instance.
        """
        if self.solver.resolved_is_3d():
            raise ValueError(
                "fiber source requires solver.mode='2d' (with y_cut set for "
                "the XZ plane) — currently mode='3d'"
            )
        fiber_source = FiberSource(**kwargs)
        self.fiber_source = fiber_source
        return fiber_source

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
            if not ports and self.solver.resolved_plane() != "xz":
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

    def _stack_material_extent(
        self, stack: Any | None = None
    ) -> tuple[float, float] | None:
        """Return (zmin, zmax) spanning all non-air layers and dielectrics.

        This is the reference used by auto z-crop and by the fiber-aware
        margin expansion: it preserves the full fabricated stack (BOX,
        core, cladding, passive, ...) and trims only the synthetic air
        padding added by the extractor.
        """
        stack = stack if stack is not None else self.geometry.stack
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

    def _resolve_xy_background_material(self, component: Any, stack: Any) -> str:
        """Resolve the blanket medium represented by a collapsed XY Z cut."""
        if self.solver.resolved_plane() != "xy":
            return "air"

        cut = self.solver.resolved_cut()
        if isinstance(cut, int | float):
            cut_z = float(cut)
        else:
            from gsim.meep.ports import _find_highest_n_layer_in_component

            reference_layer, _ = _find_highest_n_layer_in_component(component, stack)
            if reference_layer is None:
                logger.warning(
                    "Could not resolve a drawn layer for XY 2D z_cut='auto'; "
                    "using air as the background material."
                )
                return "air"
            cut_z = (reference_layer.zmin + reference_layer.zmax) / 2.0

        containing_dielectrics = [
            dielectric
            for dielectric in stack.dielectrics
            if dielectric.get("material") != "air"
            and float(dielectric["zmin"]) <= cut_z <= float(dielectric["zmax"])
        ]
        if not containing_dielectrics:
            return "air"

        # At a shared interface, the region starting at the cut takes priority.
        background = max(
            containing_dielectrics,
            key=lambda dielectric: float(dielectric["zmin"]),
        )
        return str(background["material"])

    def _resolve_mode_solver_z_window(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Resolve mode-solver bounds and equivalent internal margins."""
        extent = self._stack_material_extent()
        if extent is None:
            raise ValueError("Could not resolve a non-air Z extent for mode solving.")
        stack_low, stack_high = extent

        if self.domain.z_bounds != "auto":
            z_low, z_high = self.domain.z_bounds
            if z_low > stack_low or z_high < stack_high:
                raise ValueError(
                    "Explicit domain.z_bounds must contain the modeled stack for "
                    f"mode solving: requested ({z_low}, {z_high}), stack spans "
                    f"({stack_low}, {stack_high})."
                )
            margins = (stack_low - z_low, z_high - stack_high)
            self._resolved_z_bounds = (z_low, z_high)
            return (z_low, z_high), margins

        legacy_fields = self.domain.legacy_z_fields_used()
        if legacy_fields:
            margin_low, margin_high = self.domain.resolved_margin_z()
        else:
            margin_low = margin_high = _AUTO_Z_PADDING
        bounds = (stack_low - margin_low, stack_high + margin_high)
        self._resolved_z_bounds = bounds

        if legacy_fields:
            field_names = " and ".join(f"domain.{name}" for name in legacy_fields)
            warnings.warn(
                f"Use of {field_names} is deprecated and will be removed in a "
                f"future release; replace it with domain.z_bounds={bounds!r}.",
                FutureWarning,
                stacklevel=3,
            )
        return bounds, (margin_low, margin_high)

    def _resolve_active_z_bounds(
        self, component: Any, stack: Any | None = None
    ) -> tuple[tuple[float, float], str, float | None, bool]:
        """Resolve the public Z-domain setting to one concrete inner window.

        Returns:
            ``(bounds, reference_name, reference_n, is_auto)``. Explicit bounds
            are returned unchanged. ``"auto"`` fits the drawn optical layer
            with deterministic padding and may add fiber-source headroom.
        """
        if self.domain.z_bounds != "auto":
            z_low, z_high = self.domain.z_bounds
            self._validate_fiber_z_bounds(z_low, z_high)
            return (z_low, z_high), "explicit", None, False

        ref_zmin, ref_zmax, ref_name, ref_n, is_auto_ref = self._resolve_z_ref_extent(
            component=component,
            stack=stack,
        )
        legacy_fields = self.domain.legacy_z_fields_used()
        if legacy_fields:
            margin_low, margin_high = self.domain.resolved_margin_z()
        else:
            margin_low = margin_high = _AUTO_Z_PADDING

        z_low = ref_zmin - margin_low
        z_high = ref_zmax + margin_high

        if self.fiber_source is not None:
            fiber_headroom = max(self.fiber_source.waist / 2.0, _AUTO_Z_PADDING)
            z_high = max(z_high, self.fiber_source.z + fiber_headroom)

        bounds = (float(z_low), float(z_high))
        if legacy_fields:
            field_names = " and ".join(f"domain.{name}" for name in legacy_fields)
            warnings.warn(
                f"Use of {field_names} is deprecated and will be removed in a "
                f"future release; replace it with domain.z_bounds={bounds!r}.",
                FutureWarning,
                stacklevel=3,
            )

        return bounds, ref_name, ref_n, is_auto_ref

    def _validate_fiber_z_bounds(self, z_low: float, z_high: float) -> None:
        """Require an explicit window to contain all fiber-source headroom."""
        if self.fiber_source is None:
            return
        source = self.fiber_source
        required_low = source.z - _FIBER_FLUX_OFFSET
        required_high = source.z + max(source.waist / 2.0, _AUTO_Z_PADDING)
        if z_low <= required_low and z_high >= required_high:
            return
        raise ValueError(
            "Explicit domain.z_bounds does not contain the fiber source and "
            f"monitor headroom: requested ({z_low}, {z_high}), requires at "
            f"least ({required_low}, {required_high}). Expand z_bounds or use "
            "z_bounds='auto'."
        )

    def _prepare_stack_for_z_crop(self) -> None:
        """Restore a fresh stack before resolving and applying each Z crop."""
        stack = self.geometry.stack
        if stack is None:
            raise ValueError("No stack configured for z-crop.")

        if (
            self._z_cropped
            and stack is self._last_cropped_stack
            and self._z_crop_source_stack is not None
        ):
            self.geometry.stack = self._z_crop_source_stack.model_copy(deep=True)
        else:
            self._z_crop_source_stack = stack.model_copy(deep=True)

        self._z_cropped = False
        self._last_cropped_stack = None

    def _resolve_z_ref_extent(
        self,
        component: Any | None = None,
        stack: Any | None = None,
    ) -> tuple[float, float, str, float | None, bool]:
        """Resolve ``domain.z_ref`` to a vertical reference window.

        Single source of truth for both the fiber-margin expansion and the
        z-crop. Interprets ``domain.z_ref``:

        - ``None`` -> auto: highest-n layer the component actually draws
          (the photonic core). Falls back to the full stack extent.
        - ``"stack"`` -> the full non-air material extent (BOX..cladding).
        - ``"<name>"`` -> that specific layer's z-extent.

        Returns:
            ``(ref_zmin, ref_zmax, ref_name, ref_n, is_auto)`` where ``ref_n``
            is the reference layer's refractive index (or None) and ``is_auto``
            is True when ``z_ref`` was left as the default.
        """
        from gsim.meep.ports import _find_highest_n_layer_in_component

        stack = stack if stack is not None else self.geometry.stack
        if stack is None:
            raise ValueError("No stack configured for z-crop.")

        z_ref = self.domain.z_ref

        if z_ref == "stack":
            extent = self._stack_material_extent(stack)
            if extent is None:
                raise ValueError(
                    "Could not detect any non-air layers/dielectrics for "
                    "z_ref='stack'. Set domain.z_ref to an explicit layer name."
                )
            return extent[0], extent[1], "stack", None, False

        if z_ref is None:
            layer, n = _find_highest_n_layer_in_component(
                component if component is not None else self.geometry.component,
                stack,
            )
            if layer is None:
                extent = self._stack_material_extent(stack)
                if extent is None:
                    raise ValueError(
                        "Could not detect any drawn optical layer or non-air "
                        "stack for auto z_ref. Set domain.z_ref explicitly."
                    )
                return extent[0], extent[1], "stack", None, True
            return layer.zmin, layer.zmax, layer.name, n, True

        # Named layer
        if z_ref not in stack.layers:
            raise ValueError(
                f"Layer '{z_ref}' not found. Available: {list(stack.layers.keys())}"
            )
        ref = stack.layers[z_ref]
        return ref.zmin, ref.zmax, z_ref, None, False

    # -------------------------------------------------------------------------
    # Internal: z-crop
    # -------------------------------------------------------------------------

    def _apply_z_crop(
        self,
        z_lo: float,
        z_hi: float,
        ref_name: str,
        ref_n: float | None,
        is_auto: bool,
    ) -> None:
        """Crop the stack to authoritative PML-inner Z bounds.

        Trims/removes layers and dielectrics outside the exact interval.

        Args:
            z_lo: Lower PML-inner Z bound (um).
            z_hi: Upper PML-inner Z bound (um).
            ref_name: Name of the reference ('stack' or a layer name).
            ref_n: Refractive index of the reference layer (or None).
            is_auto: Whether the reference was auto-detected.
        """
        from gsim.common.stack.extractor import Layer, LayerStack

        stack = self.geometry.stack
        if stack is None:
            raise ValueError("No stack configured for z-crop.")

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

        if not cropped and not cropped_dielectrics:
            raise ValueError(
                f"domain.z_bounds=({z_lo}, {z_hi}) does not intersect any "
                "layer or dielectric in the resolved stack."
            )

        self.geometry.stack = LayerStack(
            pdk_name=stack.pdk_name,
            units=stack.units,
            layers=cropped,
            materials=stack.materials,
            dielectrics=cropped_dielectrics,
            simulation=stack.simulation,
        )
        if is_auto:
            n_str = f"n={ref_n:.2f}, " if ref_n is not None else ""
            logger.info(
                "z-crop reference auto-detected: %r (%swindow z=[%.4g, %.4g])",
                ref_name,
                n_str,
                z_lo,
                z_hi,
            )
        logger.info(
            "z-crop applied (ref=%r, z=[%.4g, %.4g]); trimmed %d layer(s): %s; "
            "removed %d layer(s) fully outside crop: %s",
            ref_name,
            z_lo,
            z_hi,
            len(trimmed_names),
            trimmed_names,
            len(removed_names),
            removed_names,
        )
        # Guard against re-cropping an already-cropped stack on repeat
        # build_config() calls (e.g. plot_2d then run).
        self._z_cropped = True
        self._last_cropped_stack = self.geometry.stack

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
        """Translate ModeSource -> SourceConfig."""
        from gsim.meep.models.config import SourceConfig

        return SourceConfig(
            bandwidth=None,
            port=self.source.port,
        )

    def _stopping_config(self) -> Any:
        """Translate FDTD stopping fields -> StoppingConfig."""
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

    def _domain_config(self, z_bounds: tuple[float, float] | None = None) -> Any:
        """Translate Domain to config with optional resolved Z bounds."""
        from gsim.meep.models.config import DomainConfig

        mx = self.domain.resolved_margin_x()
        my = self.domain.resolved_margin_y()
        mz = self.domain.resolved_margin_z()
        return DomainConfig(
            z_bounds=z_bounds,
            dpml=self.domain.pml,
            extend_into_pml=self.domain.extend_into_pml,
            margin_x_low=mx[0],
            margin_x_high=mx[1],
            margin_y_low=my[0],
            margin_y_high=my[1],
            margin_z_low=mz[0],
            margin_z_high=mz[1],
            port_margin=self.domain.port_margin,
            extend_ports=self.domain.extend_ports,
            source_port_offset=self.domain.source_port_offset,
            distance_source_to_monitors=self.domain.distance_source_to_monitors,
        )

    def _resolution_config(self) -> Any:
        """Translate FDTD.resolution -> ResolutionConfig."""
        from gsim.meep.models.config import ResolutionConfig

        return ResolutionConfig(pixels_per_um=self.solver.resolution)

    def _accuracy_config(self) -> Any:
        """Translate FDTD accuracy fields -> AccuracyConfig."""
        from gsim.meep.models.config import AccuracyConfig

        return AccuracyConfig(
            eps_averaging=self.solver.subpixel,
            subpixel_maxeval=self.solver.subpixel_maxeval,
            subpixel_tol=self.solver.subpixel_tol,
            simplify_tol=self.solver.simplify_tol,
        )

    def _diagnostics_config(self) -> Any:
        """Translate FDTD diagnostic fields -> DiagnosticsConfig."""
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
                permittivity=val.permittivity,
                loss_tangent=val.loss_tangent,
            )
        return overrides

    def _resolve_stack_and_materials(
        self, *, wavelength: float
    ) -> tuple[Any, dict[str, Any]]:
        """Resolve layer stack and material data for mode solving.

        Lighter than :meth:`build_config` — no port extension, no domain
        config, no FDTD source/monitor setup. Resolves the layer stack,
        applies z-crop if configured, and resolves material optical
        properties at the given wavelength via the three-tier pipeline
        (user override > PDK overlay > built-in database).

        Args:
            wavelength: Free-space wavelength in µm for material evaluation.

        Returns:
            ``(stack, materials_dict)`` where ``stack`` is the resolved
            ``LayerStack`` and ``materials_dict`` maps material name to
            ``MaterialData``.
        """
        from gsim.meep.materials import resolve_materials

        self._ensure_stack()
        stack = self.geometry.stack
        if stack is None:
            raise ValueError("Stack resolution failed.")

        # Collect only populated physical layers. PDK stacks commonly include
        # metal levels that are absent from a passive photonic component.
        used_materials: set[str] = set()
        component = self.geometry.component
        if component is not None:
            from gsim.meep.physical_layers import materialize_physical_layers

            physical_export = materialize_physical_layers(component, stack)
            populated_layers = {
                tuple(layer) for layer in physical_export.component.layers
            }
            for layer in physical_export.stack.layers.values():
                if tuple(layer.gds_layer) in populated_layers:
                    used_materials.add(layer.material)
        elif not stack.dielectrics:
            # Slab-mode configs serialize declared dielectrics when available;
            # only stacks without them fall back to every physical layer.
            used_materials.update(layer.material for layer in stack.layers.values())
        for diel in stack.dielectrics:
            used_materials.add(diel["material"])

        material_data = resolve_materials(
            used_materials,
            overrides=self._material_overrides(),
            wavelength_um=wavelength,
        )

        return stack, material_data

    # -------------------------------------------------------------------------
    # solve_modes — eigenmode solving from mode_solver configuration
    # -------------------------------------------------------------------------

    def solve_modes(
        self,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        check_cache: bool = False,
    ) -> Any:
        """Solve eigenmodes on the cloud from ``self.mode_solver`` configuration.

        Writes the mode-solver config to a temporary directory, uploads
        it to the cloud, waits for the job to finish, and returns a
        :class:`ModeSweepResult` with reconstructed field arrays.

        For local execution (requires MEEP installed), use
        :meth:`solve_modes_local`.

        Args:
            verbose: ``"quiet"`` no output, ``"status"`` status line,
                ``"full"`` stream solver logs.
            check_cache: If ``True``, look for a completed cloud job with
                byte-identical inputs and reuse its results instead of
                submitting. A lookup failure degrades to a normal submit.

        Returns:
            :class:`ModeSweepResult` wrapping all solved :class:`ModeResult`
            objects.
        """
        import tempfile

        from gsim import gcloud

        tmp = Path(tempfile.mkdtemp(prefix="meep_mode_solver_"))
        try:
            self.write_mode_solver_config(tmp)
            input_hash = None
            if check_cache:
                input_hash, cached_job_id = gcloud.check_cache_for_dir(tmp, "meep")
                if cached_job_id is not None:
                    if verbose != "quiet":
                        print(  # noqa: T201
                            f"Cache hit: reusing job {cached_job_id}"
                        )
                    result = gcloud.wait_for_results(cached_job_id, verbose=verbose)
                    self._enrich_mode_results(result)
                    return result

            job_id = gcloud.upload(tmp, "meep", verbose=False, input_hash=input_hash)
            gcloud.start(job_id, verbose=verbose != "quiet")
            result = gcloud.wait_for_results(job_id, verbose=verbose)
            self._enrich_mode_results(result)
            return result
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def _enrich_mode_results(self, sweep: Any) -> None:
        """Attach stack, component, port, and domain context to cloud-parsed results."""
        from gsim.meep.results import ModeSweepResult

        if not isinstance(sweep, ModeSweepResult):
            return

        stack = self.geometry.stack
        component = self.geometry.component
        domain_cfg = self._domain_config(z_bounds=self._resolved_z_bounds)
        port = self.mode_solver.port
        position = self.mode_solver.position
        port_or_pos = port if port is not None else position

        for r in sweep.results:
            if r.stack is None:
                r.stack = stack
            if r.component is None:
                r.component = component
            if r.port_or_position is None:
                r.port_or_position = port_or_pos
            r.domain_config = domain_cfg

    def solve_modes_local(self) -> Any:
        """Solve eigenmodes locally from ``self.mode_solver`` configuration.

        Requires a local MEEP installation.  Reads wavelengths, band count,
        and geometry context from the :class:`ModeSolver` model and dispatches
        to the appropriate low-level solvers with shared-cell optimizations
        where applicable.

        Returns:
            :class:`ModeSweepResult` wrapping all solved :class:`ModeResult`
            objects.

        Raises:
            ValueError: If ``wavelengths`` is empty, or if cross-section
                mode is requested without a component, port, or position.
        """
        from gsim.meep.mode_solver import (
            mode_x_grid,
            mode_y_grid,
            mode_z_grid,
            solve_cross_section_mode,
            solve_slab_mode,
            solve_slab_modes,
            solve_slab_wavelength_sweep,
        )
        from gsim.meep.models.results import ModeResult
        from gsim.meep.results import ModeSweepResult

        ms = self.mode_solver

        if not ms.wavelengths:
            raise ValueError("mode_solver.wavelengths must not be empty")

        resolution = self.solver.resolution
        pml_thickness = self.domain.pml

        component = self.geometry.component
        where = ms.where

        if where == "auto":
            has_port_or_pos = ms.port is not None or ms.position is not None
            if component is not None and has_port_or_pos:
                where_effective = "cross_section"
            else:
                where_effective = "slab"
        else:
            where_effective = where

        if where_effective == "cross_section":
            if component is None:
                raise ValueError(
                    "cross_section mode requires a component — "
                    "set sim.geometry.component first."
                )
            if ms.port is None and ms.position is None:
                raise ValueError(
                    "cross_section mode requires port or position — "
                    "set mode_solver.port or mode_solver.position."
                )

        if ms.band is not None:
            band_nums = [ms.band]
        else:
            band_nums = list(range(1, ms.num_bands + 1))

        first_wavelength = ms.wavelengths[0]
        _stack, _materials = self._resolve_stack_and_materials(
            wavelength=first_wavelength
        )
        stack = self.geometry.stack
        if stack is None:
            raise ValueError("Stack resolution failed.")
        mode_z_bounds, z_margin = self._resolve_mode_solver_z_window()

        background_material = ms.background_material

        n_field_x = ms.n_field_x
        n_field_y = ms.n_field_y
        n_field_z = ms.n_field_z
        field_x_grid = (
            mode_x_grid(n_field_x, ms.x_span or 0.0, pml_thickness)
            if n_field_x > 0 and ms.x_span is not None
            else None
        )
        field_y_grid = (
            mode_y_grid(n_field_y, ms.y_span or 0.0, pml_thickness)
            if n_field_y > 0 and ms.y_span is not None
            else None
        )
        field_z_grid = (
            mode_z_grid(stack, n_field_z, z_margin, pml_thickness)
            if n_field_z > 0
            else None
        )

        results: list[ModeResult] = []

        if where_effective == "slab":
            if len(ms.wavelengths) > 1 and len(band_nums) == 1:
                sweep_results = solve_slab_wavelength_sweep(
                    stack=stack,
                    wavelengths=ms.wavelengths,
                    band_num=band_nums[0],
                    parity=ms.parity,
                    resolution=resolution,
                    z_margin=z_margin,
                    pml_thickness=pml_thickness,
                    eigensolver_tol=ms.eigensolver_tol,
                    field_z_grid=field_z_grid,
                    background_material=background_material,
                )
                results.extend(sweep_results.values())
            elif len(ms.wavelengths) == 1 and len(band_nums) > 1:
                band_results = solve_slab_modes(
                    stack=stack,
                    wavelength=ms.wavelengths[0],
                    band_nums=band_nums,
                    parity=ms.parity,
                    resolution=resolution,
                    z_margin=z_margin,
                    pml_thickness=pml_thickness,
                    eigensolver_tol=ms.eigensolver_tol,
                    field_z_grid=field_z_grid,
                    background_material=background_material,
                )
                results.extend(band_results.values())
            else:
                for wl in ms.wavelengths:
                    for bn in band_nums:
                        mode_result = solve_slab_mode(
                            stack=stack,
                            wavelength=wl,
                            band_num=bn,
                            parity=ms.parity,
                            resolution=resolution,
                            z_margin=z_margin,
                            pml_thickness=pml_thickness,
                            eigensolver_tol=ms.eigensolver_tol,
                            field_z_grid=field_z_grid,
                            background_material=background_material,
                        )
                        results.append(mode_result)
        else:
            for wl in ms.wavelengths:
                for bn in band_nums:
                    mode_result = solve_cross_section_mode(
                        component=component,
                        stack=stack,
                        port=ms.port,
                        position=ms.position,
                        x_span=ms.x_span,
                        y_span=ms.y_span,
                        wavelength=wl,
                        band_num=bn,
                        parity=ms.parity,
                        resolution=resolution,
                        z_margin=z_margin,
                        pml_thickness=pml_thickness,
                        eigensolver_tol=ms.eigensolver_tol,
                        field_x_grid=field_x_grid,
                        field_y_grid=field_y_grid,
                        field_z_grid=field_z_grid,
                        background_material=background_material,
                    )
                    results.append(mode_result)

        domain_cfg = self._domain_config(z_bounds=mode_z_bounds)
        for r in results:
            r.domain_config = domain_cfg

        return ModeSweepResult(results)

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

        is_3d = self.solver.resolved_is_3d()
        plane = self.solver.resolved_plane()

        # Resolve stack
        self._ensure_stack()
        if self.geometry.stack is None:
            raise ValueError("Stack resolution failed.")
        if self.geometry.component is None:
            raise ValueError("No geometry set.")

        from gsim.meep.physical_layers import materialize_physical_layers

        # Keep the fabrication-mask component authoritative until after port
        # extension. Physical layers are evaluated onto unique simulation-only
        # tuples so two LayerLevels can never alias through a shared GDS target.
        original_component = self.geometry.component

        # Resolve one authoritative PML-inner Z interval for every simulation
        # with an active Z axis. XY 2D collapses Z and therefore rejects an
        # explicit interval.
        resolved_z_bounds: tuple[float, float] | None = None
        if is_3d or plane == "xz":
            self._prepare_stack_for_z_crop()
            if self.geometry.stack is None:  # pragma: no cover - guarded above
                raise ValueError("Stack resolution failed.")
            reference_export = materialize_physical_layers(
                original_component,
                self.geometry.stack,
            )
            (
                resolved_z_bounds,
                ref_name,
                ref_n,
                is_auto,
            ) = self._resolve_active_z_bounds(
                reference_export.component,
                reference_export.stack,
            )
            self._resolved_z_bounds = resolved_z_bounds
            self._apply_z_crop(
                resolved_z_bounds[0],
                resolved_z_bounds[1],
                ref_name,
                ref_n,
                is_auto,
            )
        else:
            self._resolved_z_bounds = None
            if self.domain.z_bounds != "auto":
                raise ValueError(
                    "Explicit domain.z_bounds requires an active Z axis (3D or "
                    "XZ 2D). Use solver.z_cut to choose an XY slice."
                )

        import gdsfactory as gf

        # Resolve the XZ cut Y-coordinate ('auto'/None -> bbox center).
        cut = self.solver.resolved_cut()
        if plane == "xz":
            if cut is None or cut == "auto":
                bbox = original_component.dbbox()
                y_cut: float | None = (bbox.bottom + bbox.top) / 2.0
            else:
                y_cut = float(cut)
        else:
            y_cut = None

        # Build config objects
        domain_cfg = self._domain_config(z_bounds=resolved_z_bounds)
        wl_cfg = self._wavelength_config()
        source_cfg = self._source_config()
        stopping_cfg = self._stopping_config()
        resolution_cfg = self._resolution_config()
        accuracy_cfg = self._accuracy_config()
        diagnostics_cfg = self._diagnostics_config()

        # Compute port extension length
        extend_length = domain_cfg.extend_ports
        if extend_length == 0.0:
            extend_length = (
                max(
                    domain_cfg.margin_x_low,
                    domain_cfg.margin_x_high,
                    domain_cfg.margin_y_low,
                    domain_cfg.margin_y_high,
                )
                + domain_cfg.dpml
            )

        # Extend source-mask waveguide ports before evaluating physical layers.
        # Extending an already-materialized component would draw onto the source
        # tuple and leave the remapped physical core unextended.
        original_bbox: list[float] | None = None
        if extend_length > 0:
            bbox = original_component.dbbox()
            original_bbox = [bbox.left, bbox.bottom, bbox.right, bbox.top]
            extended_source_component = gf.components.extend_ports(
                original_component, length=extend_length
            )
        else:
            extended_source_component = original_component

        if self.geometry.stack is None:  # pragma: no cover - guarded above
            raise ValueError("Stack resolution failed.")
        physical_export = materialize_physical_layers(
            extended_source_component,
            self.geometry.stack,
        )
        component = physical_export.component
        from gsim.meep.pml import extend_dielectrics_into_pml

        stack = extend_dielectrics_into_pml(
            physical_export.stack,
            domain_cfg,
        )
        background_material = self._resolve_xy_background_material(component, stack)

        # Build layer stack entries
        layer_stack_entries = []
        used_materials: set[str] = set()
        populated_layers = {tuple(layer) for layer in component.layers}
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
            if tuple(layer.gds_layer) in populated_layers:
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
            if self.solver.resolved_is_3d():
                raise ValueError(
                    "fiber source requires solver.mode='2d' (with y_cut set)"
                )
            if plane != "xz":
                raise ValueError("fiber source requires solver.y_cut set (XZ plane)")

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

        # Resolve materials. Explicit per-simulation constants take priority;
        # otherwise project-first MaterialCards are authoritative.
        active_source = (
            self.fiber_source if self.fiber_source is not None else self.source
        )
        from gsim.meep.materials import resolve_fdtd_materials

        material_data = resolve_fdtd_materials(
            used_materials,
            overrides=self._material_overrides(),
            wavelength_um=active_source.wavelength,
            wavelength_span_um=active_source.wavelength_span,
            resolution=resolution_cfg.pixels_per_um,
        )

        fwidth = source_cfg.compute_fwidth(wl_cfg.fcen, wl_cfg.df)
        source_for_config = source_cfg.model_copy(update={"fwidth": fwidth})

        # Translate domain.symmetries -> SymmetryEntry for config
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

        # Compute meep verbosity from `meep.native` logger level, with
        # `run(verbose="full")` override.
        _run_verbose = getattr(self, "_run_verbose", None)
        if _run_verbose == "full":
            meep_verbosity = 2
        else:
            _level = logging.getLogger("meep.native").getEffectiveLevel()
            if _level == logging.NOTSET or _level >= logging.WARNING:
                meep_verbosity = 0
            elif _level >= logging.INFO:
                meep_verbosity = 1
            else:
                meep_verbosity = 2

        # Build SimConfig
        sim_config = SimConfig(
            is_3d=is_3d,
            # Internal config plane is always concrete; 3D ignores it, so
            # default None -> "xy" to keep the emitted JSON schema unchanged.
            plane=plane or "xy",
            y_cut=y_cut,
            fiber_source=fiber_source_cfg,
            gds_filename="layout.gds",
            component_bbox=original_bbox,
            layer_stack=layer_stack_entries,
            dielectrics=dielectric_entries,
            ports=port_infos,
            monitor_z_span=monitor_z_span,
            materials=material_data,
            background_material=background_material,
            wavelength=wl_cfg,
            source=source_for_config,
            stopping=stopping_cfg,
            resolution=resolution_cfg,
            domain=domain_cfg,
            accuracy=accuracy_cfg,
            diagnostics=diagnostics_cfg,
            verbose_interval=diagnostics_cfg.verbose_interval,
            meep_verbosity=meep_verbosity,
            symmetries=symmetry_entries,
        )
        # Forward any private hints into the config
        if self._hints:
            sim_config._hints.update(self._hints)  # noqa: SLF001

        return BuildResult(
            config=sim_config,
            component=component,
            original_component=original_component,
            stack=stack,
            gdsfactory_stack=physical_export.gdsfactory_stack,
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
        import klayout.db as kdb

        from gsim.meep.script import generate_meep_script

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = self.build_config()

        # The solver only consumes polygons, so write a canonical flattened
        # layout without metadata, hierarchy names, or timestamps. Those GDS
        # details do not affect the simulation but would change the cache key.
        gds_path = output_dir / "layout.gds"
        save_options = kdb.SaveLayoutOptions()
        save_options.gds2_write_timestamps = False
        result.component.write_gds(
            gds_path,
            save_options=save_options,
            with_metadata=False,
        )

        canonical_layout = kdb.Layout()
        canonical_layout.read(str(gds_path))
        top_cells = canonical_layout.top_cells()
        if len(top_cells) != 1:  # pragma: no cover - write_gds emits one top cell
            raise RuntimeError(f"Expected one top GDS cell, found {len(top_cells)}")
        top_cell = top_cells[0]
        top_cell.flatten(True)
        top_cell.name = "layout"
        save_options.select_this_cell(top_cell.cell_index())
        save_options.gds2_write_cell_properties = False
        save_options.gds2_write_file_properties = False
        canonical_layout.write(str(gds_path), save_options)

        # Write JSON config
        result.config.to_json(output_dir / "sim_config.json")

        # Write runner script
        script_path = output_dir / "run_meep.py"
        script_content = generate_meep_script(config_filename="sim_config.json")
        script_path.write_text(script_content, encoding="utf-8")

        logger.info("Config written to %s", output_dir)
        return output_dir

    def write_mode_solver_config(self, output_dir: str | Path) -> Path:
        """Serialize mode solver config for cloud eigenmode solving.

        For slab modes (1D) the runner builds the cell from
        ``layer_stack`` alone.  For cross-section modes the geometry is
        pre-computed client-side into ``cross_section_geometry`` so the
        runner does not need gdsfactory / KLayout.

        Args:
            output_dir: Directory to write ``mode_solver_config.json``
                and ``run_meep.py``.

        Returns:
            Path to the output directory.

        Raises:
            ValueError: If ``mode_solver`` has no wavelengths, stack
                resolution fails, or cross-section prerequisites are
                missing (component, port, etc.).
        """
        from gsim.meep.models.config import (
            CrossSectionGeometry,
            DielectricEntry,
            ModeSolverConfig,
        )
        from gsim.meep.script import generate_meep_mode_solver_script

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        ms = self.mode_solver
        if not ms.wavelengths:
            raise ValueError("mode_solver.wavelengths must not be empty")

        bands = [ms.band] if ms.band is not None else list(range(1, ms.num_bands + 1))

        resolution = self.solver.resolution
        pml_thickness = self.domain.pml

        first_wavelength = ms.wavelengths[0]
        _stack, material_data = self._resolve_stack_and_materials(
            wavelength=first_wavelength
        )
        stack = self.geometry.stack
        if stack is None:
            raise ValueError(
                "Stack resolution failed — set a geometry with stack first"
            )
        _mode_z_bounds, z_margin = self._resolve_mode_solver_z_window()

        if stack.dielectrics:
            dielectrics = [
                DielectricEntry(
                    name=diel["name"],
                    zmin=diel["zmin"],
                    zmax=diel["zmax"],
                    material=diel["material"],
                )
                for diel in stack.dielectrics
            ]
        else:
            dielectrics = [
                DielectricEntry(
                    name=layer.name,
                    zmin=layer.zmin,
                    zmax=layer.zmax,
                    material=layer.material,
                )
                for layer in stack.layers.values()
            ]

        component = self.geometry.component
        where = ms.where
        if where == "auto":
            has_port_or_pos = ms.port is not None or ms.position is not None
            where_effective = (
                "cross_section" if component is not None and has_port_or_pos else "slab"
            )
        else:
            where_effective = where

        cross_section_geometry: CrossSectionGeometry | None = None
        n_field_x = ms.n_field_x
        n_field_y = ms.n_field_y

        if where_effective == "cross_section":
            cross_section_geometry = _precompute_cross_section_geometry(
                component=component,
                stack=stack,
                port=ms.port,
                position=ms.position,
                x_span=ms.x_span,
                y_span=ms.y_span,
                z_margin=z_margin,
                pml_thickness=pml_thickness,
                resolution=resolution,
                material_data=material_data,
                background_material=ms.background_material,
            )

        config = ModeSolverConfig(
            wavelengths=ms.wavelengths,
            bands=bands,
            parity=ms.parity,
            resolution=resolution,
            pml_thickness=pml_thickness,
            z_margin=z_margin,
            background_material=ms.background_material,
            eigensolver_tol=ms.eigensolver_tol,
            n_field_z=ms.n_field_z,
            layer_stack=dielectrics,
            materials=material_data,
            cross_section_geometry=cross_section_geometry,
            n_field_x=n_field_x,
            n_field_y=n_field_y,
        )

        config.to_json(output_dir / "mode_solver_config.json")

        script_path = output_dir / "run_meep.py"
        script_path.write_text(generate_meep_mode_solver_script(), encoding="utf-8")

        logger.info("Mode solver config written to %s", output_dir)
        return output_dir

    # -------------------------------------------------------------------------
    # Cloud: fine-grained control
    # -------------------------------------------------------------------------

    def _prepare_upload_dir(self) -> Path:
        """Write the config files to a fresh temp directory for upload.

        Returns:
            Path to the temp directory, also recorded on ``_config_dir``.
        """
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="meep_"))
        self.write_config(tmp)
        self._config_dir = tmp
        return tmp

    def upload(self, *, verbose: bool = True) -> str:
        """Write config and upload to the cloud. Does NOT start execution.

        Args:
            verbose: Print progress messages.

        Returns:
            ``job_id`` string for use with :meth:`start`, :meth:`get_status`,
            or :func:`gsim.wait_for_results`.
        """
        from gsim import gcloud
        from gsim.hashing import compute_input_hash

        tmp = self._prepare_upload_dir()
        self._input_hash = compute_input_hash(tmp, "meep")
        self._job_id = gcloud.upload(
            tmp, "meep", verbose=verbose, input_hash=self._input_hash
        )
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
        check_cache: bool = False,
    ) -> Any:
        """Run MEEP simulation on the cloud.

        Args:
            parent_dir: Where to create the sim directory.
                Defaults to the current working directory.
            verbose: ``"quiet"`` no output, ``"status"`` status line,
                ``"full"`` stream solver logs.
            wait: If ``True`` (default), block until results are ready.
                If ``False``, upload + start and return the ``job_id``.
            check_cache: If ``True``, look for a completed cloud job with
                byte-identical inputs and reuse its results instead of
                submitting. A lookup failure degrades to a normal submit.

        Returns:
            ``SParameterResult`` when ``wait=True``, or ``job_id`` string
            when ``wait=False``.
        """
        from gsim import gcloud

        self._run_verbose = verbose
        if check_cache:
            tmp = self._prepare_upload_dir()
            self._input_hash, cached_job_id = gcloud.check_cache_for_dir(tmp, "meep")
            if cached_job_id is not None:
                self._job_id = cached_job_id
                if verbose != "quiet":
                    print(f"Cache hit: reusing job {cached_job_id}")  # noqa: T201
                if not wait:
                    return self._job_id
                return self.wait_for_results(verbose=verbose, parent_dir=parent_dir)
            self._job_id = gcloud.upload(
                tmp, "meep", verbose=False, input_hash=self._input_hash
            )
        else:
            self.upload(verbose=False)
        self.start(verbose=verbose != "quiet")
        if not wait:
            return self._job_id
        return self.wait_for_results(verbose=verbose, parent_dir=parent_dir)

    def run_local(
        self,
        output_dir: str | Path | None = None,
        *,
        python_executable: str | Path | None = None,
        num_processes: int | None = None,
        verbose: bool = True,
    ) -> Any:
        """Run MEEP simulation locally.

        Writes config, GDS, and runner script to ``output_dir``, then
        executes ``run_meep.py`` via Python (with meep installed).

        Args:
            output_dir: Directory for config/GDS/script output.
                If None, a temporary directory is created.
            python_executable: Path to the Python interpreter that has
                meep installed. If None, uses the current interpreter
                (``sys.executable``).
            num_processes: Number of MPI processes. If None (default),
                runs as a single process. When >1, uses ``mpirun -np``.
            verbose: Print progress messages.

        Returns:
            :class:`SParameterResult` parsed from the output CSV.

        Raises:
            FileNotFoundError: If meep is not installed.
            RuntimeError: If simulation fails.
        """
        import shutil
        import subprocess
        import sys
        import tempfile

        from gsim.meep.models.results import SParameterResult

        # Always regenerate config to reflect current settings
        self._run_verbose = "quiet" if not verbose else "status"
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="meep_local_"))
        output_dir = Path(output_dir)
        self.write_config(output_dir)

        script_path = output_dir / "run_meep.py"
        if not script_path.exists():
            raise FileNotFoundError(
                f"run_meep.py not found in {output_dir}. "
                "write_config() should have created it."
            )

        exe = str(python_executable or sys.executable)

        if num_processes is not None and num_processes > 1:
            mpirun = shutil.which("mpirun")
            if mpirun is None:
                raise RuntimeError(
                    "mpirun not found. Install an MPI runtime to use "
                    "num_processes > 1, or omit it for single-process mode."
                )
            cmd = [mpirun, "-np", str(num_processes), exe, str(script_path)]
        else:
            cmd = [exe, str(script_path)]

        if verbose:
            logger.info("Running MEEP simulation in %s", output_dir)
            logger.info("Command: %s", " ".join(cmd))

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                cwd=output_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            if verbose and result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                for line in result.stderr.splitlines():
                    if "warning" in line.lower():
                        logger.warning(line)
                    elif verbose:
                        logger.info(line)
        except subprocess.CalledProcessError as e:
            error_msg = f"MEEP simulation failed (rc={e.returncode})"
            if e.stdout:
                error_msg += f"\n\nStdout:\n{e.stdout[-4000:]}"
            if e.stderr:
                error_msg += f"\n\nStderr:\n{e.stderr[-4000:]}"
            raise RuntimeError(error_msg) from e
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Python executable not found: {exe}. "
                "Install MEEP or provide the correct python path."
            ) from e

        if verbose:
            logger.info("Simulation completed successfully")

        csv_path = output_dir / "s_parameters.csv"
        if csv_path.exists():
            return SParameterResult.from_csv(csv_path)

        return SParameterResult.from_directory(output_dir)

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def _add_index_plot_context(
        self,
        result: BuildResult,
        kwargs: dict[str, Any],
    ) -> None:
        """Add center-wavelength material and simulation context to a plot."""
        from gsim.meep.materials import resolve_materials

        # ``layer_stack`` contains every layer declared by the PDK, including
        # unpopulated metal layers.  The built material map is already limited
        # to materials that occur in the simulation geometry.
        used_materials = set(result.config.materials)
        kwargs["material_data"] = resolve_materials(
            used_materials,
            overrides=self._material_overrides(),
            wavelength_um=result.config.wavelength.wavelength,
        )
        kwargs["wavelength"] = result.config.wavelength.wavelength
        kwargs["is_3d"] = result.config.is_3d
        kwargs["plane"] = result.config.plane
        kwargs["background_material"] = result.config.background_material
        kwargs["layer_order"] = [
            entry.layer_name for entry in result.config.layer_stack
        ]

    def plot_2d(self, **kwargs: Any) -> Any:
        """Plot 2D cross-sections of the geometry.

        Uses :meth:`build_config` so the plot shows exactly what meep
        processes — including extended ports and PML boundaries.

        In XZ 2D mode (``solver.mode='2d'`` with ``y_cut`` set), ``slices``
        defaults to ``"y"`` and ``y`` defaults to the resolved ``y_cut``.

        The default ``kind="index"`` colors resolved material geometry by
        refractive index at the simulation center wavelength and shows PML,
        source, and monitor annotations. Pass ``kind="layers"`` for the
        categorical legacy view.

        Accepts the same keyword arguments as :func:`gsim.meep.viz.plot_2d`.
        """
        from gsim.meep.viz import plot_2d

        result = self.build_config()

        if self.solver.resolved_plane() == "xz":
            kwargs.setdefault("slices", "y")
            if kwargs.get("slices") == "y":
                kwargs.setdefault("y", result.config.y_cut)

        if kwargs.get("kind", "index") == "index":
            self._add_index_plot_context(result, kwargs)

        return plot_2d(
            component=result.component,
            stack=result.stack,
            domain_config=result.config.domain,
            source_port=result.config.source.port,
            extend_ports_length=0,
            gdsfactory_stack=result.gdsfactory_stack,
            port_data=result.config.ports,
            component_bbox=result.config.component_bbox,
            fiber_source=result.config.fiber_source,
            monitor_z_span=result.config.monitor_z_span,
            **kwargs,
        )

    def plot_2d_interactive(self, **kwargs: Any) -> Any:
        """Plot an interactive 2D cross-section using Plotly.

        Each layer and overlay element is a separate trace, so users can
        zoom, pan, and toggle individual layers/materials on and off via
        the legend.

        Uses :meth:`build_config` so the plot shows exactly what meep
        processes — including extended ports and PML boundaries.

        In XZ 2D mode (``solver.mode='2d'`` with ``y_cut`` set), ``slices``
        defaults to ``"y"`` and ``y`` defaults to the resolved ``y_cut``.

        The default ``kind="index"`` uses the same refractive-index map and
        simulation annotations as :meth:`plot_2d`. Pass ``kind="layers"`` for
        the categorical legacy view.

        Accepts the same keyword arguments as
        :func:`gsim.meep.viz.plot_2d_interactive`.

        Returns:
            ``plotly.graph_objects.Figure``.
        """
        from gsim.meep.viz import plot_2d_interactive

        result = self.build_config()

        if self.solver.resolved_plane() == "xz":
            kwargs.setdefault("slices", "y")
            if kwargs.get("slices") == "y":
                kwargs.setdefault("y", result.config.y_cut)

        if kwargs.get("kind", "index") == "index":
            self._add_index_plot_context(result, kwargs)

        return plot_2d_interactive(
            component=result.component,
            stack=result.stack,
            domain_config=result.config.domain,
            source_port=result.config.source.port,
            extend_ports_length=0,
            gdsfactory_stack=result.gdsfactory_stack,
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
            stack=result.stack,
            domain_config=result.config.domain,
            extend_ports_length=0,
            gdsfactory_stack=result.gdsfactory_stack,
            **kwargs,
        )
