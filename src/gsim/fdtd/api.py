"""User-facing configuration objects for GDSFactory FDTD simulations."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from math import isfinite
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

Vector3 = tuple[float, float, float]
Axis = Literal["x", "y", "z"]
SignedAxis = Literal["+x", "-x", "+y", "-y", "+z", "-z"]
Waveform = Literal["pulse", "continuous_wave"]
HeatmapQuantity = Literal[
    "abs_e", "intensity", "abs_h", "ex", "ey", "ez", "hx", "hy", "hz"
]


class _MutableModel(BaseModel):
    """Validated model that supports concise batch updates."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    def __call__(self, **updates: Any) -> Self:
        """Validate and apply several settings atomically."""
        updated = type(self).model_validate({**self.model_dump(), **updates})
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, getattr(updated, field_name))
        return self


class Geometry(BaseModel):
    """Component and meshing settings, expressed in micrometers and nanometers."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    component: Any | None = None
    settings: dict[str, Any] = Field(default_factory=dict)
    mesh_size_nm: float = Field(default=1000.0, gt=0)
    geometry_tolerance_nm: float = Field(default=10.0, gt=0, le=30.0)
    _resolver: Any = PrivateAttr(default=None)

    def bind(self, resolver: Any) -> None:
        """Attach the simulation's canonical PDK geometry resolver."""
        self._resolver = resolver

    def __call__(
        self,
        component: Any | None = None,
        *,
        settings: Mapping[str, Any] | None = None,
        mesh_size_nm: float | None = None,
        geometry_tolerance_nm: float | None = None,
    ) -> Any:
        """Configure geometry and return its canonical resolved representation."""
        updates: dict[str, Any] = {}
        if component is not None:
            updates["component"] = component
        if settings is not None:
            updates["settings"] = dict(settings)
        if mesh_size_nm is not None:
            updates["mesh_size_nm"] = mesh_size_nm
        if geometry_tolerance_nm is not None:
            updates["geometry_tolerance_nm"] = geometry_tolerance_nm
        updated = type(self).model_validate({**self.model_dump(), **updates})
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, getattr(updated, field_name))
        if self.component is None:
            raise ValueError("component is required")
        return self._resolver() if self._resolver is not None else self


class Material(_MutableModel):
    """Compatibility shorthand for a constant-index MaterialCard override."""

    refractive_index: float = Field(gt=0)


class Materials(_MutableModel):
    """Background card selection and optional constant-card overrides."""

    background: str = Field(default="SiO2", min_length=1)
    overrides: dict[str, Material] = Field(default_factory=dict)

    @field_validator("overrides", mode="before")
    @classmethod
    def coerce_materials(cls, value: Any) -> Any:
        """Allow ``{"si": 3.47}`` as a compact material override."""
        if value is None:
            return {}
        return {
            name: Material(refractive_index=item)
            if isinstance(item, (int, float))
            else item
            for name, item in dict(value).items()
        }

    def __getitem__(self, name: str) -> Material:
        """Return one explicit material override by name."""
        return self.overrides[name]

    def __setitem__(
        self, name: str, value: float | Material | Mapping[str, Any]
    ) -> None:
        """Validate and store one explicit material override."""
        updated = dict(self.overrides)
        updated[name] = (
            Material(refractive_index=value)
            if isinstance(value, (int, float))
            else Material.model_validate(value)
        )
        self(overrides=updated)


class Heatmap(_MutableModel):
    """Steady-state scalar field images requested from a plane monitor."""

    quantity: HeatmapQuantity = "intensity"
    wavelengths_um: list[float] = Field(min_length=1)

    @field_validator("wavelengths_um")
    @classmethod
    def validate_wavelengths(cls, values: list[float]) -> list[float]:
        """Require physical heatmap wavelengths."""
        if any(value <= 0 for value in values):
            raise ValueError("heatmap wavelengths must be positive")
        return values


class Source(_MutableModel):
    """Settings shared by every FDTD source."""

    waveform: Waveform = "pulse"
    wavelength_um: float = Field(default=1.55, gt=0)
    wavelength_span_um: float = Field(default=0.1, ge=0)
    num_wavelengths: int = Field(default=101, ge=1)
    amplitude: float = 1.0

    @model_validator(mode="after")
    def validate_spectrum(self) -> Self:
        """Require a physical sweep and a nonzero source."""
        if self.wavelength_span_um / 2 >= self.wavelength_um:
            raise ValueError(
                "wavelength_span_um must be smaller than twice wavelength_um"
            )
        if self.waveform == "continuous_wave" and self.num_wavelengths != 1:
            raise ValueError("continuous_wave requires num_wavelengths=1")
        if self.amplitude == 0:
            raise ValueError("source amplitude cannot be zero")
        return self

    @property
    def wavelength_halfspan_um(self) -> float:
        """Return the half-span expected by the ZapFDTD runtime schema."""
        return self.wavelength_span_um / 2


class PortSource(Source):
    """Excite an explicitly selected PDK port."""

    type: Literal["port"] = "port"
    port: str | None = Field(default=None, min_length=1)
    vertical_axis: Literal["+z", "-z"] = "+z"
    vertical_aperture_width_um: float | None = Field(default=None, gt=0)
    vertical_waist_radius_um: float | None = Field(default=None, gt=0)
    vertical_monitor_heatmap: Heatmap | None = None


class DipoleSource(Source):
    """Single-cell electric-current source."""

    type: Literal["dipole"] = "dipole"
    position_um: Vector3
    current_axis: Axis


class LineCurrentSource(Source):
    """Line of in-phase electric-current sources."""

    type: Literal["line_current"] = "line_current"
    position_um: Vector3
    line_axis: Axis
    current_axis: Axis
    length_um: float = Field(gt=0)


class GaussianBeamSource(Source):
    """Focused Gaussian beam injected through an axis-aligned aperture."""

    type: Literal["gaussian_beam"] = "gaussian_beam"
    center_um: Vector3
    size_um: Vector3
    aperture_normal: SignedAxis
    propagation_direction: Vector3
    e_polarization: Vector3
    focal_point_um: Vector3
    waist_radius_um: float = Field(gt=0)
    refractive_index: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_aperture(self) -> Self:
        """Require a plane whose zero-size axis matches the normal."""
        if any(size < 0 for size in self.size_um):
            raise ValueError("size_um components cannot be negative")
        axis = {"x": 0, "y": 1, "z": 2}[self.aperture_normal[-1]]
        if self.size_um[axis] != 0:
            raise ValueError("size_um must be zero along aperture_normal")
        return self


SourceType = PortSource | DipoleSource | LineCurrentSource | GaussianBeamSource


class FiberMode(_MutableModel):
    """Analytic Gaussian fiber profile for overlap projection."""

    propagation_direction: Vector3
    e_polarization: Vector3
    focal_point_um: Vector3
    waist_radius_um: float = Field(gt=0)
    refractive_index: float | None = Field(default=None, gt=0)


class PlaneMonitor(_MutableModel):
    """Flux, field, and optional fiber-overlap measurement on a plane."""

    name: str = Field(min_length=1)
    center_um: Vector3
    size_um: Vector3
    normal: SignedAxis
    flux: bool = True
    wavelengths_um: list[float] | None = None
    heatmap: Heatmap | None = None
    fiber_mode: FiberMode | None = None

    @model_validator(mode="after")
    def validate_plane(self) -> Self:
        """Require planar geometry and at least one measurement."""
        if any(size < 0 for size in self.size_um):
            raise ValueError("size_um components cannot be negative")
        axis = {"x": 0, "y": 1, "z": 2}[self.normal[-1]]
        if self.size_um[axis] != 0:
            raise ValueError("size_um must be zero along the monitor normal")
        if self.wavelengths_um is not None and any(
            wavelength <= 0 for wavelength in self.wavelengths_um
        ):
            raise ValueError("monitor wavelengths must be positive")
        if not self.flux and self.heatmap is None and self.fiber_mode is None:
            raise ValueError(
                "a plane monitor must request flux, a heatmap, or fiber overlap"
            )
        return self


class Monitors:
    """Named collection of additional plane monitors; port monitors are implicit."""

    def __init__(self, monitors: list[PlaneMonitor | Mapping[str, Any]] | None = None):
        """Validate an optional initial list of named plane monitors."""
        self._items: dict[str, PlaneMonitor] = {}
        for monitor in monitors or []:
            validated = PlaneMonitor.model_validate(monitor)
            self.add(validated)

    def add(self, monitor: PlaneMonitor) -> PlaneMonitor:
        """Add an already configured monitor, requiring a unique name."""
        if monitor.name in self._items:
            raise ValueError(f"monitor {monitor.name!r} already exists")
        self._items[monitor.name] = monitor
        return monitor

    def add_plane(self, name: str, **settings: Any) -> PlaneMonitor:
        """Create and add a named plane monitor."""
        return self.add(PlaneMonitor(name=name, **settings))

    def remove(self, name: str) -> PlaneMonitor:
        """Remove and return one named plane monitor."""
        return self._items.pop(name)

    def clear(self) -> None:
        """Remove all explicitly configured plane monitors."""
        self._items.clear()

    def __getitem__(self, name: str) -> PlaneMonitor:
        """Return one monitor by name."""
        return self._items[name]

    def __iter__(self) -> Iterator[PlaneMonitor]:
        """Iterate over monitors in insertion order."""
        return iter(self._items.values())

    def __len__(self) -> int:
        """Return the number of explicitly configured monitors."""
        return len(self._items)

    def model_dump(self) -> list[dict[str, Any]]:
        """Return serializable monitor dictionaries."""
        return [monitor.model_dump() for monitor in self]


class Domain(_MutableModel):
    """Simulation-domain padding and absorbing boundary settings."""

    padding_um: float = Field(default=1.0, gt=0)
    pml_cells: int = Field(default=32, ge=0)
    x_bounds: tuple[float, float] | None = Field(
        default=None,
        description="Optional physical X-domain bounds in micrometers.",
    )
    y_bounds: tuple[float, float] | None = Field(
        default=None,
        description="Optional physical Y-domain bounds in micrometers.",
    )
    z_bounds: tuple[float, float] | None = Field(
        default=None,
        description="Optional physical Z-domain bounds in micrometers.",
    )

    @field_validator("x_bounds", "y_bounds", "z_bounds")
    @classmethod
    def validate_axis_bounds(
        cls, value: tuple[float, float] | None
    ) -> tuple[float, float] | None:
        """Require finite, strictly ordered explicit domain bounds."""
        if value is None:
            return None
        lower, upper = value
        if not isfinite(lower) or not isfinite(upper):
            raise ValueError("domain bounds must be finite")
        if lower >= upper:
            raise ValueError("domain lower bound must be smaller than upper bound")
        return value


class Solver(_MutableModel):
    """Yee-grid resolution and termination controls."""

    cell_size_nm: float = Field(default=60.0, gt=0)
    max_timesteps: int | None = Field(default=None, gt=0)
    energy_decay_fraction: float = Field(default=1e-6, gt=0, lt=1)
    max_wall_seconds: float = Field(default=3600.0, ge=0)


__all__ = [
    "Axis",
    "DipoleSource",
    "Domain",
    "FiberMode",
    "GaussianBeamSource",
    "Geometry",
    "Heatmap",
    "HeatmapQuantity",
    "LineCurrentSource",
    "Material",
    "Materials",
    "Monitors",
    "PlaneMonitor",
    "PortSource",
    "SignedAxis",
    "Solver",
    "Source",
    "SourceType",
    "Vector3",
]
