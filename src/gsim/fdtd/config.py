"""Validated GDSFactory FDTD schema-version-1 configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gsim.common.materials import MaterialSnapshot
from gsim.fdtd.material_config import MaterialConfig, material_config_from_snapshot
from gsim.fdtd.models import FDTDConfigError, MeshManifest


class _StrictModel(BaseModel):
    """Base model that rejects fields GDSFactory FDTD does not understand."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class RegionConfig(_StrictModel):
    """Material assignment for a three-dimensional physical group."""

    phys_group: int = Field(gt=0)
    material: str = Field(min_length=1)
    priority: int = Field(ge=0)


class PortConfig(_StrictModel):
    """Layer assignment and outward normal for a port physical group."""

    phys_group: int = Field(gt=0)
    layer: str = Field(min_length=1)
    normal: tuple[int, int, int]

    @model_validator(mode="after")
    def validate_axis_aligned_normal(self) -> PortConfig:
        """Require exactly one signed unit-axis component."""
        if sum(component != 0 for component in self.normal) != 1 or any(
            component not in {-1, 0, 1} for component in self.normal
        ):
            raise ValueError("port normal must be one signed Cartesian unit axis")
        return self


class GeometryConfig(_StrictModel):
    """All mesh physical groups consumed by GDSFactory FDTD."""

    volumes: dict[str, RegionConfig] = Field(min_length=1)
    layers: dict[str, RegionConfig] = Field(min_length=1)
    ports: dict[str, PortConfig] = Field(default_factory=dict)


Vector3 = tuple[float, float, float]
Axis = Literal["x", "y", "z"]
SignedAxis = Literal["+x", "-x", "+y", "-y", "+z", "-z"]


class GaussianBeamConfig(_StrictModel):
    """Free-space Gaussian beam injected through an axis-aligned aperture."""

    region_min: Vector3
    region_max: Vector3
    aperture_normal: SignedAxis
    propagation_direction: Vector3
    e_polarization: Vector3
    focal_point: Vector3
    waist_radius: float = Field(gt=0)
    refractive_index: float = Field(gt=0)


class DipoleConfig(_StrictModel):
    """Single-cell electric-current source."""

    position: Vector3
    current_axis: Axis


class LineCurrentConfig(_StrictModel):
    """Line of in-phase electric-current sources."""

    position: Vector3
    line_axis: Axis
    current_axis: Axis
    length: float = Field(gt=0)


class FiberModeConfig(_StrictModel):
    """Analytic Gaussian fiber profile used by a plane monitor."""

    propagation_direction: Vector3
    e_polarization: Vector3
    focal_point: Vector3
    waist_radius: float = Field(gt=0)
    refractive_index: float = Field(gt=0)


class HeatmapConfig(_StrictModel):
    """Scalar field images requested from a plane monitor."""

    quantity: Literal[
        "abs_e", "intensity", "abs_h", "ex", "ey", "ez", "hx", "hy", "hz"
    ] = "intensity"
    wavelengths: list[float] = Field(min_length=1)

    @field_validator("wavelengths")
    @classmethod
    def validate_wavelengths(cls, values: list[float]) -> list[float]:
        """Require physical heatmap wavelengths."""
        if any(value <= 0 for value in values):
            raise ValueError("heatmap wavelengths must be positive")
        return values


class PlaneMonitorConfig(_StrictModel):
    """Flux and optional fiber-overlap monitor on an arbitrary plane."""

    name: str = Field(min_length=1)
    region_min: Vector3
    region_max: Vector3
    normal: SignedAxis
    flux: bool = True
    wavelengths: list[float] | None = None
    heatmap: HeatmapConfig | None = None
    fiber_mode: FiberModeConfig | None = None

    @model_validator(mode="after")
    def validate_plane(self) -> PlaneMonitorConfig:
        """Require ordered bounds and zero thickness along the normal axis."""
        if any(
            lower > upper
            for lower, upper in zip(self.region_min, self.region_max, strict=True)
        ):
            raise ValueError("monitor region_min cannot exceed region_max")
        normal_axis = {"x": 0, "y": 1, "z": 2}[self.normal[-1]]
        if self.region_min[normal_axis] != self.region_max[normal_axis]:
            raise ValueError("monitor region must be planar along its normal axis")
        if self.wavelengths is not None and any(
            wavelength <= 0 for wavelength in self.wavelengths
        ):
            raise ValueError("monitor wavelengths must be positive")
        return self


class ExcitationConfig(_StrictModel):
    """Initial eigenmode pulse configuration."""

    type: Literal["eigenmode", "dipole", "line_current", "gaussian_beam"] = "eigenmode"
    waveform: Literal["pulse", "continuous_wave"] = "pulse"
    center_wavelength: float = Field(gt=0)
    wavelength_halfspan: float = Field(ge=0)
    num_wavelengths: int = Field(ge=1)
    amplitude: float = 1.0
    default_port: str | None = Field(default=None, min_length=1)
    dipole: DipoleConfig | None = None
    line_current: LineCurrentConfig | None = None
    gaussian_beam: GaussianBeamConfig | None = None

    @model_validator(mode="after")
    def validate_wavelength_span(self) -> ExcitationConfig:
        """Keep the wavelength sweep positive."""
        if self.wavelength_halfspan >= self.center_wavelength:
            raise ValueError("wavelength_halfspan must be smaller than the center")
        if self.waveform == "continuous_wave" and self.num_wavelengths != 1:
            raise ValueError("continuous_wave requires num_wavelengths=1")
        if self.amplitude == 0:
            raise ValueError("excitation amplitude cannot be zero")
        source_blocks = {
            "dipole": self.dipole,
            "line_current": self.line_current,
            "gaussian_beam": self.gaussian_beam,
        }
        if self.type == "eigenmode":
            if self.default_port is None:
                raise ValueError("eigenmode excitation requires default_port")
            if any(block is not None for block in source_blocks.values()):
                raise ValueError("eigenmode excitation cannot include a source block")
            return self
        if self.default_port is not None:
            raise ValueError(f"{self.type} excitation cannot use default_port")
        for source_type, block in source_blocks.items():
            if (source_type == self.type) != (block is not None):
                raise ValueError(
                    f"{self.type} excitation requires only its matching source block"
                )
        return self


def _source_wavelength_range_nm(
    excitation: ExcitationConfig,
) -> tuple[float, float] | None:
    """Return the nonzero wavelength band ZapFDTD derives from the source."""
    if excitation.waveform == "continuous_wave":
        return None
    center = excitation.center_wavelength
    halfspan = excitation.wavelength_halfspan
    if excitation.type == "eigenmode":
        return (center - halfspan, center + halfspan) if halfspan > 0 else None
    frequency_center = 1.0 / center
    frequency_width = (
        0.5 * (1.0 / (center - halfspan) - 1.0 / (center + halfspan))
        if halfspan > 0
        else frequency_center * 0.1
    )
    return (
        1.0 / (frequency_center + frequency_width),
        1.0 / (frequency_center - frequency_width),
    )


class GridConfig(_StrictModel):
    """Yee-grid and PML settings."""

    nanometers_per_cell: float = Field(gt=0)
    pml_cells: int = Field(ge=0)


class RunConfig(_StrictModel):
    """GDSFactory FDTD termination controls."""

    max_timesteps: int | None = Field(default=None, gt=0)
    energy_decay_fraction: float = Field(gt=0, lt=1)
    max_wall_seconds: float = Field(ge=0)


class FDTDConfig(_StrictModel):
    """Complete GDSFactory FDTD runtime configuration."""

    schema_version: Literal[1] = 1
    mesh_file: Literal["mesh.msh"] = "mesh.msh"
    length_scale_meters: float = Field(default=1e-9, ge=1e-9, le=1e-9)
    background_refractive_index: float = Field(gt=0)
    materials: dict[str, MaterialConfig] = Field(min_length=1)
    geometry: GeometryConfig
    excitation: ExcitationConfig
    monitors: list[PlaneMonitorConfig] = Field(default_factory=list)
    grid: GridConfig
    run: RunConfig

    @model_validator(mode="after")
    def validate_references(self) -> FDTDConfig:
        """Require all material, layer, and port references to exist."""
        material_names = set(self.materials)
        for group_name, region in {
            **self.geometry.volumes,
            **self.geometry.layers,
        }.items():
            if region.material not in material_names:
                raise ValueError(
                    f"geometry group {group_name!r} references unknown material "
                    f"{region.material!r}"
                )
        layer_names = set(self.geometry.layers)
        for port_name, port in self.geometry.ports.items():
            if port.layer not in layer_names:
                raise ValueError(
                    f"port {port_name!r} references unknown layer {port.layer!r}"
                )
        if (
            self.excitation.type == "eigenmode"
            and self.excitation.default_port not in self.geometry.ports
        ):
            raise ValueError(
                f"default_port {self.excitation.default_port!r} is not declared"
            )
        monitor_names = [monitor.name for monitor in self.monitors]
        if len(monitor_names) != len(set(monitor_names)):
            raise ValueError("monitor names must be unique")
        return self


def build_fdtd_config(
    manifest: MeshManifest,
    material_snapshots: Mapping[str, MaterialSnapshot],
    *,
    background_material: str,
    center_wavelength_nm: float,
    wavelength_halfspan_nm: float,
    num_wavelengths: int,
    default_port: str | None,
    nanometers_per_cell: float,
    pml_cells: int,
    max_timesteps: int | None,
    energy_decay_fraction: float,
    max_wall_seconds: float,
    excitation: ExcitationConfig | None = None,
    gaussian_beam: GaussianBeamConfig | None = None,
    monitors: list[PlaneMonitorConfig] | None = None,
) -> FDTDConfig:
    """Build and cross-validate a GDSFactory FDTD config from a mesh manifest."""
    if background_material not in material_snapshots:
        raise FDTDConfigError(
            f"Background material {background_material!r} has no snapshot."
        )
    resolved_excitation = excitation or ExcitationConfig(
        type="gaussian_beam" if gaussian_beam is not None else "eigenmode",
        center_wavelength=center_wavelength_nm,
        wavelength_halfspan=wavelength_halfspan_nm,
        num_wavelengths=num_wavelengths,
        default_port=default_port,
        gaussian_beam=gaussian_beam,
    )
    source_wavelength_range_nm = _source_wavelength_range_nm(resolved_excitation)
    materials = {
        name: material_config_from_snapshot(snapshot, source_wavelength_range_nm)
        for name, snapshot in material_snapshots.items()
    }
    return FDTDConfig(
        background_refractive_index=material_snapshots[
            background_material
        ].refractive_index,
        materials=materials,
        geometry=GeometryConfig(
            volumes={
                name: RegionConfig(
                    phys_group=group.physical_tag,
                    material=group.material,
                    priority=group.priority,
                )
                for name, group in manifest.volumes.items()
            },
            layers={
                name: RegionConfig(
                    phys_group=group.physical_tag,
                    material=group.material,
                    priority=group.priority,
                )
                for name, group in manifest.layers.items()
            },
            ports={
                name: PortConfig(
                    phys_group=group.physical_tag,
                    layer=group.layer,
                    normal=group.normal,
                )
                for name, group in manifest.ports.items()
            },
        ),
        excitation=resolved_excitation,
        monitors=monitors or [],
        grid=GridConfig(
            nanometers_per_cell=nanometers_per_cell,
            pml_cells=pml_cells,
        ),
        run=RunConfig(
            max_timesteps=max_timesteps,
            energy_decay_fraction=energy_decay_fraction,
            max_wall_seconds=max_wall_seconds,
        ),
    )


__all__ = [
    "DipoleConfig",
    "ExcitationConfig",
    "FDTDConfig",
    "FiberModeConfig",
    "GaussianBeamConfig",
    "GeometryConfig",
    "GridConfig",
    "HeatmapConfig",
    "LineCurrentConfig",
    "MaterialConfig",
    "PlaneMonitorConfig",
    "PortConfig",
    "RegionConfig",
    "RunConfig",
    "build_fdtd_config",
]
