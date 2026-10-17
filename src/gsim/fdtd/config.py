"""Validated GDSFactory FDTD schema-version-1 configuration models."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pdk_schema import Sellmeier
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gsim.common.materials import (
    MaterialResolutionError,
    MaterialSnapshot,
    evaluate_material_card,
    tabulated_wavelengths_um,
)
from gsim.fdtd.models import FDTDConfigError, MeshManifest


class _StrictModel(BaseModel):
    """Base model that rejects fields GDSFactory FDTD does not understand."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class IndexTableConfig(_StrictModel):
    """Tabulated complex refractive index, with wavelengths in nanometers."""

    wavelength_nm: list[float] = Field(min_length=2)
    n: list[float] = Field(min_length=2)
    k: list[float] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        """Require physical samples on one increasing wavelength grid."""
        if not len(self.wavelength_nm) == len(self.n) == len(self.k):
            raise ValueError("wavelength_nm, n, and k must have equal lengths")
        if self.wavelength_nm[0] <= 0 or any(
            right <= left
            for left, right in zip(
                self.wavelength_nm, self.wavelength_nm[1:], strict=False
            )
        ):
            raise ValueError("table wavelengths must be positive and increasing")
        if any(value <= 0 for value in self.n):
            raise ValueError("table refractive indices must be positive")
        if any(value < 0 for value in self.k):
            raise ValueError("table extinction coefficients cannot be negative")
        return self


class SellmeierConfig(_StrictModel):
    """Sellmeier oscillator strengths and squared pole wavelengths in um^2."""

    b: list[float] = Field(min_length=1)
    c_um2: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_terms(self) -> Self:
        """Require one pole position for every oscillator strength."""
        if len(self.b) != len(self.c_um2):
            raise ValueError("Sellmeier b and c_um2 must have equal lengths")
        return self


class DispersionConfig(_StrictModel):
    """Wavelength-dependent material model accepted by GDSFactory FDTD."""

    wavelength_range_nm: tuple[float, float] | None = None
    table: IndexTableConfig | None = None
    sellmeier: SellmeierConfig | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Require exactly one model, and a fit range for closed forms."""
        if (self.table is None) == (self.sellmeier is None):
            raise ValueError(
                "dispersion must contain exactly one of table or sellmeier"
            )
        if self.table is not None:
            if self.wavelength_range_nm is not None:
                raise ValueError("table dispersion cannot set wavelength_range_nm")
            return self
        if self.wavelength_range_nm is None:
            raise ValueError("sellmeier dispersion needs wavelength_range_nm")
        lower, upper = self.wavelength_range_nm
        if not 0 < lower < upper:
            raise ValueError("wavelength_range_nm must satisfy 0 < lower < upper")
        return self


class MaterialConfig(_StrictModel):
    """Constant or dispersive optical material for GDSFactory FDTD."""

    refractive_index: float | None = Field(default=None, gt=0)
    dispersion: DispersionConfig | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Require exactly one constant or dispersive representation."""
        if (self.refractive_index is None) == (self.dispersion is None):
            raise ValueError(
                "material needs exactly one of refractive_index or dispersion"
            )
        return self


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


class BoundsConfig(_StrictModel):
    """Axis-aligned box in mesh units."""

    min: Vector3
    max: Vector3

    @model_validator(mode="after")
    def validate_ordered(self) -> BoundsConfig:
        """Require a box with positive extent along every axis."""
        if any(lower >= upper for lower, upper in zip(self.min, self.max, strict=True)):
            raise ValueError("bounds min must be smaller than max on every axis")
        return self


class GridConfig(_StrictModel):
    """Yee-grid and PML settings."""

    nanometers_per_cell: float = Field(gt=0)
    pml_cells: int = Field(ge=0)
    non_pml_bounds: BoundsConfig


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


# Evenly spaced samples written for a non-Sellmeier card; a tabulated card's
# own knots inside the band are added to them.
_DISPERSION_SAMPLES = 21
# Fraction of the band added on each side so the fit holds at the band edges.
_BAND_MARGIN = 0.1

_Samples = list[tuple[float, float, float]]
# Warnings point past every gsim frame, at the user's write() or run() call.
_GSIM_DIR = str(Path(__file__).parents[1])


def _evaluate_nm(
    snapshot: MaterialSnapshot, wavelength_nm: float
) -> tuple[float, float]:
    """Evaluate a snapshot's card at a wavelength in nanometers."""
    return evaluate_material_card(
        snapshot.card, wavelength_nm / 1000, snapshot.material_name
    )


def _fit_band(
    snapshot: MaterialSnapshot,
    band_nm: tuple[float, float],
) -> tuple[float, float]:
    """Widen each side of the band by the margin where the card allows it."""
    lower, upper = band_nm
    margin = _BAND_MARGIN * (upper - lower)
    edges = []
    for edge, padded in (
        (lower, max(lower - margin, lower / 2)),
        (upper, upper + margin),
    ):
        try:
            _evaluate_nm(snapshot, padded)
        except MaterialResolutionError:
            edges.append(edge)
        else:
            edges.append(padded)
    return edges[0], edges[1]


def _sample_card(
    snapshot: MaterialSnapshot,
    band_nm: tuple[float, float],
) -> tuple[tuple[float, float], _Samples]:
    """Sample (wavelength_nm, n, k) over the band and its margins."""
    fit_lower, fit_upper = _fit_band(snapshot, band_nm)
    wavelengths_nm = {
        round(float(wavelength), 6)
        for wavelength in np.linspace(fit_lower, fit_upper, _DISPERSION_SAMPLES)
    }
    wavelengths_nm.update(
        round(1000 * wavelength, 6)
        for wavelength in tabulated_wavelengths_um(snapshot.card)
        if fit_lower < 1000 * wavelength < fit_upper
    )
    try:
        samples = [
            (wavelength, *_evaluate_nm(snapshot, wavelength))
            for wavelength in sorted(wavelengths_nm)
        ]
    except MaterialResolutionError as error:
        raise FDTDConfigError(
            f"Material {snapshot.material_name!r} cannot be evaluated over the "
            f"source band {band_nm[0]:g} to {band_nm[1]:g} nm: {error}"
        ) from error
    return (fit_lower, fit_upper), samples


def _permittivity_variation(snapshot: MaterialSnapshot, samples: _Samples) -> float:
    """Return the largest relative permittivity change from the centre value."""
    center = complex(snapshot.refractive_index, snapshot.extinction_coefficient) ** 2
    return max(abs(complex(n, k) ** 2 - center) for _, n, k in samples) / abs(center)


def _lossless_scalar(snapshot: MaterialSnapshot, remedy: str) -> MaterialConfig:
    """Return the source-wavelength index of a lossless material."""
    if snapshot.extinction_coefficient != 0:
        raise FDTDConfigError(
            f"Material {snapshot.material_name!r} has extinction coefficient "
            f"{snapshot.extinction_coefficient} at the source wavelength; "
            f"GDSFactory FDTD encodes loss only in a dispersion table, {remedy}."
        )
    return MaterialConfig(refractive_index=snapshot.refractive_index)


def _flattened_change(
    snapshot: MaterialSnapshot,
    band_nm: tuple[float, float],
) -> str | None:
    """Describe how much a constant index misrepresents a card over the band."""
    try:
        _, samples = _sample_card(snapshot, band_nm)
    except FDTDConfigError:
        return "not evaluable over the whole band"
    variation = _permittivity_variation(snapshot, samples)
    if variation == 0:
        return None
    return f"{100 * variation:.2g} % permittivity change"


def _dispersive_config(
    snapshot: MaterialSnapshot,
    band_nm: tuple[float, float],
) -> MaterialConfig:
    """Write a card as a Sellmeier model, an n, k table, or a constant.

    A constant lossless card stays a scalar index, a lossless Sellmeier card is
    passed through unchanged, and any other card is written as a table.
    """
    fit_band, samples = _sample_card(snapshot, band_nm)
    lossless = all(k == 0 for _, _, k in samples)
    if lossless and _permittivity_variation(snapshot, samples) == 0:
        return MaterialConfig(refractive_index=snapshot.refractive_index)
    optical = snapshot.card.optical
    model = None if optical is None else optical.permittivity
    if (
        lossless
        and isinstance(model, Sellmeier)
        and model.offset == 0
        and model.conductivity is None
    ):
        return MaterialConfig(
            dispersion=DispersionConfig(
                wavelength_range_nm=fit_band,
                sellmeier=SellmeierConfig(
                    b=[term.b for term in model.terms],
                    c_um2=[term.c_um**2 for term in model.terms],
                ),
            )
        )
    return MaterialConfig(
        dispersion=DispersionConfig(
            table=IndexTableConfig(
                wavelength_nm=[wavelength for wavelength, _, _ in samples],
                n=[n for _, n, _ in samples],
                k=[k for _, _, k in samples],
            )
        )
    )


def _material_configs(
    material_snapshots: Mapping[str, MaterialSnapshot],
    band_nm: tuple[float, float] | None,
    dispersion: bool,
) -> dict[str, MaterialConfig]:
    """Convert every material, warning once about cards made constant."""
    if band_nm is None:
        remedy = (
            "which needs a pulsed source with a nonzero wavelength span and "
            "materials(dispersion=True)"
        )
        return {
            name: _lossless_scalar(snapshot, remedy)
            for name, snapshot in material_snapshots.items()
        }
    if dispersion:
        return {
            name: _dispersive_config(snapshot, band_nm)
            for name, snapshot in material_snapshots.items()
        }
    remedy = "which materials(dispersion=False), the default, does not write"
    materials = {
        name: _lossless_scalar(snapshot, remedy)
        for name, snapshot in material_snapshots.items()
    }
    flattened = {
        name: change
        for name, snapshot in material_snapshots.items()
        if (change := _flattened_change(snapshot, band_nm)) is not None
    }
    if flattened:
        details = ", ".join(f"{name} ({change})" for name, change in flattened.items())
        warnings.warn(
            "Material dispersion is not sent to GDSFactory FDTD; these materials "
            "use their index at the source wavelength across the "
            f"{band_nm[0]:g} to {band_nm[1]:g} nm band: {details}. "
            "materials(dispersion=True) sends dispersive models instead.",
            UserWarning,
            skip_file_prefixes=(_GSIM_DIR,),
        )
    return materials


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
    dispersion: bool = False,
) -> FDTDConfig:
    """Build and cross-validate a GDSFactory FDTD config from a mesh manifest.

    By default each material is written as its index at the source
    wavelength, with one warning naming every card that varies over the
    source band. With ``dispersion``, such cards are written as dispersive
    models instead.
    """
    if background_material not in material_snapshots:
        raise FDTDConfigError(
            f"Background material {background_material!r} has no snapshot."
        )
    if excitation is not None:
        center_wavelength_nm = excitation.center_wavelength
        wavelength_halfspan_nm = excitation.wavelength_halfspan
    single_wavelength = wavelength_halfspan_nm == 0 or (
        excitation is not None and excitation.waveform == "continuous_wave"
    )
    band_nm = (
        None
        if single_wavelength
        else (
            center_wavelength_nm - wavelength_halfspan_nm,
            center_wavelength_nm + wavelength_halfspan_nm,
        )
    )
    materials = _material_configs(material_snapshots, band_nm, dispersion)
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
        excitation=excitation
        or ExcitationConfig(
            type="gaussian_beam" if gaussian_beam is not None else "eigenmode",
            center_wavelength=center_wavelength_nm,
            wavelength_halfspan=wavelength_halfspan_nm,
            num_wavelengths=num_wavelengths,
            default_port=default_port,
            gaussian_beam=gaussian_beam,
        ),
        monitors=monitors or [],
        grid=GridConfig(
            nanometers_per_cell=nanometers_per_cell,
            pml_cells=pml_cells,
            non_pml_bounds=BoundsConfig(
                min=manifest.domain_min_nm,
                max=manifest.domain_max_nm,
            ),
        ),
        run=RunConfig(
            max_timesteps=max_timesteps,
            energy_decay_fraction=energy_decay_fraction,
            max_wall_seconds=max_wall_seconds,
        ),
    )


__all__ = [
    "BoundsConfig",
    "DipoleConfig",
    "DispersionConfig",
    "ExcitationConfig",
    "FDTDConfig",
    "FiberModeConfig",
    "GaussianBeamConfig",
    "GeometryConfig",
    "GridConfig",
    "HeatmapConfig",
    "IndexTableConfig",
    "LineCurrentConfig",
    "MaterialConfig",
    "PlaneMonitorConfig",
    "PortConfig",
    "RegionConfig",
    "RunConfig",
    "SellmeierConfig",
    "build_fdtd_config",
]
