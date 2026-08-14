"""Validated ZapFDTD schema-version-1 configuration models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from gsim.common.materials import MaterialSnapshot
from gsim.fdtd.models import FDTDConfigError, MeshManifest


class _StrictModel(BaseModel):
    """Base model that rejects fields ZapFDTD does not understand."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class MaterialConfig(_StrictModel):
    """Scalar real optical material supported by Zap schema v1."""

    refractive_index: float = Field(gt=0)


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
    """All mesh physical groups consumed by ZapFDTD."""

    volumes: dict[str, RegionConfig] = Field(min_length=1)
    layers: dict[str, RegionConfig] = Field(min_length=1)
    ports: dict[str, PortConfig] = Field(min_length=1)


class ExcitationConfig(_StrictModel):
    """Initial eigenmode pulse configuration."""

    type: Literal["eigenmode"] = "eigenmode"
    waveform: Literal["pulse", "continuous_wave"] = "pulse"
    center_wavelength: float = Field(gt=0)
    wavelength_halfspan: float = Field(ge=0)
    num_wavelengths: int = Field(ge=1)
    default_port: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_wavelength_span(self) -> ExcitationConfig:
        """Keep the wavelength sweep positive."""
        if self.wavelength_halfspan >= self.center_wavelength:
            raise ValueError("wavelength_halfspan must be smaller than the center")
        if self.waveform == "continuous_wave" and self.num_wavelengths != 1:
            raise ValueError("continuous_wave requires num_wavelengths=1")
        return self


class GridConfig(_StrictModel):
    """Yee-grid and PML settings."""

    nanometers_per_cell: float = Field(gt=0)
    pml_cells: int = Field(ge=0)


class RunConfig(_StrictModel):
    """FDTD termination controls."""

    max_timesteps: int | None = Field(default=None, gt=0)
    energy_decay_fraction: float = Field(gt=0, lt=1)
    max_wall_seconds: float = Field(ge=0)


class ZapConfig(_StrictModel):
    """Complete ZapFDTD runtime configuration."""

    schema_version: Literal[1] = 1
    mesh_file: Literal["mesh.msh"] = "mesh.msh"
    length_scale_meters: float = Field(default=1e-9, ge=1e-9, le=1e-9)
    background_refractive_index: float = Field(gt=0)
    materials: dict[str, MaterialConfig] = Field(min_length=1)
    geometry: GeometryConfig
    excitation: ExcitationConfig
    grid: GridConfig
    run: RunConfig

    @model_validator(mode="after")
    def validate_references(self) -> ZapConfig:
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
        if self.excitation.default_port not in self.geometry.ports:
            raise ValueError(
                f"default_port {self.excitation.default_port!r} is not declared"
            )
        return self


def _material_config(snapshot: MaterialSnapshot) -> MaterialConfig:
    """Convert one lossless scalar snapshot to the Zap material schema."""
    if snapshot.extinction_coefficient != 0:
        raise FDTDConfigError(
            f"Material {snapshot.material_name!r} has extinction coefficient "
            f"{snapshot.extinction_coefficient}; Zap schema v1 supports only "
            "lossless real refractive indices."
        )
    return MaterialConfig(refractive_index=snapshot.refractive_index)


def build_zap_config(
    manifest: MeshManifest,
    material_snapshots: Mapping[str, MaterialSnapshot],
    *,
    background_material: str,
    center_wavelength_nm: float,
    wavelength_halfspan_nm: float,
    num_wavelengths: int,
    default_port: str,
    nanometers_per_cell: float,
    pml_cells: int,
    max_timesteps: int | None,
    energy_decay_fraction: float,
    max_wall_seconds: float,
) -> ZapConfig:
    """Build and cross-validate a Zap config from a mesh manifest."""
    if background_material not in material_snapshots:
        raise FDTDConfigError(
            f"Background material {background_material!r} has no snapshot."
        )
    materials = {
        name: _material_config(snapshot)
        for name, snapshot in material_snapshots.items()
    }
    return ZapConfig(
        background_refractive_index=materials[background_material].refractive_index,
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
        excitation=ExcitationConfig(
            center_wavelength=center_wavelength_nm,
            wavelength_halfspan=wavelength_halfspan_nm,
            num_wavelengths=num_wavelengths,
            default_port=default_port,
        ),
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
    "ExcitationConfig",
    "GeometryConfig",
    "GridConfig",
    "MaterialConfig",
    "PortConfig",
    "RegionConfig",
    "RunConfig",
    "ZapConfig",
    "build_zap_config",
]
