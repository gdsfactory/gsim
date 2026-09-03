"""Tests for the strict GDSFactory FDTD configuration boundary."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gsim.common.materials import resolve_material_snapshot
from gsim.fdtd.config import (
    FDTDConfig,
    FiberModeConfig,
    GaussianBeamConfig,
    PlaneMonitorConfig,
    build_fdtd_config,
)
from gsim.fdtd.models import (
    MeshGroup,
    MeshManifest,
    PortMeshGroup,
)


def _manifest() -> MeshManifest:
    return MeshManifest(
        volumes={
            "background": MeshGroup(
                name="background",
                physical_tag=11,
                material="SiO2",
                priority=0,
            )
        },
        layers={
            "core": MeshGroup(
                name="core",
                physical_tag=17,
                material="Si",
                priority=1,
            )
        },
        ports={
            "o1": PortMeshGroup(
                name="o1",
                physical_name="port_o1",
                physical_tag=23,
                layer="core",
                normal=(-1, 0, 0),
            )
        },
    )


def _config() -> FDTDConfig:
    snapshots = {
        name: resolve_material_snapshot(name, 1.55, {}) for name in ("Si", "SiO2")
    }
    return build_fdtd_config(
        _manifest(),
        snapshots,
        background_material="SiO2",
        center_wavelength_nm=1550,
        wavelength_halfspan_nm=50,
        num_wavelengths=3,
        default_port="o1",
        nanometers_per_cell=31.25,
        pml_cells=16,
        max_timesteps=None,
        energy_decay_fraction=1e-6,
        max_wall_seconds=3600,
    )


def test_config_uses_manifest_tags_and_rejects_extra_fields() -> None:
    config = _config()

    assert config.length_scale_meters == 1e-9
    assert config.geometry.volumes["background"].phys_group == 11
    assert config.geometry.layers["core"].phys_group == 17
    assert config.geometry.ports["o1"].phys_group == 23

    document = config.model_dump()
    document["unsupported"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        FDTDConfig.model_validate(document)

    document = config.model_dump()
    document["excitation"]["default_port"] = "missing"
    with pytest.raises(ValidationError, match="is not declared"):
        FDTDConfig.model_validate(document)


def test_config_serializes_canonical_cards_not_mutated_snapshots() -> None:
    silicon = resolve_material_snapshot("Si", 1.55, {})
    silica = resolve_material_snapshot("SiO2", 1.55, {})
    config = build_fdtd_config(
        _manifest(),
        {"Si": silicon, "SiO2": silica},
        background_material="SiO2",
        center_wavelength_nm=1550,
        wavelength_halfspan_nm=50,
        num_wavelengths=3,
        default_port="o1",
        nanometers_per_cell=31.25,
        pml_cells=16,
        max_timesteps=None,
        energy_decay_fraction=1e-6,
        max_wall_seconds=3600,
    )

    assert config.materials["Si"].dispersion is not None
    assert config.materials["Si"].dispersion.sellmeier is not None
    assert config.materials["Si"].refractive_index is None


def test_config_supports_gaussian_beam_and_fiber_monitor() -> None:
    beam = GaussianBeamConfig(
        region_min=(0, 0, 3000),
        region_max=(10000, 10000, 3000),
        aperture_normal="-z",
        propagation_direction=(0, 0, -1),
        e_polarization=(0, 1, 0),
        focal_point=(5000, 5000, 220),
        waist_radius=5000,
        refractive_index=1.444,
    )
    monitor = PlaneMonitorConfig(
        name="fiber",
        region_min=(0, 0, 3000),
        region_max=(10000, 10000, 3000),
        normal="+z",
        fiber_mode=FiberModeConfig(
            propagation_direction=(0, 0, 1),
            e_polarization=(0, 1, 0),
            focal_point=(5000, 5000, 220),
            waist_radius=5000,
            refractive_index=1.444,
        ),
    )
    snapshots = {
        name: resolve_material_snapshot(name, 1.55, {}) for name in ("Si", "SiO2")
    }

    config = build_fdtd_config(
        _manifest(),
        snapshots,
        background_material="SiO2",
        center_wavelength_nm=1550,
        wavelength_halfspan_nm=50,
        num_wavelengths=3,
        default_port=None,
        nanometers_per_cell=31.25,
        pml_cells=16,
        max_timesteps=None,
        energy_decay_fraction=1e-6,
        max_wall_seconds=3600,
        gaussian_beam=beam,
        monitors=[monitor],
    )

    assert config.excitation.type == "gaussian_beam"
    assert config.excitation.gaussian_beam == beam
    assert config.excitation.default_port is None
    assert config.monitors == [monitor]
