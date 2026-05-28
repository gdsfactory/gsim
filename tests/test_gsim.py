"""Basic tests for gsim package."""

from __future__ import annotations


def test_mesh_config_presets():
    """Test that MeshConfig presets work."""
    from gsim.palace.mesh import MeshConfig

    coarse = MeshConfig.coarse()
    assert coarse.refined_mesh_size == 10.0
    assert coarse.max_mesh_size == 600.0

    default = MeshConfig.default()
    assert default.refined_mesh_size == 5.0
    assert default.max_mesh_size == 300.0

    fine = MeshConfig.fine()
    assert fine.refined_mesh_size == 2.0
    assert fine.max_mesh_size == 70.0


def test_material_properties():
    """Test material property lookups and frequency-aware behavior."""
    from gsim.common.stack import (
        get_material_properties,
        resolve_material_at_wavelength,
    )

    aluminum = get_material_properties("aluminum")
    assert aluminum is not None
    assert aluminum.conductivity is not None

    resolved_al_rf = resolve_material_at_wavelength("aluminum", 60000)
    assert resolved_al_rf is not None
    assert resolved_al_rf.behavior == "conductive"

    sio2 = get_material_properties("SiO2")
    assert sio2 is not None
    assert sio2.permittivity == 4.1

    resolved_sio2 = resolve_material_at_wavelength("SiO2", 1.55)
    assert resolved_sio2 is not None
    assert resolved_sio2.behavior == "dielectric"

    unknown = get_material_properties("nonexistent_material_xyz")
    assert unknown is None
