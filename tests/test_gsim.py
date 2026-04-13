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
    """Test material property lookups."""
    from gsim.common.stack import (
        get_material_properties,
        material_is_conductor,
        material_is_dielectric,
    )

    # Test conductor
    aluminum = get_material_properties("aluminum")
    assert aluminum is not None
    assert aluminum.type == "conductor"
    assert material_is_conductor("aluminum")
    assert not material_is_dielectric("aluminum")

    # Test dielectric
    sio2 = get_material_properties("SiO2")
    assert sio2 is not None
    assert sio2.type == "dielectric"
    assert material_is_dielectric("SiO2")
    assert not material_is_conductor("SiO2")

    # Test unknown material
    unknown = get_material_properties("nonexistent_material_xyz")
    assert unknown is None
