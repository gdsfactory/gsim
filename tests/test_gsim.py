"""Basic tests for gsim package."""

from __future__ import annotations


def test_package_import():
    """Test that gsim package can be imported."""
    import gsim

    assert gsim is not None
    assert hasattr(gsim, "__version__")


def test_package_version():
    """Test that package version is accessible."""
    import gsim

    assert gsim.__version__ == "0.0.0"


def test_palace_submodule_import():
    """Test that palace submodule can be imported."""
    from gsim import palace

    assert palace is not None


def test_palace_main_exports():
    """Test that main palace API functions are available."""
    from gsim.palace import (
        configure_cpw_port,
        configure_inplane_port,
        configure_via_port,
        extract_ports,
        generate_mesh,
        get_stack,
    )

    # Just verify they're callable/importable
    assert callable(get_stack)
    assert callable(configure_inplane_port)
    assert callable(configure_via_port)
    assert callable(configure_cpw_port)
    assert callable(extract_ports)
    assert callable(generate_mesh)


def test_palace_classes_import():
    """Test that palace classes can be imported."""
    from gsim.palace import (
        Layer,
        LayerStack,
        MaterialProperties,
        MeshConfig,
        MeshResult,
        PalacePort,
        PortGeometry,
        PortType,
    )

    # Verify classes exist
    assert LayerStack is not None
    assert Layer is not None
    assert PalacePort is not None
    assert PortType is not None
    assert PortGeometry is not None
    assert MeshConfig is not None
    assert MeshResult is not None
    assert MaterialProperties is not None


def test_palace_mesh_import():
    """Test that palace.mesh submodule can be imported."""
    from gsim.palace.mesh import GroundPlane, MeshConfig, MeshPreset

    assert MeshConfig is not None
    assert MeshPreset is not None
    assert GroundPlane is not None


def test_palace_ports_import():
    """Test that palace.ports submodule can be imported."""
    from gsim.palace.ports import (
        configure_cpw_port,
        configure_inplane_port,
        configure_via_port,
        extract_ports,
    )

    assert callable(configure_inplane_port)
    assert callable(configure_via_port)
    assert callable(configure_cpw_port)
    assert callable(extract_ports)


def test_palace_stack_import():
    """Test that palace.stack submodule can be imported."""
    from gsim.palace.stack import (
        get_material_properties,
        get_stack,
        load_stack_yaml,
        material_is_conductor,
        material_is_dielectric,
    )

    assert callable(get_stack)
    assert callable(load_stack_yaml)
    assert callable(get_material_properties)
    assert callable(material_is_conductor)
    assert callable(material_is_dielectric)


def test_mesh_config_presets():
    """Test that MeshConfig presets work."""
    from gsim.palace.mesh import MeshConfig

    # Test presets
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
    from gsim.palace.stack import (
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
