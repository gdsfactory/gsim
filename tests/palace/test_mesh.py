"""Tests for mesh generation pipeline."""

from __future__ import annotations

from gsim.palace.mesh import GroundPlane, MeshConfig, MeshPreset


class TestMeshConfig:
    """Test MeshConfig class."""

    def test_default_config(self):
        """Test default MeshConfig values."""
        config = MeshConfig()
        assert config.refined_mesh_size == 5.0
        assert config.max_mesh_size == 300.0
        assert config.margin == 50.0
        assert config.fmax == 100e9
        assert config.show_gui is False
        assert config.boundary_conditions is not None
        assert len(config.boundary_conditions) == 6

    def test_coarse_preset(self):
        """Test coarse mesh preset."""
        config = MeshConfig.coarse()
        assert config.refined_mesh_size == 10.0
        assert config.max_mesh_size == 600.0
        assert config.cells_per_wavelength == 5

    def test_default_preset(self):
        """Test default mesh preset."""
        config = MeshConfig.default()
        assert config.refined_mesh_size == 5.0
        assert config.max_mesh_size == 300.0
        assert config.cells_per_wavelength == 10

    def test_fine_preset(self):
        """Test fine mesh preset."""
        config = MeshConfig.fine()
        assert config.refined_mesh_size == 2.0
        assert config.max_mesh_size == 70.0
        assert config.cells_per_wavelength == 20

    def test_preset_with_overrides(self):
        """Test preset with custom overrides."""
        config = MeshConfig.coarse(margin=100.0, fmax=50e9)
        assert config.refined_mesh_size == 10.0  # From preset
        assert config.margin == 100.0  # Override
        assert config.fmax == 50e9  # Override

    def test_custom_config(self):
        """Test fully custom config."""
        config = MeshConfig(
            refined_mesh_size=3.0,
            max_mesh_size=200.0,
            margin=75.0,
            air_above=150.0,
        )
        assert config.refined_mesh_size == 3.0
        assert config.max_mesh_size == 200.0
        assert config.margin == 75.0
        assert config.air_above == 150.0


class TestGroundPlane:
    """Test GroundPlane class."""

    def test_ground_plane_defaults(self):
        """Test GroundPlane default values."""
        gp = GroundPlane(layer_name="metal1")
        assert gp.layer_name == "metal1"
        assert gp.oversize == 50.0

    def test_ground_plane_custom(self):
        """Test GroundPlane with custom values."""
        gp = GroundPlane(layer_name="metal2", oversize=100.0)
        assert gp.layer_name == "metal2"
        assert gp.oversize == 100.0


class TestMeshPreset:
    """Test MeshPreset enum."""

    def test_preset_values(self):
        """Test preset enum values."""
        assert MeshPreset.COARSE.value == "coarse"
        assert MeshPreset.DEFAULT.value == "default"
        assert MeshPreset.FINE.value == "fine"
