"""Tests for mesh configuration."""

from __future__ import annotations

import numpy as np
import pytest

from gsim.palace.mesh import MeshConfig
from gsim.palace.mesh import validation as mesh_validation
from gsim.palace.models.mesh import MeshConfig as ModelMeshConfig


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

    def test_graded_preset(self):
        """Test graded mesh preset (default sizes + refine_from_curves)."""
        config = MeshConfig.graded()
        assert config.refined_mesh_size == 5.0
        assert config.max_mesh_size == 300.0
        assert config.cells_per_wavelength == 10
        assert config.refine_from_curves is True

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
            airbox_margin=150.0,
        )
        assert config.refined_mesh_size == 3.0
        assert config.max_mesh_size == 200.0
        assert config.margin == 75.0
        assert config.airbox_margin == 150.0

    def test_refine_from_curves_alias_setter(self):
        """Legacy refine_from_curves alias setter updates new field."""
        config = MeshConfig()
        config.refine_from_curves = True
        assert config.refine_near_conductor_curves is True


class TestModelMeshConfig:
    """Test Pydantic mesh model compatibility aliases."""

    def test_model_refine_from_curves_alias_setter(self):
        """Model alias property setter updates the new field."""
        config = ModelMeshConfig()
        config.refine_from_curves = True
        assert config.refine_near_conductor_curves is True


class TestValidationHelpers:
    """Test helper routines used by mesh validation."""

    def test_parse_direction(self):
        """Direction parser returns normalized vectors and rejects unknown keys."""
        vec = mesh_validation._parse_direction("+X")
        assert np.allclose(vec, np.array([1.0, 0.0, 0.0]))

        with pytest.raises(ValueError, match="Unknown port direction"):
            mesh_validation._parse_direction("north")

    def test_perp_dist(self):
        """Perpendicular distance to an axis-aligned line is computed correctly."""
        v = np.array([0.0, 3.0, 4.0])
        origin = np.zeros(3)
        normals = [np.array([1.0, 0.0, 0.0])]
        dist = mesh_validation._perp_dist(v, normals, origin)
        assert dist == pytest.approx(5.0)

    def test_palace_obb_planar_rectangle(self):
        """OBB helper handles a simple planar rectangle point cloud."""
        pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        _center, axes, planar = mesh_validation._palace_obb(pts)
        lengths = [2.0 * float(np.linalg.norm(ax)) for ax in axes]
        assert planar is True
        assert max(lengths) == pytest.approx(2.0)
        assert min(lengths) == pytest.approx(0.0)
