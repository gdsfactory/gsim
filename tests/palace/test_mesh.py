"""Tests for mesh configuration."""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.stack import LayerStack
from gsim.palace.mesh import MeshConfig
from gsim.palace.mesh import validation as mesh_validation
from gsim.palace.mesh.geometry import GeometryData, add_dielectrics


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
        assert config.curve_fit_mode == "line"
        assert config.curve_fit_layers == ["core", "core2"]
        assert config.curve_fit_tolerance_um == 0.0
        assert config.curve_fit_min_points == 8
        assert config.curve_fit_corner_angle_deg == 45.0
        assert config.high_order_elements is False
        assert config.high_order_order == 2
        assert config.high_order_optimize is True

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
        )
        assert config.refined_mesh_size == 3.0
        assert config.max_mesh_size == 200.0
        assert config.margin == 75.0

    def test_curve_fit_overrides(self):
        """Test custom curve-fit settings."""
        config = MeshConfig(
            curve_fit_mode="bspline",
            curve_fit_layers=["core"],
            curve_fit_tolerance_um=0.02,
            curve_fit_min_points=12,
            curve_fit_corner_angle_deg=30.0,
        )
        assert config.curve_fit_mode == "bspline"
        assert config.curve_fit_layers == ["core"]
        assert config.curve_fit_tolerance_um == 0.02
        assert config.curve_fit_min_points == 12
        assert config.curve_fit_corner_angle_deg == 30.0

    def test_high_order_overrides(self):
        """Test custom high-order mesh settings."""
        config = MeshConfig(
            high_order_elements=True,
            high_order_order=3,
            high_order_optimize=False,
        )
        assert config.high_order_elements is True
        assert config.high_order_order == 3
        assert config.high_order_optimize is False


def test_add_dielectrics_margin_applies_only_to_airlike(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_box(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_box", _fake_create_box
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": -2.0, "zmax": 0.5, "material": "SiO2"},
            {"name": "air", "zmin": 0.5, "zmax": 8.0, "material": "air"},
            {"name": "passivation", "zmin": 8.0, "zmax": 9.0, "material": "passive"},
        ],
        materials={
            "SiO2": {"type": "dielectric", "permittivity": 4.1},
            "air": {"type": "dielectric", "permittivity": 1.0},
            "passive": {"type": "dielectric", "permittivity": 6.6},
        },
    )

    add_dielectrics(
        _Kernel(), geometry, stack, margin_x=5.0, margin_y=7.0, air_margin=0.0
    )

    # SiO2 and passive keep original bbox; only air expands by margins.
    assert calls[0] == (0.0, 0.0, -2.0, 10.0, 20.0, 0.5)
    assert calls[1] == (-5.0, -7.0, 0.5, 15.0, 27.0, 8.0)
    assert calls[2] == (0.0, 0.0, 8.0, 10.0, 20.0, 9.0)


def test_add_dielectrics_explicit_airbox_z_extents(monkeypatch) -> None:
    calls: list[tuple[float, float, float, float, float, float]] = []

    def _fake_create_box(_kernel, xmin, ymin, zmin, xmax, ymax, zmax):
        calls.append((xmin, ymin, zmin, xmax, ymax, zmax))
        return len(calls)

    monkeypatch.setattr(
        "gsim.palace.mesh.geometry.gmsh_utils.create_box", _fake_create_box
    )

    class _Kernel:
        def synchronize(self) -> None:
            return

    geometry = GeometryData(polygons=[], bbox=(0.0, 0.0, 10.0, 20.0), layer_bboxes={})
    stack = LayerStack(
        dielectrics=[
            {"name": "oxide", "zmin": -2.0, "zmax": 0.5, "material": "SiO2"},
            {"name": "air", "zmin": 0.5, "zmax": 8.0, "material": "air"},
        ],
        materials={
            "SiO2": {"type": "dielectric", "permittivity": 4.1},
            "air": {"type": "dielectric", "permittivity": 1.0},
        },
    )

    tags = add_dielectrics(
        _Kernel(),
        geometry,
        stack,
        margin_x=5.0,
        margin_y=7.0,
        air_margin=0.0,
        airbox_z_above=100.0,
        airbox_z_below=100.0,
    )

    assert "airbox" in tags
    # Air layer is skipped when explicit airbox is built.
    assert list(tags.keys()) == ["SiO2", "airbox"]


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
