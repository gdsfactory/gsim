"""Tests for gsim.common.mesh module."""

from __future__ import annotations

import pytest

from gsim.common.mesh.geometry import GeometryData, get_layer_info
from gsim.common.mesh.types import MeshResult


class TestGeometryData:
    """Test GeometryData dataclass."""

    def test_create_empty(self):
        gd = GeometryData(polygons=[], bbox=(0, 0, 10, 10), layer_bboxes={})
        assert gd.polygons == []
        assert gd.bbox == (0, 0, 10, 10)
        assert gd.layer_bboxes == {}

    def test_create_with_polygons(self):
        polys = [(1, [0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [])]
        gd = GeometryData(
            polygons=polys,
            bbox=(0, 0, 1, 1),
            layer_bboxes={1: [0, 0, 1, 1]},
        )
        assert len(gd.polygons) == 1
        assert gd.polygons[0][0] == 1
        assert 1 in gd.layer_bboxes


class TestMeshResult:
    """Test MeshResult dataclass."""

    def test_create_minimal(self, tmp_path):
        msh = tmp_path / "test.msh"
        result = MeshResult(mesh_path=msh)
        assert result.mesh_path == msh
        assert result.mesh_stats == {}
        assert result.groups == {}

    def test_create_with_data(self, tmp_path):
        result = MeshResult(
            mesh_path=tmp_path / "test.msh",
            mesh_stats={"nodes": 100},
            groups={"volumes": {"SiO2": {"phys_group": 1, "tags": [1, 2]}}},
        )
        assert result.mesh_stats["nodes"] == 100
        assert "SiO2" in result.groups["volumes"]


class TestGetLayerInfo:
    """Test get_layer_info with a mock LayerStack."""

    @pytest.fixture
    def mock_stack(self):
        """Create a minimal LayerStack-like object for testing."""
        from gsim.common.stack.extractor import Layer, LayerStack

        stack = LayerStack(pdk_name="test")
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(10, 0),
            zmin=0.0,
            zmax=0.5,
            thickness=0.5,
            material="copper",
            layer_type="conductor",
        )
        stack.layers["oxide"] = Layer(
            name="oxide",
            gds_layer=(20, 0),
            zmin=-1.0,
            zmax=0.0,
            thickness=1.0,
            material="SiO2",
            layer_type="dielectric",
        )
        return stack

    def test_found(self, mock_stack):
        info = get_layer_info(mock_stack, 10)
        assert info is not None
        assert info["name"] == "metal1"
        assert info["material"] == "copper"
        assert info["type"] == "conductor"
        assert info["zmin"] == 0.0
        assert info["thickness"] == 0.5

    def test_not_found(self, mock_stack):
        info = get_layer_info(mock_stack, 999)
        assert info is None

    def test_dielectric_layer(self, mock_stack):
        info = get_layer_info(mock_stack, 20)
        assert info is not None
        assert info["name"] == "oxide"
        assert info["type"] == "dielectric"


class TestExtractGeometry:
    """Test extract_geometry with a real gdsfactory component."""

    @pytest.fixture
    def simple_component_and_stack(self):
        """Create a simple gdsfactory component with matching stack."""
        import gdsfactory as gf

        from gsim.common.stack.extractor import Layer, LayerStack

        gf.gpdk.PDK.activate()
        c = gf.Component()
        c.add_polygon(
            [(0, 0), (10, 0), (10, 5), (0, 5)],
            layer=(1, 0),
        )

        stack = LayerStack(pdk_name="test")
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=0.5,
            thickness=0.5,
            material="copper",
            layer_type="conductor",
        )
        return c, stack

    def test_extract_geometry(self, simple_component_and_stack):
        from gsim.common.mesh.geometry import extract_geometry

        c, stack = simple_component_and_stack
        gd = extract_geometry(c, stack)

        assert len(gd.polygons) == 1
        layernum, pts_x, _pts_y, holes = gd.polygons[0]
        assert layernum == 1
        assert len(pts_x) >= 3
        assert holes == []
        # Bbox should match the polygon (in um)
        assert gd.bbox[0] == pytest.approx(0.0, abs=0.01)
        assert gd.bbox[2] == pytest.approx(10.0, abs=0.01)  # 10um


class TestAddLayerVolumes:
    """Test add_layer_volumes with gmsh."""

    @pytest.fixture
    def geometry_and_stack(self):
        """Create geometry data and stack for testing."""
        from gsim.common.stack.extractor import Layer, LayerStack

        gd = GeometryData(
            polygons=[
                (1, [0.0, 10.0, 10.0, 0.0], [0.0, 0.0, 5.0, 5.0], []),
            ],
            bbox=(0, 0, 10, 5),
            layer_bboxes={1: [0, 0, 10, 5]},
        )
        stack = LayerStack(pdk_name="test")
        stack.layers["metal1"] = Layer(
            name="metal1",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=0.5,
            thickness=0.5,
            material="copper",
            layer_type="conductor",
        )
        return gd, stack

    def test_add_layer_volumes(self, geometry_and_stack):
        import gmsh

        from gsim.common.mesh.geometry import add_layer_volumes

        gd, stack = geometry_and_stack

        gmsh.initialize()
        gmsh.model.add("test_volumes")
        kernel = gmsh.model.occ

        try:
            result = add_layer_volumes(kernel, gd, stack)
            assert "metal1" in result
            assert len(result["metal1"]) == 1
            assert isinstance(result["metal1"][0], int)
        finally:
            gmsh.clear()
            gmsh.finalize()


class TestCollectMeshStats:
    """Test collect_mesh_stats."""

    def test_collect_stats_keys(self):
        import gmsh

        from gsim.common.mesh.generator import collect_mesh_stats

        gmsh.initialize()
        gmsh.model.add("test_stats")
        kernel = gmsh.model.occ

        try:
            # Create a simple box and mesh it
            kernel.addBox(0, 0, 0, 10, 10, 10)
            kernel.synchronize()
            gmsh.model.mesh.generate(3)

            stats = collect_mesh_stats()
            assert "nodes" in stats
            assert "elements" in stats
            assert "tetrahedra" in stats
            assert "bbox" in stats
            assert stats["nodes"] > 0
            assert stats["tetrahedra"] > 0
        finally:
            gmsh.clear()
            gmsh.finalize()
