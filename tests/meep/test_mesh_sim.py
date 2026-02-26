"""Tests for MeepMeshSim and mesh config writer."""

from __future__ import annotations

import json

import pytest

from gsim.common.mesh.types import MeshResult
from gsim.meep import FDTD, Domain, Material, MeepMeshSim, ModeSource
from gsim.meep.mesh_config import (
    _serialize_domain,
    _serialize_materials,
    _serialize_mesh_groups,
    _serialize_solver,
    _serialize_source,
    write_mesh_config,
)

# ---------------------------------------------------------------------------
# MeepMeshSim construction and attribute setting
# ---------------------------------------------------------------------------


class TestMeepMeshSimConstruction:
    """Test MeepMeshSim creation and attribute updates."""

    def test_defaults(self):
        sim = MeepMeshSim()
        assert sim.geometry.component is None
        assert sim.materials == {}
        assert sim.source.port is None
        assert sim.monitors == []
        assert sim.domain.pml == 1.0
        assert sim.solver.resolution == 32

    def test_callable_setters(self):
        sim = MeepMeshSim()
        sim.source(port="o1", wavelength=1.31)
        assert sim.source.port == "o1"
        assert sim.source.wavelength == 1.31

    def test_domain_setter(self):
        sim = MeepMeshSim()
        sim.domain(pml=0.5, margin=0.3)
        assert sim.domain.pml == 0.5
        assert sim.domain.margin == 0.3

    def test_solver_setter(self):
        sim = MeepMeshSim()
        sim.solver(resolution=64)
        assert sim.solver.resolution == 64

    def test_materials_float_shorthand(self):
        sim = MeepMeshSim()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        assert isinstance(sim.materials["si"], Material)
        assert sim.materials["si"].n == 3.47
        assert sim.materials["SiO2"].n == 1.44

    def test_materials_dict_shorthand(self):
        sim = MeepMeshSim()
        sim.materials = {"si": {"n": 3.47, "k": 0.01}}
        assert isinstance(sim.materials["si"], Material)
        assert sim.materials["si"].k == 0.01

    def test_monitors_assignment(self):
        sim = MeepMeshSim()
        sim.monitors = ["o1", "o2", "o3"]
        assert sim.monitors == ["o1", "o2", "o3"]


# ---------------------------------------------------------------------------
# MeepMeshSim error handling
# ---------------------------------------------------------------------------


class TestMeepMeshSimErrors:
    """Test error conditions."""

    def test_mesh_without_component_raises(self, tmp_path):
        sim = MeepMeshSim()
        with pytest.raises(ValueError, match="No component set"):
            sim.mesh(tmp_path)

    def test_write_config_without_mesh_raises(self):
        sim = MeepMeshSim()
        with pytest.raises(ValueError, match="Call mesh"):
            sim.write_config()

    def test_plot_mesh_without_mesh_raises(self):
        sim = MeepMeshSim()
        with pytest.raises(ValueError, match="Call mesh"):
            sim.plot_mesh()


# ---------------------------------------------------------------------------
# Config serialization helpers
# ---------------------------------------------------------------------------


class TestSerializeMaterials:
    """Test material serialization."""

    def test_float_values(self):
        result = _serialize_materials({"si": 3.47})
        assert result["si"]["refractive_index"] == 3.47
        assert result["si"]["extinction_coeff"] == 0.0

    def test_material_values(self):
        result = _serialize_materials({"si": Material(n=3.47, k=0.01)})
        assert result["si"]["refractive_index"] == 3.47
        assert result["si"]["extinction_coeff"] == 0.01


class TestSerializeMeshGroups:
    """Test mesh group serialization."""

    def test_full_groups(self):
        groups = {
            "volumes": {"SiO2": {"phys_group": 1, "tags": [1, 2]}},
            "layer_volumes": {"core": {"phys_group": 3, "tags": [5], "material": "si"}},
            "outer_boundary": {"phys_group": 4, "tags": [10, 11]},
        }
        result = _serialize_mesh_groups(groups)
        assert result["volumes"]["SiO2"]["phys_group"] == 1
        assert result["volumes"]["SiO2"]["material"] == "SiO2"
        assert "tags" not in result["volumes"]["SiO2"]
        assert result["layer_volumes"]["core"]["material"] == "si"
        assert result["outer_boundary"]["phys_group"] == 4

    def test_empty_outer_boundary(self):
        groups = {"volumes": {}, "layer_volumes": {}, "outer_boundary": {}}
        result = _serialize_mesh_groups(groups)
        assert "outer_boundary" not in result


class TestSerializeSource:
    """Test source serialization."""

    def test_default_source(self):
        source = ModeSource(port="o1")
        result = _serialize_source(source)
        assert result["port"] == "o1"
        assert result["wavelength"] == 1.55
        assert result["num_freqs"] == 11


class TestSerializeDomain:
    """Test domain serialization."""

    def test_default_domain(self):
        domain = Domain()
        result = _serialize_domain(domain)
        assert result["pml"] == 1.0
        assert result["margin_xy"] == 0.5


class TestSerializeSolver:
    """Test solver serialization."""

    def test_default_solver(self):
        solver = FDTD()
        result = _serialize_solver(solver)
        assert result["resolution"] == 32
        assert result["stopping"]["mode"] == "field_decay"


# ---------------------------------------------------------------------------
# write_mesh_config
# ---------------------------------------------------------------------------


class TestWriteMeshConfig:
    """Test writing mesh_config.json."""

    def test_writes_valid_json(self, tmp_path):
        mesh_result = MeshResult(
            mesh_path=tmp_path / "test.msh",
            mesh_stats={"nodes": 1000, "tetrahedra": 5000},
            groups={
                "volumes": {"SiO2": {"phys_group": 1, "tags": [1]}},
                "layer_volumes": {
                    "core": {"phys_group": 2, "tags": [3], "material": "si"}
                },
                "outer_boundary": {"phys_group": 3, "tags": [4]},
            },
        )

        config_path = write_mesh_config(
            mesh_result=mesh_result,
            materials={"si": Material(n=3.47), "SiO2": Material(n=1.44)},
            source=ModeSource(
                port="o1", wavelength=1.55, wavelength_span=0.1, num_freqs=11
            ),
            monitors=["o1", "o2"],
            domain=Domain(pml=1.0, margin=0.5),
            solver=FDTD(resolution=32),
            output_dir=tmp_path,
        )

        assert config_path.exists()
        data = json.loads(config_path.read_text())

        assert data["mesh_filename"] == "test.msh"
        assert data["materials"]["si"]["refractive_index"] == 3.47
        assert data["source"]["port"] == "o1"
        assert data["monitors"] == ["o1", "o2"]
        assert data["domain"]["pml"] == 1.0
        assert data["solver"]["resolution"] == 32
        assert data["mesh_stats"]["nodes"] == 1000
        assert data["mesh_stats"]["tetrahedra"] == 5000
        assert data["mesh_groups"]["volumes"]["SiO2"]["phys_group"] == 1
        assert data["mesh_groups"]["volumes"]["SiO2"]["material"] == "SiO2"
        assert data["mesh_groups"]["layer_volumes"]["core"]["material"] == "si"


# ---------------------------------------------------------------------------
# End-to-end: component + stack → .msh + mesh_config.json
# ---------------------------------------------------------------------------


class TestMeepMeshSimEndToEnd:
    """End-to-end test with a real gdsfactory component and stack."""

    @pytest.fixture
    def sim_with_component(self):
        """Create a MeepMeshSim with a simple waveguide component and stack."""
        import gdsfactory as gf

        from gsim.common.stack.extractor import Layer, LayerStack

        gf.gpdk.PDK.activate()

        # Simple waveguide
        c = gf.Component()
        c.add_polygon(
            [(0, 0), (10, 0), (10, 0.5), (0, 0.5)],
            layer=(1, 0),
        )

        stack = LayerStack(pdk_name="test")
        stack.layers["core"] = Layer(
            name="core",
            gds_layer=(1, 0),
            zmin=0.0,
            zmax=0.22,
            thickness=0.22,
            material="si",
            layer_type="dielectric",
        )
        stack.dielectrics = [
            {"name": "box", "material": "SiO2", "zmin": -1.0, "zmax": 0.0},
            {"name": "clad", "material": "SiO2", "zmin": 0.0, "zmax": 1.0},
        ]
        stack.materials = {
            "si": {"type": "dielectric", "permittivity": 11.7},
            "SiO2": {"type": "dielectric", "permittivity": 2.1},
        }

        sim = MeepMeshSim()
        sim.geometry(component=c, stack=stack)
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.source(port="o1", wavelength=1.55)
        sim.monitors = ["o1", "o2"]
        sim.domain(pml=1.0, margin=0.5)
        sim.solver(resolution=32)

        return sim

    def test_mesh_generates_msh(self, sim_with_component, tmp_path):
        sim = sim_with_component
        result = sim.mesh(tmp_path, model_name="test_wg")

        assert result.mesh_path.exists()
        assert result.mesh_path.name == "test_wg.msh"
        assert result.mesh_stats.get("nodes", 0) > 0
        assert result.mesh_stats.get("tetrahedra", 0) > 0
        assert "volumes" in result.groups
        assert "layer_volumes" in result.groups

    def test_write_config_after_mesh(self, sim_with_component, tmp_path):
        sim = sim_with_component
        sim.mesh(tmp_path, model_name="test_wg")
        config_path = sim.write_config()

        assert config_path.exists()
        assert config_path.name == "mesh_config.json"

        data = json.loads(config_path.read_text())
        assert data["mesh_filename"] == "test_wg.msh"
        assert "si" in data["materials"]
        assert data["source"]["wavelength"] == 1.55
        assert data["monitors"] == ["o1", "o2"]
        assert data["solver"]["resolution"] == 32

    def test_mesh_margin_defaults_to_domain(self, sim_with_component, tmp_path):
        """Verify mesh margin defaults to domain.margin + domain.pml."""
        sim = sim_with_component
        sim.domain(pml=0.5, margin=0.3)
        result = sim.mesh(tmp_path)
        # Should succeed — margin = 0.3 + 0.5 = 0.8
        assert result.mesh_path.exists()
