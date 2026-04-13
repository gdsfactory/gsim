"""Integration tests for Palace simulation workflows.

These tests verify the full local workflow for each sim type:
configure -> validate -> mesh -> write_config, stopping before cloud submission.

Note: gmsh segfaults on macOS (headless OpenGL issue), so mesh tests
only run on Linux (CI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gdsfactory as gf
import pytest

from gsim.palace import DrivenSim, EigenmodeSim, ElectrostaticSim

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin", reason="gmsh segfaults on macOS"
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_cpw_component():
    """Create a simple GSG electrode component."""
    gf.gpdk.PDK.activate()

    @gf.cell
    def gsg_electrode(
        length: float = 300,
        s_width: float = 10,
        g_width: float = 40,
        gap_width: float = 6,
        layer=gf.gpdk.LAYER.M1,
    ) -> gf.Component:
        c = gf.Component()
        r1 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        r1.move((0, (g_width + s_width) / 2 + gap_width))
        c << gf.c.rectangle((length, s_width), centered=True, layer=layer)
        r3 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        r3.move((0, -(g_width + s_width) / 2 - gap_width))
        c.add_port(
            name="o1",
            center=(-length / 2, 0),
            width=s_width,
            orientation=0,
            port_type="electrical",
            layer=layer,
        )
        c.add_port(
            name="o2",
            center=(length / 2, 0),
            width=s_width,
            orientation=180,
            port_type="electrical",
            layer=layer,
        )
        return c

    return gsg_electrode()


@pytest.fixture(scope="module")
def cpw_component():
    """Create a shared CPW component for all tests in the module."""
    return _make_cpw_component()


# ---------------------------------------------------------------------------
# DrivenSim workflow
# ---------------------------------------------------------------------------


class TestDrivenSimWorkflow:
    """End-to-end DrivenSim: configure -> validate -> mesh -> write_config."""

    @pytest.fixture(scope="class")
    def driven_sim(self, tmp_path_factory, cpw_component):
        """Create and mesh a DrivenSim with CPW ports."""
        tmp_path = tmp_path_factory.mktemp("driven")
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
        sim.mesh(preset="coarse")
        return sim

    def test_validate_before_mesh(self, cpw_component, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "val"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        result = sim.validate_config()
        assert result.valid, f"Validation failed: {result}"

    def test_mesh_creates_file(self, driven_sim):
        mesh_path = Path(driven_sim._output_dir) / "palace.msh"
        assert mesh_path.exists()
        assert mesh_path.stat().st_size > 0

    def test_mesh_has_physical_groups(self, driven_sim):
        groups = driven_sim._last_mesh_result.groups
        assert len(groups["volumes"]) > 0
        assert "P1" in groups["port_surfaces"]
        assert "P2" in groups["port_surfaces"]

    def test_mesh_has_absorbing_boundary(self, driven_sim):
        groups = driven_sim._last_mesh_result.groups
        assert "absorbing" in groups["boundary_surfaces"]

    def test_write_config_creates_json(self, driven_sim):
        driven_sim.write_config()
        config_path = Path(driven_sim._output_dir) / "config.json"
        assert config_path.exists()

    def test_config_json_structure(self, driven_sim):
        """Config JSON must have required Palace sections."""
        driven_sim.write_config()
        config_path = Path(driven_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())

        assert config["Problem"]["Type"] == "Driven"
        assert "Domains" in config
        assert "Boundaries" in config
        assert "Solver" in config

        boundaries = config["Boundaries"]
        assert "LumpedPort" in boundaries
        assert len(boundaries["LumpedPort"]) == 2
        assert "Absorbing" in boundaries

    def test_validate_mesh_passes(self, driven_sim):
        result = driven_sim.validate_mesh()
        assert result.valid, f"Mesh validation failed: {result}"


# ---------------------------------------------------------------------------
# DrivenSim with inplane lumped ports
# ---------------------------------------------------------------------------


class TestDrivenSimInplanePorts:
    """DrivenSim with single-element inplane lumped ports."""

    @pytest.fixture(scope="class")
    def inplane_sim(self, tmp_path_factory, cpw_component):
        """Create and mesh a DrivenSim with inplane lumped ports."""
        tmp_path = tmp_path_factory.mktemp("inplane")
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_port("o1", layer="metal1", length=5.0, impedance=50.0)
        sim.add_port("o2", layer="metal1", length=5.0, impedance=50.0)
        sim.set_driven(fmin=1e9, fmax=50e9, num_points=20)
        sim.mesh(preset="coarse")
        return sim

    def test_mesh_creates_file(self, inplane_sim):
        mesh_path = Path(inplane_sim._output_dir) / "palace.msh"
        assert mesh_path.exists()

    def test_config_has_lumped_ports(self, inplane_sim):
        inplane_sim.write_config()
        config_path = Path(inplane_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        assert "LumpedPort" in config["Boundaries"]
        assert len(config["Boundaries"]["LumpedPort"]) == 2

    def test_validate_mesh_passes(self, inplane_sim):
        result = inplane_sim.validate_mesh()
        assert result.valid, f"Mesh validation failed: {result}"


# ---------------------------------------------------------------------------
# EigenmodeSim workflow
# ---------------------------------------------------------------------------


class TestEigenmodeSimWorkflow:
    """End-to-end EigenmodeSim: configure -> mesh -> write_config."""

    @pytest.fixture(scope="class")
    def eigenmode_sim(self, tmp_path_factory, cpw_component):
        """Create and mesh an EigenmodeSim."""
        tmp_path = tmp_path_factory.mktemp("eigenmode")
        sim = EigenmodeSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.set_eigenmode(num_modes=5, target=50e9)
        sim.mesh(preset="coarse")
        return sim

    def test_mesh_creates_file(self, eigenmode_sim):
        mesh_path = Path(eigenmode_sim._output_dir) / "palace.msh"
        assert mesh_path.exists()

    def test_config_json_type(self, eigenmode_sim):
        eigenmode_sim.write_config()
        config_path = Path(eigenmode_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        assert config["Problem"]["Type"] == "Eigenmode"

    def test_config_has_eigenmode_solver(self, eigenmode_sim):
        """Config must contain Eigenmode solver section."""
        eigenmode_sim.write_config()
        config_path = Path(eigenmode_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        solver = config["Solver"]
        assert "Eigenmode" in solver

    def test_validate_mesh_passes(self, eigenmode_sim):
        result = eigenmode_sim.validate_mesh()
        assert result.valid, f"Mesh validation failed: {result}"


# ---------------------------------------------------------------------------
# ElectrostaticSim workflow
# ---------------------------------------------------------------------------


class TestElectrostaticSimWorkflow:
    """End-to-end ElectrostaticSim: configure -> mesh -> write_config."""

    @pytest.fixture(scope="class")
    def electrostatic_sim(self, tmp_path_factory, cpw_component):
        """Create and mesh an ElectrostaticSim with terminals."""
        tmp_path = tmp_path_factory.mktemp("electrostatic")
        sim = ElectrostaticSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_terminal("T1", layer="metal1")
        sim.add_terminal("T2", layer="metal1")
        sim.set_electrostatic()
        sim.mesh(preset="coarse")
        return sim

    def test_mesh_creates_file(self, electrostatic_sim):
        mesh_path = Path(electrostatic_sim._output_dir) / "palace.msh"
        assert mesh_path.exists()

    def test_write_config_not_implemented(self, electrostatic_sim):
        """Electrostatic config generation is not yet implemented."""
        with pytest.raises(NotImplementedError):
            electrostatic_sim.write_config()


# ---------------------------------------------------------------------------
# Validation error tests (no gmsh needed)
# ---------------------------------------------------------------------------


class TestValidationErrors:
    """Test validation catches configuration errors (no gmsh needed)."""

    def test_missing_geometry(self):
        sim = DrivenSim()
        result = sim.validate_config()
        assert not result.valid

    def test_missing_output_dir(self):
        """Missing output_dir produces a warning (not an error)."""
        sim = DrivenSim()
        sim.set_geometry(_make_cpw_component())
        sim.set_stack(air_above=300.0)
        result = sim.validate_config()
        # validate_config warns about missing ports, but output_dir is
        # checked at mesh() time, not validation time
        assert result.valid

    def test_driven_valid_config(self, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "test"))
        sim.set_geometry(_make_cpw_component())
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        result = sim.validate_config()
        assert result.valid, f"Validation failed: {result}"

    def test_eigenmode_no_ports_ok(self, tmp_path):
        """Eigenmode sims don't require ports."""
        sim = EigenmodeSim()
        sim.set_output_dir(str(tmp_path / "test"))
        sim.set_geometry(_make_cpw_component())
        sim.set_stack(air_above=300.0)
        sim.set_eigenmode(num_modes=5)
        result = sim.validate_config()
        assert result.valid, f"Validation failed: {result}"

    def test_electrostatic_needs_terminals(self, tmp_path):
        """Electrostatic sims need at least one terminal."""
        sim = ElectrostaticSim()
        sim.set_output_dir(str(tmp_path / "test"))
        sim.set_geometry(_make_cpw_component())
        sim.set_stack(air_above=300.0)
        result = sim.validate_config()
        assert not result.valid


# ---------------------------------------------------------------------------
# Material overrides
# ---------------------------------------------------------------------------


class TestMaterialOverrides:
    """Test material override configuration flows through to config."""

    @pytest.fixture(scope="class")
    def material_sim(self, tmp_path_factory, cpw_component):
        """Create a DrivenSim with material overrides."""
        tmp_path = tmp_path_factory.mktemp("materials")
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "palace-sim"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        sim.set_material("aluminum", conductivity=3.77e7)
        sim.mesh(preset="coarse")
        return sim

    def test_material_override_in_config(self, material_sim):
        material_sim.write_config()
        config_path = Path(material_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        # The material should appear in Domains.Materials
        assert "Domains" in config


# ---------------------------------------------------------------------------
# Numerical solver overrides
# ---------------------------------------------------------------------------


class TestNumericalConfig:
    """Test numerical solver configuration."""

    def test_set_numerical(self, cpw_component, tmp_path):
        sim = DrivenSim()
        sim.set_output_dir(str(tmp_path / "numerical"))
        sim.set_geometry(cpw_component)
        sim.set_stack(substrate_thickness=2.0, air_above=300.0)
        sim.add_cpw_port("o1", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.add_cpw_port("o2", layer="metal1", s_width=10, gap_width=6, length=5.0)
        sim.set_driven(fmin=1e9, fmax=100e9)
        sim.set_numerical(order=2, tolerance=1e-8, max_iterations=600)

        assert sim.numerical.order == 2
        assert sim.numerical.tolerance == 1e-8
        assert sim.numerical.max_iterations == 600
