"""Integration tests for mesh generation with IHP-style components.

These tests verify that mesh generation produces correct physical groups
and Palace config without actually running Palace.

Note: gmsh segfaults on macOS (headless OpenGL issue), so these tests
only run on Linux (CI).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import gdsfactory as gf
import pytest

from gsim.common import Layer, LayerStack
from gsim.palace import DrivenSim

pytestmark = pytest.mark.skipif(
    sys.platform == "darwin", reason="gmsh segfaults on macOS"
)


def _make_cpw_component():
    """Create an IHP-style GSG electrode component."""
    gf.gpdk.PDK.activate()

    @gf.cell
    def gsg_electrode(
        length: float = 500,
        s_width: float = 10,
        g_width: float = 50,
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


def _make_sim(component, tmp_path, planar_conductors=False, layer="topmetal2"):
    """Create, configure, and mesh a DrivenSim."""
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(component)
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    sim.add_cpw_port("o1", layer=layer, s_width=10, gap_width=6, length=5.0)
    sim.add_cpw_port("o2", layer=layer, s_width=10, gap_width=6, length=5.0)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
    sim.mesh(preset="coarse", planar_conductors=planar_conductors)
    return sim


@pytest.fixture(scope="module")
def volumetric_sim(tmp_path_factory):
    """Mesh once with volumetric conductors, share across tests."""
    tmp_path = tmp_path_factory.mktemp("volumetric")
    component = _make_cpw_component()
    return _make_sim(component, tmp_path, planar_conductors=False, layer="metal1")


@pytest.fixture(scope="module")
def planar_sim(tmp_path_factory):
    """Mesh once with planar conductors, share across tests."""
    tmp_path = tmp_path_factory.mktemp("planar")
    component = _make_cpw_component()
    return _make_sim(component, tmp_path, planar_conductors=True, layer="metal1")


class TestCPWMeshVolumetricConductors:
    """Test mesh generation with volumetric (non-planar) conductors."""

    def test_mesh_has_conductor_surfaces(self, volumetric_sim):
        """Volumetric conductors must produce metal_xy and metal_z groups."""
        groups = volumetric_sim._last_mesh_result.groups
        conductor_names = list(groups["conductor_surfaces"].keys())
        assert any("xy" in name for name in conductor_names), (
            f"No _xy conductor surfaces found. Got: {conductor_names}"
        )
        assert any("z" in name for name in conductor_names), (
            f"No _z conductor surfaces found. Got: {conductor_names}"
        )

    def test_mesh_has_volumes(self, volumetric_sim):
        """Dielectric volumes must be present."""
        groups = volumetric_sim._last_mesh_result.groups
        assert len(groups["volumes"]) > 0, "No dielectric volumes found"

    def test_mesh_has_port_surfaces(self, volumetric_sim):
        """CPW ports must produce port surface groups."""
        groups = volumetric_sim._last_mesh_result.groups
        assert "P1" in groups["port_surfaces"], "Port P1 not found"
        assert "P2" in groups["port_surfaces"], "Port P2 not found"
        for port_name in ("P1", "P2"):
            port = groups["port_surfaces"][port_name]
            assert port["type"] == "cpw"
            assert len(port["elements"]) >= 2, f"{port_name} should have >=2 elements"

    def test_mesh_has_absorbing_boundary(self, volumetric_sim):
        """Absorbing boundary surfaces must be present."""
        groups = volumetric_sim._last_mesh_result.groups
        assert "absorbing" in groups["boundary_surfaces"], "No absorbing boundary"

    def test_config_json_valid(self, volumetric_sim):
        """Generated config.json must have required Palace sections."""
        volumetric_sim.write_config()
        config_path = Path(volumetric_sim._output_dir) / "config.json"
        assert config_path.exists()

        config = json.loads(config_path.read_text())
        assert config["Problem"]["Type"] == "Driven"
        assert "Domains" in config
        boundaries = config["Boundaries"]
        assert "Conductivity" in boundaries, "Missing Conductivity boundary"
        assert len(boundaries["Conductivity"]) > 0, "No conductor entries"
        assert "LumpedPort" in boundaries, "Missing LumpedPort boundary"
        assert len(boundaries["LumpedPort"]) == 2, "Expected 2 lumped ports"
        assert "Absorbing" in boundaries, "Missing Absorbing boundary"


class TestCPWMeshPlanarConductors:
    """Test mesh generation with planar (PEC) conductors."""

    def test_mesh_has_pec_surfaces(self, planar_sim):
        """Planar conductors must produce PEC surface groups."""
        groups = planar_sim._last_mesh_result.groups
        assert len(groups["pec_surfaces"]) > 0, "No PEC surfaces found"

    def test_config_json_has_pec(self, planar_sim):
        """Config must include PEC boundary when using planar conductors."""
        planar_sim.write_config()
        config_path = Path(planar_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        assert "PEC" in config["Boundaries"], "Missing PEC boundary"


# ---------------------------------------------------------------------------
# QPDK-style: zero-thickness superconductor on sapphire
# ---------------------------------------------------------------------------

QPDK_LAYER = (2, 0)
SUBSTRATE_LAYER = (1, 0)
VACUUM_LAYER = (3, 0)


def _make_qpdk_component():
    """Create a QPDK-style CPW with zero-thickness conductor on sapphire."""
    gf.gpdk.PDK.activate()

    @gf.cell
    def qpdk_cpw(
        length: float = 500,
        s_width: float = 10,
        g_width: float = 50,
        gap_width: float = 6,
    ) -> gf.Component:
        c = gf.Component()
        # Ground plane with gap (simplified: two ground strips + signal)
        r1 = c << gf.c.rectangle((length, g_width), centered=True, layer=QPDK_LAYER)
        r1.move((0, (g_width + s_width) / 2 + gap_width))
        c << gf.c.rectangle((length, s_width), centered=True, layer=QPDK_LAYER)
        r3 = c << gf.c.rectangle((length, g_width), centered=True, layer=QPDK_LAYER)
        r3.move((0, -(g_width + s_width) / 2 - gap_width))

        # Substrate and vacuum full-area rectangles
        bbox_h = 2 * g_width + 2 * gap_width + s_width
        for layer in (SUBSTRATE_LAYER, VACUUM_LAYER):
            c << gf.c.rectangle((length, bbox_h), centered=True, layer=layer)

        c.add_port(
            name="o1",
            center=(-length / 2, 0),
            width=s_width,
            orientation=0,
            port_type="electrical",
            layer=QPDK_LAYER,
        )
        c.add_port(
            name="o2",
            center=(length / 2, 0),
            width=s_width,
            orientation=180,
            port_type="electrical",
            layer=QPDK_LAYER,
        )
        return c

    return qpdk_cpw()


def _make_qpdk_stack():
    """Build a sapphire + superconductor + vacuum stack (QPDK-style)."""
    substrate_thickness = 500
    stack = LayerStack()
    stack.layers["sapphire"] = Layer(
        name="sapphire",
        gds_layer=SUBSTRATE_LAYER,
        zmin=0,
        zmax=substrate_thickness,
        thickness=substrate_thickness,
        material="sapphire",
        layer_type="dielectric",
        mesh_resolution="coarse",
    )
    stack.layers["superconductor"] = Layer(
        name="superconductor",
        gds_layer=QPDK_LAYER,
        zmin=substrate_thickness,
        zmax=substrate_thickness,
        thickness=0,
        material="niobium",
        layer_type="conductor",
        mesh_resolution="fine",
    )
    stack.layers["vacuum"] = Layer(
        name="vacuum",
        gds_layer=VACUUM_LAYER,
        zmin=substrate_thickness,
        zmax=substrate_thickness + 500,
        thickness=500,
        material="vacuum",
        layer_type="dielectric",
        mesh_resolution="coarse",
    )
    stack.dielectrics = [
        {"material": "sapphire", "zmin": 0, "zmax": substrate_thickness},
        {
            "material": "vacuum",
            "zmin": substrate_thickness,
            "zmax": substrate_thickness + 500,
        },
    ]
    stack.materials = {
        "sapphire": {
            "type": "dielectric",
            "permittivity": 9.3,
            "loss_tangent": 0.0,
        },
        "vacuum": {"type": "dielectric", "permittivity": 1.0, "loss_tangent": 0.0},
        "niobium": {"type": "conductor", "conductivity": 0.0},
    }
    return stack


def _make_qpdk_sim(component, stack, tmp_path):
    """Create, configure, and mesh a QPDK-style DrivenSim."""
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim-qpdk"))
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.add_cpw_port("o1", layer="superconductor", s_width=10, gap_width=6, length=5.0)
    sim.add_cpw_port("o2", layer="superconductor", s_width=10, gap_width=6, length=5.0)
    sim.set_driven(fmin=5e9, fmax=10e9, num_points=20)
    sim.mesh(preset="coarse", margin=0)
    return sim


@pytest.fixture(scope="module")
def qpdk_sim(tmp_path_factory):
    """Mesh once with QPDK-style zero-thickness conductors."""
    tmp_path = tmp_path_factory.mktemp("qpdk")
    component = _make_qpdk_component()
    stack = _make_qpdk_stack()
    return _make_qpdk_sim(component, stack, tmp_path)


class TestQPDKMesh:
    """Test mesh generation with QPDK-style zero-thickness conductors."""

    def test_mesh_has_pec_surfaces(self, qpdk_sim):
        """Zero-thickness conductors must produce PEC surface groups."""
        groups = qpdk_sim._last_mesh_result.groups
        assert len(groups["pec_surfaces"]) > 0, (
            f"No PEC surfaces found. Got: {groups['pec_surfaces']}"
        )

    def test_mesh_has_volumes(self, qpdk_sim):
        """Sapphire and vacuum volumes must be present."""
        groups = qpdk_sim._last_mesh_result.groups
        vol_names = list(groups["volumes"].keys())
        assert len(vol_names) >= 2, f"Expected >= 2 volumes, got: {vol_names}"

    def test_mesh_has_port_surfaces(self, qpdk_sim):
        """CPW ports must produce port surface groups."""
        groups = qpdk_sim._last_mesh_result.groups
        assert "P1" in groups["port_surfaces"], "Port P1 not found"
        assert "P2" in groups["port_surfaces"], "Port P2 not found"

    def test_no_conductor_volume_surfaces(self, qpdk_sim):
        """Zero-thickness conductors must NOT produce volumetric surfaces."""
        groups = qpdk_sim._last_mesh_result.groups
        assert len(groups["conductor_surfaces"]) == 0, (
            f"Unexpected volumetric conductor surfaces: "
            f"{list(groups['conductor_surfaces'].keys())}"
        )

    def test_config_json_has_pec_not_conductivity(self, qpdk_sim):
        """Config must have PEC boundary (not Conductivity) for superconductors."""
        qpdk_sim.write_config()
        config_path = Path(qpdk_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        boundaries = config["Boundaries"]
        assert "PEC" in boundaries, "Missing PEC boundary for superconductor"
        assert "LumpedPort" in boundaries, "Missing LumpedPort"
        assert len(boundaries["LumpedPort"]) == 2, "Expected 2 lumped ports"

    def test_validate_mesh_passes(self, qpdk_sim):
        """validate_mesh() must pass for a valid QPDK setup."""
        result = qpdk_sim.validate_mesh()
        assert result.valid, f"Validation failed:\n{result}"


# ---------------------------------------------------------------------------
# PEC block support
# ---------------------------------------------------------------------------

PEC_LAYER = (65000, 0)


def _make_pec_component():
    """Create a component with a PEC polygon on layer (65000, 0)."""
    gf.gpdk.PDK.activate()

    @gf.cell
    def pec_block_cpw(
        length: float = 300,
        s_width: float = 20,
        g_width: float = 40,
        gap_width: float = 15,
        pec_width: float = 100,
        pec_height: float = 5,
        layer=(99, 0),
    ) -> gf.Component:
        c = gf.Component()
        # GSG electrode (same as above)
        r1 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        r1.move((0, (g_width + s_width) / 2 + gap_width))
        c << gf.c.rectangle((length, s_width), centered=True, layer=layer)
        r3 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        r3.move((0, -(g_width + s_width) / 2 - gap_width))

        # PEC block polygons at each port boundary
        pec_left = c << gf.c.rectangle(
            (pec_height, pec_width), centered=True, layer=PEC_LAYER
        )
        pec_left.move((-length / 2, 0))
        pec_right = c << gf.c.rectangle(
            (pec_height, pec_width), centered=True, layer=PEC_LAYER
        )
        pec_right.move((length / 2, 0))

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

    return pec_block_cpw()


@pytest.fixture(scope="module")
def pec_block_sim(tmp_path_factory):
    """Mesh with PEC blocks at port boundaries."""
    tmp_path = tmp_path_factory.mktemp("pec_block")
    component = _make_pec_component()
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(component)
    sim.set_stack(substrate_thickness=2.0, air_above=300.0)
    sim.add_cpw_port("o1", layer="topmetal2", s_width=20, gap_width=15, length=5.0)
    sim.add_cpw_port("o2", layer="topmetal2", s_width=20, gap_width=15, length=5.0)
    sim.add_pec(gds_layer=PEC_LAYER, from_layer="metal1", to_layer="topmetal2")
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=40)
    sim.mesh(preset="coarse")
    return sim


class TestPECBlockMesh:
    """Test mesh generation with PEC blocks."""

    def test_mesh_has_pec_surfaces(self, pec_block_sim):
        """PEC blocks must produce PEC surface groups."""
        groups = pec_block_sim._last_mesh_result.groups
        pec_names = list(groups["pec_surfaces"].keys())
        assert len(pec_names) > 0, f"No PEC surfaces found. Got: {pec_names}"
        # Should contain pec_block_0 entries
        assert any("pec_block" in name for name in pec_names), (
            f"No pec_block entries in PEC surfaces. Got: {pec_names}"
        )

    def test_config_json_has_pec(self, pec_block_sim):
        """Config must include PEC boundary when PEC blocks are present."""
        pec_block_sim.write_config()
        config_path = Path(pec_block_sim._output_dir) / "config.json"
        config = json.loads(config_path.read_text())
        assert "PEC" in config["Boundaries"], "Missing PEC boundary"

    def test_mesh_has_conductor_surfaces(self, pec_block_sim):
        """Volumetric conductors should still be present alongside PEC blocks."""
        groups = pec_block_sim._last_mesh_result.groups
        assert len(groups["conductor_surfaces"]) > 0, (
            "No conductor surfaces found — PEC blocks should not replace them"
        )

    def test_validate_mesh_passes(self, pec_block_sim):
        """validate_mesh() must pass with PEC blocks."""
        result = pec_block_sim.validate_mesh()
        assert result.valid, f"Validation failed:\n{result}"
