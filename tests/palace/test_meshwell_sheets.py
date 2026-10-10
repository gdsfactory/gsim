"""Component sheet meshing and port-selected electrostatic terminals."""

from __future__ import annotations

import json
from pathlib import Path

import gdsfactory as gf
import gmsh
import pytest

from gsim.palace import ElectrostaticSim


def _component() -> gf.Component:
    """Two electrodes with disconnected outer and inner ground conductors."""
    gf.gpdk.PDK.activate()
    component = gf.Component()
    for bottom, top in [(-25, -8), (-7, -2), (-0.006, 0.006), (2, 7), (8, 25)]:
        component.add_polygon(
            [(0, bottom), (4, bottom), (4, top), (0, top)], layer=(1, 0)
        )
    for name, y in [("lower", -4), ("upper", 4)]:
        component.add_port(
            name=name,
            center=(0, y),
            width=2,
            orientation=180,
            layer=(1, 0),
            port_type="electrical",
        )
    return component


@pytest.mark.parametrize("depth", [None, 4.0])
def test_component_sheet_mesh_config(tmp_path: Path, depth: float | None) -> None:
    sim = ElectrostaticSim()
    sim.set_geometry(_component())
    sim.set_output_dir(tmp_path)
    mesh = sim.mesh_sheets(
        conductor_layer=(1, 0),
        terminal_ports={"T1": "lower", "T2": "upper"},
        domain_bounds=(0, -25, 4, 25),
        height=25,
        near_mesh=0.4,
        far_mesh=5,
        normalization_depth_um=depth,
    )
    sim.solver.linear.initial_guess = False
    config = json.loads(sim.write_config().read_text())
    assert config["Solver"]["Linear"]["InitialGuess"] is False
    assert config["Model"].get("Lc") == depth
    assert len(config["Boundaries"]["Terminal"]) == 2
    assert mesh.metadata["terminal_names"] == ("T1", "T2")
    gmsh.initialize()
    try:
        gmsh.open(str(mesh.mesh_path))
        dim = 2 if depth else 3
        assert {
            gmsh.model.getPhysicalName(d, tag)
            for d, tag in gmsh.model.getPhysicalGroups(dim)
        } == {"air", "silicon"}
        assert {
            gmsh.model.getPhysicalName(d, tag)
            for d, tag in gmsh.model.getPhysicalGroups(dim - 1)
        } == {"T1", "T2", "ground"}
    finally:
        gmsh.finalize()


def test_shorted_sheet_terminals_rejected(tmp_path: Path) -> None:
    sim = ElectrostaticSim()
    component = _component()
    component.add_port(
        name="also_lower",
        center=(4, -4),
        width=2,
        orientation=0,
        layer=(1, 0),
        port_type="electrical",
    )
    sim.set_geometry(component)
    sim.set_output_dir(tmp_path)
    with pytest.raises(ValueError, match="unselected conductor"):
        sim.mesh_sheets(
            conductor_layer=(1, 0),
            terminal_ports={"T1": "lower", "T2": "also_lower"},
            domain_bounds=(0, -25, 4, 25),
            height=25,
            near_mesh=0.4,
            far_mesh=5,
        )


def test_ground_frame_hole_keeps_electrodes_separate(tmp_path: Path) -> None:
    gf.gpdk.PDK.activate()
    component = gf.Component()
    for name, y in [("lower", -4), ("upper", 4)]:
        component.add_polygon(
            [(0, y - 1), (4, y - 1), (4, y + 1), (0, y + 1)], layer=(1, 0)
        )
        component.add_port(
            name=name,
            center=(0, y),
            width=2,
            orientation=180,
            layer=(1, 0),
            port_type="electrical",
        )
    outer = gf.c.rectangle(size=(14, 50), layer=(1, 0))
    inner = gf.c.rectangle(size=(10, 16), layer=(1, 0))
    inner_component = gf.Component()
    ref = inner_component.add_ref(inner)
    ref.dmove((2, 17))
    frame = component.add_ref(
        gf.boolean(outer, inner_component, operation="not", layer=(1, 0))
    )
    frame.dmove((-5, -25))
    sim = ElectrostaticSim()
    sim.set_geometry(component)
    sim.set_output_dir(tmp_path)
    mesh = sim.mesh_sheets(
        conductor_layer=(1, 0),
        terminal_ports={"T1": "lower", "T2": "upper"},
        domain_bounds=(-5, -25, 9, 25),
        height=25,
        near_mesh=0.5,
        far_mesh=5,
    )
    assert set(mesh.groups["pec_surfaces"]) == {"T1", "T2", "ground"}
    assert sim.validate_mesh().valid
