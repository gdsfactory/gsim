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
@pytest.mark.parametrize("threads", [1, 2])
def test_component_sheet_mesh_config(
    tmp_path: Path, depth: float | None, threads: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = []
    generate = gmsh.model.mesh.generate

    def record_threads(dim: int) -> None:
        options.append(gmsh.option.getNumber("General.NumThreads"))
        generate(dim)

    monkeypatch.setattr(gmsh.model.mesh, "generate", record_threads)
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
        num_threads=threads,
    )
    sim.solver.linear.initial_guess = False
    config = json.loads(sim.write_config().read_text())
    assert config["Solver"]["Linear"]["InitialGuess"] is False
    assert config["Model"].get("Lc") == depth
    assert len(config["Boundaries"]["Terminal"]) == 2
    assert mesh.metadata["terminal_names"] == ("T1", "T2")
    assert mesh.metadata["mesh_threads"] == threads
    assert options == [threads]
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


def test_meshing_progress_preserves_mesh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = []
    generate = gmsh.model.mesh.generate

    def record_terminal(dim: int) -> None:
        # Native DLL output on Windows bypasses pytest's descriptor capture.
        options.append(gmsh.option.getNumber("General.Terminal"))
        generate(dim)

    monkeypatch.setattr(gmsh.model.mesh, "generate", record_terminal)
    meshes = []
    for verbose in (False, True):
        sim = ElectrostaticSim()
        sim.set_geometry(_component())
        sim.set_output_dir(tmp_path / str(verbose))
        mesh = sim.mesh_sheets(
            conductor_layer=(1, 0),
            terminal_ports={"T1": "lower", "T2": "upper"},
            domain_bounds=(0, -25, 4, 25),
            height=25,
            near_mesh=0.4,
            far_mesh=5,
            normalization_depth_um=4,
            verbose=verbose,
        )
        assert options[-1] == int(verbose)
        meshes.append(mesh.mesh_path.read_bytes())
    assert meshes[0] == meshes[1]


def test_transverse_mesh_refines_edges_of_long_conductors(tmp_path: Path) -> None:
    gf.gpdk.PDK.activate()
    component = gf.Component()
    for bottom, top in [(-150, -120), (-119, -40), (-20, 20), (40, 119), (120, 150)]:
        component.add_polygon(
            [(0, bottom), (4, bottom), (4, top), (0, top)], layer=(1, 0)
        )
    for name, y in [("lower", -80), ("upper", 80)]:
        component.add_port(
            name=name,
            center=(0, y),
            width=2,
            orientation=180,
            layer=(1, 0),
            port_type="electrical",
        )
    sim = ElectrostaticSim()
    sim.set_geometry(component)
    sim.set_output_dir(tmp_path)
    mesh = sim.mesh_sheets(
        conductor_layer=(1, 0),
        terminal_ports={"T1": "lower", "T2": "upper"},
        domain_bounds=(0, -150, 4, 150),
        height=150,
        near_mesh=0.02,
        far_mesh=8,
        normalization_depth_um=4,
    )
    assert mesh.mesh_stats["triangles"] < 20_000
    assert sim.validate_mesh().valid


@pytest.mark.parametrize("threads", [0, -1])
def test_invalid_mesh_thread_count(tmp_path: Path, threads: int) -> None:
    sim = ElectrostaticSim()
    sim.set_geometry(_component())
    sim.set_output_dir(tmp_path)
    with pytest.raises(ValueError, match="thread count must be positive"):
        sim.mesh_sheets(
            conductor_layer=(1, 0),
            terminal_ports={"T1": "lower", "T2": "upper"},
            domain_bounds=(0, -25, 4, 25),
            height=25,
            near_mesh=0.4,
            far_mesh=5,
            num_threads=threads,
        )
