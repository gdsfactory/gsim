"""BoundaryModeSim defaults to MGMaxLevels 1 (awslabs/palace#1177).

Palace's geometric multigrid gives wrong losses for 2D boundary-mode problems
with conductivity or impedance boundaries, so only BoundaryModeSim turns it off
by default. An explicit ``mg_max_levels`` must still win on every API path.
"""

from __future__ import annotations

import json

import pytest

import gsim.palace as pa
from gsim.common.stack.extractor import LayerStack
from gsim.palace.mesh.generator import MeshResult

PROBLEMS = [
    (pa.DrivenSim, "driven", {"fmin": 2e9, "fmax": 8e9}, 100),
    (pa.EigenmodeSim, "eigenmode", {"target": 4e9}, 100),
    (pa.ElectrostaticSim, "electrostatic", {}, 100),
    (pa.BoundaryModeSim, "boundary_mode", {"freq": 10e9}, 1),
]


def _written_linear(sim, monkeypatch, tmp_path) -> dict:
    """Write ``config.json`` for a meshless simulation; return ``Solver.Linear``."""
    monkeypatch.setattr(type(sim), "_resolve_stack", lambda self: LayerStack())
    sim._last_ports = []
    sim._last_mesh_result = MeshResult(
        mesh_path=tmp_path / "palace.msh",
        output_dir=tmp_path,
        groups={
            "volumes": {"airbox": {"phys_group": 1}},
            "conductor_surfaces": {},
            "pec_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {"absorbing": {"phys_group": [2]}},
        },
    )
    path = sim.write_config(validate_mesh=False)
    return json.loads(path.read_text())["Solver"]["Linear"]


def test_boundary_mode_solver_config_defaults_to_one_level():
    assert pa.BoundaryModeSolverConfig().linear.mg_max_levels == 1
    assert pa.BoundaryModeSim().solver.linear.mg_max_levels == 1
    assert pa.BoundaryModeSolverConfig().to_linear_solver_config()["MGMaxLevels"] == 1


@pytest.mark.parametrize(("sim_class", "group", "problem", "levels"), PROBLEMS)
def test_written_config_uses_the_class_default(
    tmp_path, monkeypatch, sim_class, group, problem, levels
):
    sim = sim_class(solver={group: problem})
    linear = _written_linear(sim, monkeypatch, tmp_path)
    assert linear["MGMaxLevels"] == levels


@pytest.mark.parametrize(
    "build",
    [
        lambda: pa.BoundaryModeSim(solver={"linear": {"mg_max_levels": 4}}),
        lambda: pa.BoundaryModeSim(
            solver={"linear": pa.LinearSolverConfig(mg_max_levels=4)}
        ),
        lambda: pa.BoundaryModeSim(
            solver=pa.BoundaryModeSolverConfig(
                linear=pa.LinearSolverConfig(mg_max_levels=4)
            )
        ),
        lambda: pa.BoundaryModeSim(solver={"mg_max_levels": 4}),
    ],
    ids=["dict", "linear_model", "solver_model", "legacy_flat"],
)
def test_explicit_mg_max_levels_is_kept_at_construction(tmp_path, monkeypatch, build):
    sim = build()
    assert sim.solver.linear.mg_max_levels == 4
    assert _written_linear(sim, monkeypatch, tmp_path)["MGMaxLevels"] == 4


def test_explicit_mg_max_levels_is_kept_by_assignment_and_roundtrip():
    sim = pa.BoundaryModeSim()
    sim.solver.linear.mg_max_levels = 4
    assert sim.solver.linear.mg_max_levels == 4
    restored = pa.BoundaryModeSim.model_validate(sim.model_dump())
    assert restored.solver.linear.mg_max_levels == 4
    # An explicit 100 is also an explicit choice, not the multigrid-off default.
    sim.solver.linear.mg_max_levels = 100
    restored = pa.BoundaryModeSim.model_validate(sim.model_dump())
    assert restored.solver.linear.mg_max_levels == 100


@pytest.mark.parametrize("setter", ["set_solver", "set_numerical"])
def test_setters_keep_the_default_unless_levels_are_given(setter):
    sim = pa.BoundaryModeSim()
    getattr(sim, setter)(order=1, tolerance=1e-8)
    assert sim.solver.linear.mg_max_levels == 1
    getattr(sim, setter)(order=1, mg_max_levels=4)
    assert sim.solver.linear.mg_max_levels == 4
    getattr(sim, setter)(order=1, mg_max_levels=100)
    assert sim.solver.linear.mg_max_levels == 100
    getattr(sim, setter)(order=1)
    assert sim.solver.linear.mg_max_levels == 1


@pytest.mark.parametrize("sim_class", [pa.DrivenSim, pa.EigenmodeSim])
def test_other_simulations_keep_one_hundred_levels_through_set_solver(sim_class):
    sim = sim_class()
    sim.set_solver(order=1)
    assert sim.solver.linear.mg_max_levels == 100
    sim.set_solver(order=1, mg_max_levels=1)
    assert sim.solver.linear.mg_max_levels == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_legacy_numerical_settings_keep_the_default_unless_levels_are_given():
    sim = pa.BoundaryModeSim(numerical=pa.NumericalConfig(max_iterations=17))
    assert sim.solver.linear.max_iterations == 17
    assert sim.solver.linear.mg_max_levels == 1
    sim.numerical = pa.NumericalConfig(mg_max_levels=4)
    assert sim.solver.linear.mg_max_levels == 4
    sim.numerical = pa.NumericalConfig(tolerance=1e-8)
    assert sim.solver.linear.mg_max_levels == 1


def test_set_boundary_mode_keeps_the_linear_default():
    sim = pa.BoundaryModeSim()
    sim.set_boundary_mode(freq=10e9)
    assert sim.solver.linear.mg_max_levels == 1
