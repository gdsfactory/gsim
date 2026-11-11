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


# The low-level functions take no simulation object, so they pick the defaults
# themselves from the simulation type.
def _low_level_groups() -> dict:
    return {
        "volumes": {"airbox": {"phys_group": 1}},
        "conductor_surfaces": {},
        "pec_surfaces": {},
        "port_surfaces": {},
        "boundary_surfaces": {"absorbing": {"phys_group": [2]}},
    }


def _generate_config(tmp_path, simulation_type, **options) -> dict:
    from gsim.palace.mesh.config_generator import generate_palace_config
    from gsim.palace.models import BoundaryModeConfig

    tmp_path.mkdir()
    path = generate_palace_config(
        groups=_low_level_groups(),
        ports=[],
        port_info=[],
        stack=LayerStack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=10e9,
        simulation_type=simulation_type,
        boundary_mode_config=BoundaryModeConfig(freq=10e9),
        absorbing_boundary=False,
        **options,
    )
    return json.loads(path.read_text())["Solver"]["Linear"]


@pytest.mark.parametrize(
    ("simulation_type", "levels"),
    [
        ("boundarymode", 1),
        ("driven", 100),
        ("eigenmode", 100),
        ("electrostatic", 100),
        ("magnetostatic", 100),
    ],
)
def test_config_without_numerical_config_uses_the_problem_default(
    tmp_path, simulation_type, levels
):
    linear = _generate_config(tmp_path / "out", simulation_type)
    assert linear["MGMaxLevels"] == levels


def test_write_config_without_numerical_config_turns_multigrid_off(tmp_path):
    from gsim.palace.mesh import write_config

    def run(simulation_type):
        out = tmp_path / simulation_type
        out.mkdir()
        result = MeshResult(
            mesh_path=out / "palace.msh",
            output_dir=out,
            groups=_low_level_groups(),
        )
        path = write_config(
            result,
            LayerStack(),
            [],
            simulation_type=simulation_type,
            absorbing_boundary=False,
        )
        return json.loads(path.read_text())["Solver"]["Linear"]["MGMaxLevels"]

    assert run("boundarymode") == 1
    assert run("driven") == 100


def test_generate_mesh_without_numerical_config_turns_multigrid_off(tmp_path):
    import gdsfactory as gf

    from gsim.common.stack import Layer
    from gsim.palace.mesh.generator import generate_mesh
    from gsim.palace.models import BoundaryModeConfig, CrossSectionPlaneConfig

    gf.gpdk.PDK.activate()
    component = gf.Component()
    component.add_polygon([(0, 0), (2, 0), (2, 2), (0, 2)], layer=(1, 0))
    stack = LayerStack(
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0,
                zmax=1,
                thickness=1,
                material="core",
                layer_type="dielectric",
            )
        },
        materials={"core": {"permittivity": 4}, "air": {"permittivity": 1}},
        dielectrics=[],
    )
    result = generate_mesh(
        component,
        stack,
        [],
        tmp_path,
        simulation_type="boundarymode",
        cross_section=CrossSectionPlaneConfig(axis="x", value=1),
        boundary_mode_config=BoundaryModeConfig(freq=1e9, num_modes=1),
        max_mesh_size=2,
        refined_mesh_size=1,
        margin_x=1,
        margin_y=1,
        air_margin=1,
        verbosity=0,
    )
    assert result.config_path is not None
    linear = json.loads(result.config_path.read_text())["Solver"]["Linear"]
    assert linear["MGMaxLevels"] == 1


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    ("build", "levels"),
    [
        (lambda: pa.SolverConfig(), 1),
        (lambda: pa.SolverConfig(linear=pa.LinearSolverConfig(mg_max_levels=4)), 4),
        (lambda: pa.SolverConfig(linear=pa.LinearSolverConfig(mg_max_levels=100)), 100),
        (lambda: pa.NumericalConfig(), 1),
        (lambda: pa.NumericalConfig(mg_max_levels=4), 4),
        (lambda: pa.NumericalConfig(mg_max_levels=100), 100),
    ],
    ids=[
        "solver_unset",
        "solver_4",
        "solver_explicit_100",
        "numerical_unset",
        "numerical_4",
        "numerical_explicit_100",
    ],
)
def test_supplied_config_keeps_explicit_levels_and_fills_unset_ones(
    tmp_path, build, levels
):
    linear = _generate_config(
        tmp_path / "out", "boundarymode", numerical_config=build()
    )
    assert linear["MGMaxLevels"] == levels
