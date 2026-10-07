"""Legacy and grouped settings must preserve complete Palace cache inputs."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import gsim.palace as pa
from gsim.common.stack.extractor import LayerStack
from gsim.palace.mesh.generator import MeshResult

COMMON = {
    "order": 1,
    "tolerance": 1e-8,
    "max_iterations": 123,
    "solver_type": "Default",
    "preconditioner": "AMS",
    "device": "CPU",
}
CASES = [
    (pa.DrivenSim, "driven", {"fmin": 2e9, "fmax": 8e9, "num_points": 7}),
    (pa.EigenmodeSim, "eigenmode", {"target": 4e9, "tolerance": 1e-8}),
    (pa.ElectrostaticSim, "electrostatic", {"save_fields": 2}),
]


@pytest.mark.parametrize(("sim_class", "group", "problem"), CASES)
@pytest.mark.parametrize(
    "style",
    [
        "legacy_constructor",
        "legacy_setter",
        "legacy_assignment",
        "grouped_constructor",
        "grouped_setter",
    ],
)
def test_complete_config_matches_main_baseline(
    tmp_path, monkeypatch, sim_class, group, problem, style
):
    monkeypatch.setattr(sim_class, "_resolve_stack", lambda self: LayerStack())
    if style == "legacy_constructor":
        sim = sim_class(numerical=pa.NumericalConfig(**COMMON), **{group: problem})
    elif style == "grouped_constructor":
        sim = sim_class(
            solver={
                "order": COMMON["order"],
                "device": COMMON["device"],
                "linear": {
                    key: value
                    for key, value in COMMON.items()
                    if key not in ("order", "device")
                },
                group: problem,
            }
        )
    else:
        sim = sim_class(solver={group: problem})
        if style == "legacy_assignment":
            sim.numerical = pa.NumericalConfig(**COMMON)
        else:
            setter = sim.set_numerical if style == "legacy_setter" else sim.set_solver
            setter(**COMMON)
    sim.stack = LayerStack()
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
    output = sim.write_config(validate_mesh=False)
    # Recorded through the old API on main at 0950c7f. Byte comparison also
    # protects key ordering, which participates in the cloud input hash.
    reference = Path(__file__).with_suffix("") / f"{group}.txt"
    assert output.read_bytes() == reference.read_bytes().removesuffix(b"\n")


@pytest.mark.parametrize(("sim_class", "group", "problem"), CASES)
def test_numerical_copy_retains_all_common_settings(sim_class, group, problem):
    sim = sim_class(solver={group: problem})
    sim.set_solver(**COMMON)
    copied = pa.NumericalConfig(**sim.numerical.model_dump())
    assert copied.model_dump() == COMMON
    assert "linear" not in copied.model_dump()


def test_numerical_copy_accepts_linear_model():
    copied = pa.NumericalConfig(
        linear=pa.LinearSolverConfig(tolerance=1e-8, max_iterations=123)
    )
    assert copied.tolerance == 1e-8
    assert copied.max_iterations == 123


def test_numerical_copy_rejects_ambiguous_settings():
    with pytest.raises(ValidationError, match="supplied twice"):
        pa.NumericalConfig(tolerance=1e-8, linear={"tolerance": 1e-6})


@pytest.mark.parametrize(("sim_class", "group", "problem"), CASES[1:])
def test_old_serialized_inputs_allow_null_driven_group(sim_class, group, problem):
    sim = sim_class.model_validate(
        {"numerical": COMMON, "driven": None, group: problem, "geometry": None}
    )
    assert sim.solver.order == 1
    assert sim.solver.linear.tolerance == 1e-8
    assert getattr(sim.solver, group).model_dump().items() >= problem.items()
