"""Grouped Palace solver configuration and legacy API compatibility."""

from __future__ import annotations

import json
import warnings

import pytest
from pydantic import ValidationError

import gsim.palace as pa

SIMULATIONS = [
    (pa.DrivenSim, "driven", "num_points", 12),
    (pa.EigenmodeSim, "eigenmode", "target", 4e9),
    (pa.ElectrostaticSim, "electrostatic", "save_fields", 2),
    (pa.BoundaryModeSim, "boundary_mode", "freq", 6e9),
]


@pytest.mark.parametrize(("sim_class", "group", "field", "value"), SIMULATIONS)
def test_grouped_constructor_and_serialization(sim_class, group, field, value):
    sim = sim_class(
        solver={
            "order": 3,
            "linear": {"tolerance": 2e-7, "max_iterations": 123},
            group: {field: value},
        }
    )
    assert sim.solver.order == 3
    assert sim.solver.linear.tolerance == 2e-7
    assert getattr(getattr(sim.solver, group), field) == value
    dumped = sim.model_dump()
    assert "numerical" not in dumped
    assert group not in dumped
    assert dumped["solver"][group][field] == value
    assert sim_class.model_validate(dumped).solver == sim.solver


@pytest.mark.parametrize(("sim_class", "group", "field", "value"), SIMULATIONS)
@pytest.mark.parametrize("setter", ["set_solver", "set_numerical"])
def test_setter_defaults_match_constructor_and_preserve_problem(
    sim_class, group, field, value, setter
):
    sim = sim_class(solver={group: {field: value}})
    problem = getattr(sim.solver, group)
    original = sim.solver.model_dump()
    getattr(sim, setter)()
    assert getattr(sim.solver, group) is problem
    assert sim.solver.model_dump() == original
    assert (
        sim.solver.order == pa.SolverConfig().order == pa.NumericalConfig().order == 2
    )
    getattr(sim, setter)(order=1, tolerance=1e-8)
    assert getattr(sim.solver, group) is problem
    assert sim.solver.order == 1
    assert sim.solver.linear.tolerance == 1e-8
    assert getattr(getattr(sim.solver, group), field) == value


@pytest.mark.parametrize(("sim_class", "group", "field", "value"), SIMULATIONS)
def test_legacy_constructor_and_properties(sim_class, group, field, value):
    sim = sim_class(
        numerical=pa.NumericalConfig(order=3, tolerance=1e-8),
        **{group: {field: value}},
    )
    assert sim.numerical is sim.solver
    assert getattr(sim, group) is getattr(sim.solver, group)
    sim.numerical.max_iterations = 123
    assert sim.solver.linear.max_iterations == 123
    setattr(sim, group, {field: value})
    assert getattr(getattr(sim.solver, group), field) == value
    original = sim.numerical
    sim.numerical = pa.NumericalConfig(order=1)
    assert sim.solver.order == 1
    assert getattr(getattr(sim.solver, group), field) == value
    sim.numerical = original
    assert sim.solver.order == 3
    assert sim.solver.linear.max_iterations == 123


def test_linear_and_eigenmode_tolerances_are_independent():
    sim = pa.EigenmodeSim()
    sim.solver.eigenmode.target = 4e9
    sim.solver.eigenmode.num_modes = 2
    sim.solver.eigenmode.tolerance = 1e-8
    sim.solver.linear.tolerance = 1e-6
    config = sim.solver.to_palace_config()
    assert config["Linear"]["Tol"] == 1e-6
    assert config["Eigenmode"] == {"N": 2, "Tol": 1e-8, "Target": 4.0}


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("order", 0),
        ("device", "invalid"),
        ("linear.tolerance", 0),
        ("linear.max_iterations", 0),
        ("linear.solver_type", "invalid"),
        ("linear.preconditioner", "invalid"),
        ("eigenmode.num_modes", 0),
        ("eigenmode.target", -1),
    ],
)
def test_nested_assignment_validates(path, value):
    sim = pa.EigenmodeSim()
    owner = sim.solver
    parts = path.split(".")
    for name in parts[:-1]:
        owner = getattr(owner, name)
    with pytest.raises(ValidationError):
        setattr(owner, parts[-1], value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": {}, "numerical": {}},
        {"solver": {"eigenmode": {"target": 4e9}}, "eigenmode": {"target": 5e9}},
        {"solver": {"driven": {}}},
        {"driven": {"fmin": 2e9}},
        {"solver": {"tolerance": 1e-8, "linear": {"tolerance": 1e-6}}},
    ],
)
def test_conflicting_or_inapplicable_settings_rejected(kwargs):
    with pytest.raises(ValidationError):
        pa.EigenmodeSim(**kwargs)


def test_common_model_can_be_used_in_grouped_constructor():
    sim = pa.EigenmodeSim(solver=pa.SolverConfig(order=1))
    assert sim.solver.order == 1
    assert sim.solver.eigenmode.num_modes == 10


def test_simulation_defaults_are_independent():
    first = pa.EigenmodeSim()
    second = pa.EigenmodeSim()
    first.solver.linear.tolerance = 1e-8
    first.solver.eigenmode.target = 4e9
    assert second.solver.linear.tolerance == 1e-6
    assert second.solver.eigenmode.target is None


def test_canonical_settings_emit_no_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        sim = pa.EigenmodeSim(solver={"eigenmode": {"target": 4e9}})
        sim.solver.eigenmode.tolerance = 1e-8
        sim.set_solver(order=1)
        sim.solver.to_palace_config()
        sim.model_dump()
        sim.validate_config()


def test_legacy_settings_warn_with_replacements():
    with pytest.warns(DeprecationWarning, match="NumericalConfig.*SolverConfig"):
        numerical = pa.NumericalConfig(order=1)
    with pytest.warns(DeprecationWarning, match="numerical constructor.*solver"):
        sim = pa.EigenmodeSim.model_validate({"numerical": numerical})
    with pytest.warns(DeprecationWarning, match=r"sim.numerical.*sim.solver"):
        assert sim.numerical.order == 1
    with pytest.warns(DeprecationWarning, match=r"set_numerical\(\).*set_solver\(\)"):
        sim.set_numerical()
    assert sim.solver.order == 2
    with pytest.warns(DeprecationWarning, match=r"sim.eigenmode.*sim.solver.eigenmode"):
        sim.eigenmode = pa.EigenmodeConfig(target=4e9)
    with pytest.warns(DeprecationWarning, match=r"sim.eigenmode.*sim.solver.eigenmode"):
        eigenmode = sim.eigenmode
    assert eigenmode is not None
    assert eigenmode.target == 4e9


def test_numerical_serialization_retains_old_flat_format():
    with pytest.warns(DeprecationWarning, match="NumericalConfig.*SolverConfig"):
        numerical = pa.NumericalConfig(order=3, tolerance=1e-8)
    assert numerical.model_dump() == {
        "order": 3,
        "device": "CPU",
        "tolerance": 1e-8,
        "max_iterations": 400,
        "solver_type": "Default",
        "preconditioner": "Default",
    }
    assert numerical.model_dump(include={"order", "tolerance"}) == {
        "order": 3,
        "tolerance": 1e-8,
    }
    assert list(numerical.model_dump()) == [
        "order",
        "tolerance",
        "max_iterations",
        "solver_type",
        "preconditioner",
        "device",
    ]
    with pytest.warns(DeprecationWarning, match="numerical constructor.*solver"):
        sim = pa.EigenmodeSim.model_validate(
            {"numerical": numerical.model_dump(), "eigenmode": {"target": 4e9}}
        )
    assert sim.solver.order == 3
    assert sim.solver.linear.tolerance == 1e-8
    assert sim.solver.eigenmode.target == 4e9


def test_palace_solver_serialization_preserves_cache_inputs():
    """Equivalent settings must keep the original config bytes for cache hits."""
    expected = (
        '{"Linear": {"Type": "AMS", "KSPType": "GMRES", "Tol": 1e-08, '
        '"MaxIts": 123}, "Order": 3, "Device": "CPU"}'
    )
    solver = pa.SolverConfig(
        order=3,
        linear=pa.LinearSolverConfig(
            tolerance=1e-8, max_iterations=123, preconditioner="AMS"
        ),
    )
    with pytest.warns(DeprecationWarning, match="NumericalConfig.*SolverConfig"):
        numerical = pa.NumericalConfig(
            order=3, tolerance=1e-8, max_iterations=123, preconditioner="AMS"
        )
    assert json.dumps(solver.to_solver_config()) == expected
    assert json.dumps(numerical.to_solver_config()) == expected
