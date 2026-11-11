"""Default solver policy and explicit multigrid overrides across Palace APIs."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

import gsim.palace as pa

SIMULATIONS = [
    (pa.DrivenSim, 1e-6),
    (pa.EigenmodeSim, 1e-9),
    (pa.ElectrostaticSim, 1e-6),
    (pa.BoundaryModeSim, 1e-6),
]


@pytest.mark.parametrize(("simulation_class", "tolerance"), SIMULATIONS)
def test_complex_mumps_multigrid_default(simulation_class, tolerance):
    simulation = simulation_class()
    expected = {
        "Type": "MUMPS",
        "KSPType": "GMRES",
        "Tol": tolerance,
        "MaxIts": 400,
        "MGMaxLevels": 100,
        "PCMatReal": False,
        "ComplexCoarseSolve": True,
    }
    assert simulation.solver.to_linear_solver_config() == expected
    simulation.solver.linear.tolerance = 1e-4
    simulation.set_solver()
    assert simulation.solver.to_linear_solver_config() == expected
    if isinstance(simulation, pa.EigenmodeSim):
        simulation.set_eigenmode(target=8e9)
        assert simulation.solver.eigenmode.tolerance == 1e-9


@pytest.mark.parametrize(("simulation_class", "tolerance"), SIMULATIONS)
@pytest.mark.parametrize("setter", ["set_solver", "set_numerical"])
def test_multigrid_overrides_survive_export_and_roundtrip(
    simulation_class, tolerance, setter
):
    simulation = simulation_class()
    assert simulation.solver.linear.tolerance == tolerance
    getattr(simulation, setter)(
        order=2,
        solver_type="MUMPS",
        tolerance=1e-8,
        max_iterations=17,
        mg_max_levels=1,
        pc_mat_real=True,
        complex_coarse_solve=False,
    )
    restored = simulation_class.model_validate(simulation.model_dump())
    assert restored.solver.to_linear_solver_config() == {
        "Type": "MUMPS",
        "KSPType": "GMRES",
        "Tol": 1e-8,
        "MaxIts": 17,
        "MGMaxLevels": 1,
        "PCMatReal": True,
        "ComplexCoarseSolve": False,
    }
    restored.solver.linear.mg_max_levels = 2
    assert restored.solver.to_linear_solver_config()["MGMaxLevels"] == 2


@pytest.mark.parametrize("linear", [{}, {"max_iterations": 17}, {"mg_max_levels": 1}])
def test_partial_eigenmode_settings_keep_tight_tolerance(linear):
    simulation = pa.EigenmodeSim(solver={"linear": linear})
    assert simulation.solver.linear.tolerance == 1e-9
    assert simulation.solver.eigenmode.tolerance == 1e-9


@pytest.mark.parametrize("tolerance", [None, 1e-6, 2e-8])
@pytest.mark.parametrize(
    "settings_style", ["dictionary", "linear_model", "solver_model", "numerical_model"]
)
@pytest.mark.parametrize(
    "entry_point", ["solver", "numerical", "assignment", "numerical_with_problem"]
)
def test_eigenmode_tolerance_defaults_across_configuration_paths(
    tolerance, settings_style, entry_point
):
    linear = {"max_iterations": 17}
    if tolerance is not None:
        linear["tolerance"] = tolerance
    if settings_style == "dictionary":
        settings = {"linear": linear}
    elif settings_style == "linear_model":
        settings = {"linear": pa.LinearSolverConfig(**linear)}
    elif settings_style == "solver_model":
        settings = pa.SolverConfig(linear=pa.LinearSolverConfig(**linear))
    else:
        settings = pa.NumericalConfig(**linear)

    if entry_point == "assignment":
        simulation = pa.EigenmodeSim(solver={"eigenmode": {"target": 4e9}})
        simulation.numerical = settings
        assert simulation.solver.eigenmode.target == 4e9
    elif entry_point == "numerical_with_problem":
        legacy_settings = {"numerical": settings, "eigenmode": {"target": 4e9}}
        simulation = pa.EigenmodeSim(**legacy_settings)
        assert simulation.solver.eigenmode.target == 4e9
    else:
        simulation = pa.EigenmodeSim(**{entry_point: settings})

    expected_tolerance = 1e-9 if tolerance is None else tolerance
    assert simulation.solver.linear.tolerance == expected_tolerance
    assert simulation.solver.linear.max_iterations == 17
    assert simulation.solver.eigenmode.tolerance == 1e-9


def test_eigenmode_solver_model_preserves_nested_assignments():
    settings = pa.SolverConfig()
    settings.linear.max_iterations = 17
    simulation = pa.EigenmodeSim(solver=settings)
    assert simulation.solver.linear.tolerance == 1e-9
    assert simulation.solver.linear.max_iterations == 17

    settings.linear.tolerance = 1e-6
    simulation = pa.EigenmodeSim(solver=settings)
    assert simulation.solver.linear.tolerance == 1e-6


def test_eigenmode_problem_assignment_keeps_the_linear_settings_model():
    simulation = pa.EigenmodeSim()
    original = simulation.solver.linear
    simulation.set_eigenmode(num_modes=2, target=4e9)
    simulation.solver.order = 1
    assert simulation.solver.linear is original
    assert simulation.solver.to_linear_solver_config()["Tol"] == 1e-9


@pytest.mark.parametrize("simulation_class", [pa.DrivenSim, pa.EigenmodeSim])
@pytest.mark.parametrize("group", ["numerical", "solver"])
def test_legacy_mumps_reload_uses_multigrid_and_saved_iteration_limit(
    simulation_class, group
):
    legacy_settings = {
        "solver_type": "MUMPS",
        "tolerance": 1e-6,
        "max_iterations": 17,
    }
    simulation = simulation_class.model_validate_json(
        json.dumps({group: legacy_settings})
    )
    expected = {
        "Type": "MUMPS",
        "KSPType": "GMRES",
        "Tol": 1e-6,
        "MaxIts": 17,
        "MGMaxLevels": 100,
        "PCMatReal": False,
        "ComplexCoarseSolve": True,
    }
    assert simulation.solver.to_linear_solver_config() == expected
    restored = simulation_class.model_validate_json(simulation.model_dump_json())
    assert restored.solver.to_linear_solver_config() == expected


def test_explicit_eigenmode_tolerances_are_independent():
    simulation = pa.EigenmodeSim(
        solver={"linear": {"tolerance": 2e-6}, "eigenmode": {"tolerance": 3e-7}}
    )
    assert simulation.solver.linear.tolerance == 2e-6
    assert simulation.solver.eigenmode.tolerance == 3e-7


def test_legacy_numerical_copy_keeps_multigrid_overrides():
    simulation = pa.DrivenSim(
        solver={"linear": {"mg_max_levels": 1, "pc_mat_real": True}}
    )
    copied = pa.NumericalConfig(**simulation.solver.model_dump())
    assert (
        copied.to_linear_solver_config() == simulation.solver.to_linear_solver_config()
    )


def test_multigrid_levels_validate_on_assignment():
    linear = pa.LinearSolverConfig()
    with pytest.raises(ValidationError):
        linear.mg_max_levels = 0


def test_estimator_settings_are_written_last_in_palace_read_order():
    default = pa.LinearSolverConfig().to_palace_config()
    linear = pa.LinearSolverConfig(estimator_mg=True, estimator_tolerance=1e-3)
    config = linear.to_palace_config()
    assert list(config) == [*default, "EstimatorTol", "EstimatorMG"]
    assert config["EstimatorTol"] == 1e-3
    assert config["EstimatorMG"] is True
    linear.estimator_max_iterations = 0
    assert list(linear.to_palace_config())[-3:] == [
        "EstimatorTol",
        "EstimatorMaxIts",
        "EstimatorMG",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [("estimator_tolerance", 0), ("estimator_max_iterations", -1)],
)
def test_estimator_settings_validate_and_return_to_automatic(field, value):
    with pytest.raises(ValidationError):
        pa.LinearSolverConfig(**{field: value})
    linear = pa.LinearSolverConfig(estimator_max_iterations=0, estimator_mg=False)
    with pytest.raises(ValidationError):
        setattr(linear, field, value)
    assert linear.to_palace_config()["EstimatorMaxIts"] == 0
    linear.estimator_max_iterations = None
    linear.estimator_mg = None
    assert "EstimatorMaxIts" not in linear.to_palace_config()
    assert "EstimatorMG" not in linear.to_palace_config()


MAGNETOSTATIC_LINEAR = {
    "Type": "AMS",
    "KSPType": "CG",
    "Tol": 1e-8,
    "MaxIts": 400,
    "MGMaxLevels": 100,
    "PCMatReal": False,
    "ComplexCoarseSolve": True,
}


def test_magnetostatic_defaults_to_ams_conjugate_gradient():
    solver = pa.MagnetostaticSolverConfig()
    assert solver.linear.solver_type == "Default"
    assert solver.linear.preconditioner == "AMS"
    assert solver.to_linear_solver_config() == MAGNETOSTATIC_LINEAR
    assert solver.to_palace_config()["Magnetostatic"] == {"Save": 0}


@pytest.mark.parametrize("setter", ["set_solver", "set_numerical"])
def test_setters_keep_each_class_linear_defaults(setter):
    simulation = pa.MagnetostaticSim()
    simulation.solver.linear.preconditioner = "Default"
    getattr(simulation, setter)(order=1)
    assert simulation.solver.order == 1
    assert simulation.solver.to_linear_solver_config() == MAGNETOSTATIC_LINEAR
    getattr(simulation, setter)(order=1, tolerance=1e-6)
    assert simulation.solver.to_linear_solver_config()["Tol"] == 1e-6
    assert simulation.solver.to_linear_solver_config()["Type"] == "AMS"


@pytest.mark.parametrize(
    "settings",
    [
        {"linear": {"max_iterations": 50}},
        {"linear": pa.LinearSolverConfig(max_iterations=50)},
        pa.SolverConfig(linear=pa.LinearSolverConfig(max_iterations=50)),
    ],
)
def test_partial_magnetostatic_settings_keep_ams_conjugate_gradient(settings):
    solver = pa.MagnetostaticSolverConfig.model_validate(settings)
    assert solver.to_linear_solver_config() == {
        **MAGNETOSTATIC_LINEAR,
        "MaxIts": 50,
    }


def test_explicit_magnetostatic_linear_settings_are_kept():
    solver = pa.MagnetostaticSolverConfig(
        linear={"solver_type": "MUMPS", "tolerance": 1e-6}
    )
    assert solver.linear.solver_type == "MUMPS"
    assert solver.linear.preconditioner == "AMS"
    assert solver.linear.tolerance == 1e-6
