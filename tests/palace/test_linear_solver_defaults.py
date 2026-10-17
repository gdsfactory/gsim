"""Default solver policy and explicit multigrid overrides across Palace APIs."""

from __future__ import annotations

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
