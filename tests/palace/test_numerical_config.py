"""Regression coverage for Palace's numerical solver configuration contract."""

from __future__ import annotations

import pytest

from gsim.palace.models import NumericalConfig


@pytest.mark.parametrize("exporter", ["to_solver_config", "to_palace_config"])
@pytest.mark.parametrize("preconditioner", ["Default", "AMS", "BoomerAMG"])
def test_numerical_export_selects_preconditioner_type(exporter, preconditioner):
    """Both public exporters select preconditioners through Linear.Type."""
    settings = NumericalConfig(
        order=3,
        tolerance=1e-8,
        max_iterations=120,
        preconditioner=preconditioner,
        device="GPU",
    )

    assert getattr(settings, exporter)() == {
        "Order": 3,
        "Device": "GPU",
        "Linear": {
            "Type": preconditioner,
            "KSPType": "GMRES",
            "Tol": 1e-8,
            "MaxIts": 120,
        },
    }


@pytest.mark.parametrize("exporter", ["to_solver_config", "to_palace_config"])
@pytest.mark.parametrize("solver_type", ["SuperLU", "STRUMPACK", "MUMPS"])
def test_explicit_solver_takes_precedence(exporter, solver_type):
    """Choosing a direct backend preserves its settings and ignores AMS."""
    settings = NumericalConfig(solver_type=solver_type, preconditioner="AMS")
    linear = getattr(settings, exporter)()["Linear"]

    expected = {
        "Type": solver_type,
        "KSPType": "GMRES",
        "Tol": 1e-6,
        "MaxIts": 400,
    }
    if solver_type == "MUMPS":
        expected.update(
            MaxIts=1,
            MGMaxLevels=1,
            EstimatorMaxIts=0,
            EstimatorTol=1e-6,
            DivFreeTol=1e-6,
            DivFreeMaxIts=0,
            PCMatReal=False,
            ComplexCoarseSolve=True,
        )
    assert linear == expected
