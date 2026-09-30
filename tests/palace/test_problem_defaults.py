"""Validate required problem settings before exporting Palace input."""

from __future__ import annotations

import json

import pytest

from gsim.common import LayerStack
from gsim.palace import EigenmodeSim
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import EigenmodeConfig


def test_eigenmode_export_requires_target():
    with pytest.raises(ValueError, match="target frequency"):
        EigenmodeConfig().to_palace_config()


def test_eigenmode_validation_reports_missing_target():
    sim = EigenmodeSim()
    assert any("target frequency" in error for error in sim.validate_config().errors)

    sim.set_eigenmode(target=5e9)
    assert not any(
        "target frequency" in error for error in sim.validate_config().errors
    )


@pytest.mark.parametrize("target", [0, -1, float("inf"), float("nan")])
def test_eigenmode_target_must_be_positive_and_finite(target):
    with pytest.raises(ValueError):
        EigenmodeConfig(target=target)


def test_explicit_eigenmode_target_exports_in_ghz():
    assert EigenmodeConfig(
        target=5e9, num_modes=4, tolerance=1e-8, save=2
    ).to_palace_config() == {"Target": 5.0, "N": 4, "Tol": 1e-8, "Save": 2}


@pytest.mark.parametrize("simulation_type", ["driven", "eigenmode"])
def test_generated_fallback_problem_config(tmp_path, simulation_type):
    config_path = generate_palace_config(
        groups={
            "volumes": {"airbox": {"phys_group": 1}},
            "conductor_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {},
        },
        ports=[],
        port_info=[],
        stack=LayerStack(),
        output_path=tmp_path,
        model_name="palace",
        fmax=60e9,
        simulation_type=simulation_type,
    )
    solver = json.loads(config_path.read_text())["Solver"]

    if simulation_type == "driven":
        assert solver["Driven"] == {
            "Samples": [
                {
                    "Type": "Linear",
                    "MinFreq": 1.0,
                    "MaxFreq": 60.0,
                    "FreqStep": 1.5,
                    "SaveStep": 0,
                }
            ],
            "AdaptiveTol": 0.02,
        }
    else:
        assert solver["Eigenmode"] == {"N": 10, "Tol": 1e-6, "Target": 60.0}
