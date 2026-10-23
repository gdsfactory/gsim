"""The error estimator settings gsim writes depend on whether AMR is on."""

from __future__ import annotations

import json
import logging

import pytest

import gsim.palace as pa
from gsim.common.stack.extractor import LayerStack
from gsim.palace.mesh.generator import MeshResult

PROBLEMS = [
    (pa.DrivenSim, "driven", {"fmin": 2e9, "fmax": 8e9}),
    (pa.EigenmodeSim, "eigenmode", {"target": 4e9}),
    (pa.ElectrostaticSim, "electrostatic", {}),
    (pa.BoundaryModeSim, "boundary_mode", {"freq": 10e9}),
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


@pytest.mark.parametrize(("sim_class", "group", "problem"), PROBLEMS)
def test_amr_off_writes_a_loose_estimator_tolerance_last(
    tmp_path, monkeypatch, sim_class, group, problem
):
    linear = _written_linear(sim_class(solver={group: problem}), monkeypatch, tmp_path)
    assert list(linear)[-1] == "EstimatorTol"
    assert linear["EstimatorTol"] == 1e-2
    assert "EstimatorMaxIts" not in linear
    assert "EstimatorMG" not in linear


@pytest.mark.parametrize(("sim_class", "group", "problem"), PROBLEMS)
def test_amr_on_leaves_the_estimator_to_palace(
    tmp_path, monkeypatch, sim_class, group, problem
):
    sim = sim_class(solver={group: problem})
    sim.set_refinement(max_its=2)
    linear = _written_linear(sim, monkeypatch, tmp_path)
    assert not {"EstimatorTol", "EstimatorMaxIts", "EstimatorMG"} & set(linear)


@pytest.mark.parametrize(
    ("explicit", "expected"),
    [
        ({"estimator_tolerance": 1e-6}, {"EstimatorTol": 1e-6}),
        ({"estimator_mg": False}, {"EstimatorTol": 1e-2, "EstimatorMG": False}),
        (
            {"estimator_mg": True, "estimator_max_iterations": 0},
            {"EstimatorTol": 1e-2, "EstimatorMaxIts": 0, "EstimatorMG": True},
        ),
    ],
)
def test_explicit_estimator_settings_win_key_by_key(
    tmp_path, monkeypatch, explicit, expected
):
    sim = pa.DrivenSim(solver={"driven": {"fmin": 2e9, "fmax": 8e9}})
    sim.set_solver(**explicit)
    linear = _written_linear(sim, monkeypatch, tmp_path)
    assert list(linear.items())[-len(expected) :] == list(expected.items())
    assert len([key for key in linear if key.startswith("Estimator")]) == len(expected)


@pytest.mark.parametrize(("max_its", "warnings"), [(2, 1), (0, 0)])
def test_skipped_estimator_warns_only_with_amr(
    tmp_path, monkeypatch, caplog, max_its, warnings
):
    sim = pa.DrivenSim(solver={"driven": {"fmin": 2e9, "fmax": 8e9}})
    sim.set_refinement(max_its=max_its)
    sim.set_solver(estimator_max_iterations=0)
    caplog.set_level(logging.WARNING, logger="gsim.palace.mesh.config_generator")
    linear = _written_linear(sim, monkeypatch, tmp_path)
    assert linear["EstimatorMaxIts"] == 0
    records = [
        r for r in caplog.records if "estimator_max_iterations=0" in r.getMessage()
    ]
    assert len(records) == warnings
    if warnings:
        assert "max_its=2" in records[0].getMessage()
