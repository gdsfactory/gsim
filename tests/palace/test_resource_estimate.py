"""Memory and time estimate of a Palace run: formula, DOF counts, warnings, hook.

No cloud job and no Palace binary: the estimate reads a ``config.json`` and the
mesh statistics, so every case here builds those by hand or with ``write_config``.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

import gsim.palace as pa
from gsim import gcloud
from gsim.common.stack.extractor import LayerStack
from gsim.palace import estimate as estimate_module
from gsim.palace.estimate import (
    DEFAULT_MACHINE_MEMORY_GIB,
    count_excitations,
    count_sweep_points,
    estimate_from_inputs,
    load_fits,
    log_estimate,
    nedelec_dofs,
)
from gsim.palace.mesh.generator import MeshResult
from gsim.palace.mesh.metadata import update_field_dofs

LOGGER = "gsim.palace.estimate"

# One fit with round numbers, so every expected value below is worked by hand:
# memory = 0.01 * dofs, time = 0.001 * dofs * work, bands exp(2 * loo).
FITS: dict[str, Any] = {
    "fits_version": 7,
    "data_date": None,
    "default_machine": {"memory_gib": 50.0, "mpi_procs": 8},
    "groups": [
        {
            "solver": "driven_uniform",
            "linear_solver": "MUMPS",
            "runs": 20,
            "dofs_min": 100,
            "dofs_max": 10000,
            "memory": {"a": 0.01, "p": 1.0, "loo": 0.1},
            "time": {"b": 0.001, "q": 1.0, "w": 1.0, "loo": 0.2},
            "dofs_per_tet": {"2": {"median": 5.0, "spread": 0.1}},
        },
        {
            "solver": "driven_adaptive",
            "linear_solver": "MUMPS",
            "runs": 20,
            "dofs_min": 100,
            "dofs_max": 10000,
            "memory": {"a": 0.01, "p": 1.0, "loo": 0.1},
            "time": {"b": 0.001, "q": 1.0, "w": 2.0, "loo": 0.6},
            "samples_seen": {"min": 2, "median": 4, "max": 6},
        },
    ],
}

# Two tetrahedra sharing a face: nine edges, seven faces (see test_mesh_metadata).
TWO_TETS: dict[str, Any] = {
    "tetrahedra": 2,
    "topology": {"edges": 9, "triangular_faces": 7},
}


def _shipped_fits() -> dict[str, Any]:
    fits = load_fits()
    assert fits is not None
    return fits


def _reason(est) -> str:
    assert est.reason is not None
    return est.reason


def _driven_config(
    *, adaptive: bool = False, order: int = 2, linear: str = "MUMPS", extra=None
) -> dict:
    solver = {
        "Order": order,
        "Linear": {"Type": linear},
        "Driven": {
            "Samples": [{"Type": "Point", "Freq": [1.0, 2.0, 3.0, 4.0, 5.0]}],
            "AdaptiveTol": 0.02 if adaptive else 0,
        },
    }
    config = {
        "Problem": {"Type": "Driven"},
        "Solver": solver,
        "Boundaries": {
            "LumpedPort": [
                {"Index": 1, "Excitation": 1},
                {"Index": 2, "Excitation": 2},
                {"Index": 3, "Excitation": False},
            ]
        },
    }
    config.update(extra or {})
    return config


def _estimate(config=None, stats=None, **kwargs):
    return estimate_from_inputs(
        _driven_config() if config is None else config,
        {} if stats is None else stats,
        fits=FITS,
        **kwargs,
    )


# --- the formula --------------------------------------------------------------


def test_memory_and_time_follow_the_fit_for_given_dofs():
    est = _estimate(dofs=1000)

    assert est.available
    assert (est.solver, est.linear_solver) == ("driven_uniform", "MUMPS")
    assert est.dofs == 1000
    assert est.dofs_source == "given"
    assert est.memory_gib == pytest.approx(10.0)
    # 2 excited ports x 5 points = 10 units of work
    assert est.time_s == pytest.approx(10.0)
    assert est.memory_range_gib == pytest.approx(
        (10 / math.exp(0.2), 10 * math.exp(0.2))
    )
    assert est.time_range_s == pytest.approx((10 / math.exp(0.4), 10 * math.exp(0.4)))
    assert (est.memory_error, est.time_error) == (0.1, 0.2)
    assert est.fits_version == 7


def test_tets_ratio_widens_the_range_by_the_dofs_per_tet_spread():
    est = _estimate(stats={"tetrahedra": 200})

    assert est.dofs == 1000
    assert est.dofs_source == "tets_ratio"
    assert est.memory_gib == pytest.approx(10.0)
    # band: exp(2 loo) times the ratio spread factor exp(2 spread) ** exponent
    wide = math.exp(0.2) * math.exp(0.2)
    assert est.memory_range_gib == pytest.approx((10 / wide, 10 * wide))


@pytest.mark.parametrize(("order", "dofs"), [(1, 9), (2, 32), (3, 75), (4, 144)])
def test_dofs_are_counted_from_the_mesh_topology_for_every_order(order, dofs):
    est = _estimate(_driven_config(order=order), TWO_TETS)

    assert est.dofs_source == "mesh_topology"
    assert est.dofs == dofs
    assert est.memory_gib == pytest.approx(0.01 * dofs)
    # a counted number carries no per-tetrahedron spread
    assert est.memory_range_gib == pytest.approx(
        (est.memory_gib / math.exp(0.2), est.memory_gib * math.exp(0.2))
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_nedelec_count_agrees_with_the_mesh_metadata_estimate(order):
    stats = dict(TWO_TETS)
    update_field_dofs(stats, field_order=order, problem_type="driven")

    assert stats["field_dofs"]["estimated_field_dofs"] == nedelec_dofs(order, 9, 7, 2)


@pytest.mark.parametrize("order", [0, -1, True, 2.0])
def test_nedelec_count_rejects_a_bad_order(order):
    with pytest.raises(ValueError, match="positive integer"):
        nedelec_dofs(order, 9, 7, 2)


def test_a_large_gap_between_counted_and_calibrated_dofs_is_noted():
    est = _estimate(stats=TWO_TETS)  # 32 counted against 2 x 5 calibrated

    assert est.dofs == 32
    assert any("differ from the calibrated" in note for note in est.notes)


def test_adaptive_time_uses_typical_samples_and_the_cap_for_the_upper_end():
    config = _driven_config(adaptive=True)
    config["Solver"]["Driven"]["AdaptiveMaxSamples"] = 10
    est = _estimate(config, dofs=1000)

    # work = samples x excitations; time = 0.001 * 1000 * work ** 2
    typical, capped = 4 * 2, 10 * 2
    band = math.exp(2 * 0.6)
    assert est.solver == "driven_adaptive"
    assert est.time_s == pytest.approx(typical**2)
    assert est.time_range_s == pytest.approx((typical**2 / band, capped**2 * band))
    assert any("not known before the run" in note for note in est.notes)


def test_adaptive_typical_samples_never_exceed_the_cap():
    config = _driven_config(adaptive=True)
    config["Solver"]["Driven"]["AdaptiveMaxSamples"] = 3
    est = _estimate(config, dofs=1000)

    assert est.time_s == pytest.approx((3 * 2) ** 2)


def test_time_fit_that_is_loose_is_noted_not_warned():
    est = _estimate(_driven_config(adaptive=True), dofs=1000)  # time loo 0.6

    assert any("order of magnitude" in note for note in est.notes)
    assert not any("order of magnitude" in text for text in est.warnings)


def test_shipped_fits_are_complete_and_consistent():
    fits = _shipped_fits()

    assert isinstance(fits["fits_version"], int)
    assert "data_date" in fits
    assert fits["default_machine"]["memory_gib"] == DEFAULT_MACHINE_MEMORY_GIB
    keys = [(g["solver"], g["linear_solver"]) for g in fits["groups"]]
    assert len(keys) == len(set(keys))
    assert ("driven_adaptive", "MUMPS") in keys
    for group in fits["groups"]:
        assert 0 < group["dofs_min"] < group["dofs_max"]
        assert group["memory"]["a"] > 0
        assert group["time"]["b"] > 0
        assert group["memory"]["loo"] >= 0
        assert group["time"]["loo"] >= 0
    adaptive = next(g for g in fits["groups"] if g["solver"] == "driven_adaptive")
    assert adaptive["samples_seen"]["min"] <= adaptive["samples_seen"]["max"]


def test_shipped_fits_reproduce_a_documented_example():
    # Driven adaptive, MUMPS, order 2, 2 excitations, 8 samples, 3e5 tetrahedra
    # at the calibrated 6.75 DOFs per tetrahedron (so 2.025e6 DOFs).
    config = _driven_config(adaptive=True)
    config["Solver"]["Driven"]["AdaptiveMaxSamples"] = 8
    est = estimate_from_inputs(config, {"tetrahedra": 300_000})

    assert est.dofs == pytest.approx(2.025e6)
    assert est.memory_gib == pytest.approx(23.4, abs=0.05)
    assert est.memory_range_gib == pytest.approx((15.9, 34.4), abs=0.05)
    assert est.time_s == pytest.approx(212, abs=1)
    assert est.time_range_s == pytest.approx((102, 437), abs=1)


# --- the sweep and the excitations --------------------------------------------


@pytest.mark.parametrize(
    ("samples", "points"),
    [
        ([{"Type": "Point", "Freq": [5.0]}], 1),
        ([{"Type": "List", "Freq": [1.0, 2.0, 4.0]}], 3),
        ([{"Type": "Log", "NSample": 12}], 12),
        (
            [{"Type": "Linear", "MinFreq": 2.0, "MaxFreq": 8.0, "FreqStep": 1.0}],
            7,
        ),
        (
            [
                {
                    "Type": "Linear",
                    "MinFreq": 1.0,
                    "MaxFreq": 100.0,
                    "FreqStep": 99 / 39,
                },
                {"Type": "Point", "Freq": [150.0]},
            ],
            41,
        ),
        ([{"Type": "Surprise"}], None),
        ([], None),
        (None, None),
    ],
)
def test_sweep_points_are_counted_from_the_config(samples, points):
    assert count_sweep_points(samples) == points


def test_excitations_count_only_excited_ports_and_never_fewer_than_one():
    assert count_excitations(_driven_config()) == 2
    assert count_excitations({}) == 1
    assert (
        count_excitations({"Boundaries": {"LumpedPort": [{"Excitation": False}]}}) == 1
    )
    config = {
        "Boundaries": {
            "LumpedPort": [{"Excitation": 1}],
            "WavePort": [{"Excitation": 2}],
        }
    }
    assert count_excitations(config) == 2


def test_unreadable_sweep_gives_memory_without_time():
    config = _driven_config()
    config["Solver"]["Driven"]["Samples"] = [{"Type": "Surprise"}]
    est = _estimate(config, dofs=1000)

    assert est.available
    assert est.memory_gib == pytest.approx(10.0)
    assert est.time_s is None
    assert est.time_range_s is None
    assert "Time:" not in est.summary()


# --- warnings -----------------------------------------------------------------


def test_no_warning_when_memory_and_range_fit_the_machine():
    est = _estimate(dofs=1000, machine_memory_gib=50)

    assert est.warnings == ()
    assert not est.exceeds_machine


def test_warning_when_memory_exceeds_the_machine():
    est = _estimate(dofs=1000, machine_memory_gib=5)

    assert est.exceeds_machine
    assert len(est.warnings) == 1
    assert "10.0 GiB" in est.warnings[0]
    assert "above the machine memory of 5 GiB" in est.warnings[0]


def test_warning_when_only_the_upper_end_exceeds_the_machine():
    # memory 10 GiB, range 8.2 to 12.2 GiB
    est = _estimate(dofs=1000, machine_memory_gib=11)

    assert not est.exceeds_machine
    assert len(est.warnings) == 1
    assert "upper end of the memory range" in est.warnings[0]


def test_machine_defaults_to_the_fits_default_machine():
    est = _estimate(dofs=1000)

    assert est.machine_memory_gib == 50.0
    assert "largest successful run" in est.machine_assumption
    assert "not a limit of the cloud service" in est.machine_assumption


def test_machine_given_by_the_caller_is_said_so():
    est = _estimate(dofs=1000, machine_memory_gib=64)

    assert est.machine_memory_gib == 64
    assert est.machine_assumption == "set by the caller"


@pytest.mark.parametrize("machine", [0, -3])
def test_machine_memory_must_be_positive(machine):
    with pytest.raises(ValueError, match="positive"):
        _estimate(dofs=1000, machine_memory_gib=machine)


def test_dofs_beyond_the_fitted_range_are_flagged_as_extrapolation():
    inside = _estimate(dofs=15_000, machine_memory_gib=1000)
    outside = _estimate(dofs=15_001, machine_memory_gib=1000)

    assert inside.warnings == ()
    assert len(outside.warnings) == 1
    assert "extrapolation" in outside.warnings[0]


def test_dofs_below_the_fitted_range_are_noted_not_warned():
    inside = _estimate(dofs=100 / 1.5)
    below = _estimate(dofs=100 / 1.5 - 1)

    assert not any("below the smallest run" in note for note in inside.notes)
    assert [("below the smallest run" in note) for note in below.notes].count(True) == 1
    assert below.warnings == ()


@pytest.mark.parametrize(
    ("refinement", "expected"),
    [
        ({"MaxIts": 2}, "adaptive mesh refinement is on"),
        ({"UniformLevels": 1}, "uniform mesh refinement levels"),
    ],
)
def test_mesh_refinement_is_flagged_because_the_estimate_is_for_the_input_mesh(
    refinement, expected
):
    config = _driven_config(extra={"Model": {"Refinement": refinement}})
    est = _estimate(config, dofs=1000, machine_memory_gib=1000)

    assert [expected in text for text in est.warnings] == [True]


def test_log_estimate_logs_the_summary_and_each_warning(caplog):
    est = _estimate(dofs=15_001, machine_memory_gib=5)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        log_estimate(est)

    info = [r for r in caplog.records if r.levelno == logging.INFO]
    warned = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(info) == 1
    assert "Palace resource estimate (driven_uniform, MUMPS)" in info[0].getMessage()
    assert len(warned) == len(est.warnings) == 2


def test_log_estimate_keeps_an_unavailable_estimate_quiet(caplog):
    est = _estimate({"Problem": {"Type": "Electrostatic"}})
    with caplog.at_level(logging.INFO, logger=LOGGER):
        log_estimate(est)

    assert [r for r in caplog.records if r.levelno >= logging.INFO] == []


# --- no estimate, no crash ----------------------------------------------------


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        ({"Problem": {"Type": "Electrostatic"}}, "Electrostatic problem type"),
        ({"Problem": {"Type": "Magnetostatic"}}, "Magnetostatic problem type"),
        ({}, "None problem type"),
        (_driven_config(linear="STRUMPACK"), "STRUMPACK linear solver"),
        # a group that has no fit: adaptive driven with an iterative solver
        (_driven_config(adaptive=True, linear="AMS"), "no fit for driven_adaptive"),
        (_driven_config(linear="BoomerAMG"), "no fit for driven_uniform"),
    ],
)
def test_unknown_group_gives_no_estimate_and_does_not_raise(config, reason):
    est = _estimate(config, dofs=1000)

    assert not est.available
    assert reason in _reason(est)
    assert est.memory_gib is None
    assert est.time_s is None
    assert est.machine_memory_gib == 50.0
    assert "not available" in est.summary()
    assert str(est) == est.summary()


def test_boundary_mode_without_a_dof_count_gives_no_estimate():
    fits = {
        **FITS,
        "groups": [
            {
                "solver": "boundary_mode",
                "linear_solver": "MUMPS",
                "dofs_min": 10,
                "dofs_max": 100,
                "memory": {"a": 1.0, "p": 1.0, "loo": 0.1},
                "time": {"b": 1.0, "q": 1.0, "w": 1.0, "loo": 0.1},
            }
        ],
    }
    config = {
        "Problem": {"Type": "BoundaryMode"},
        "Solver": {"Linear": {"Type": "MUMPS"}},
    }

    none = estimate_from_inputs(config, {"elements": 500}, fits=fits)
    given = estimate_from_inputs(config, {"elements": 500}, fits=fits, dofs=50)

    assert not none.available
    assert "no DOF count" in _reason(none)
    assert given.available
    assert given.memory_gib == pytest.approx(50.0)


def test_mesh_without_tetrahedra_gives_no_estimate():
    est = _estimate(stats={"elements": 10})

    assert not est.available
    assert "no DOF count" in _reason(est)


def test_default_linear_solver_is_read_as_superlu_with_a_note():
    fits = {
        **FITS,
        "groups": [{**FITS["groups"][0], "linear_solver": "SuperLU"}],
    }
    est = estimate_from_inputs(
        _driven_config(linear="Default"), {}, fits=fits, dofs=1000
    )

    assert est.available
    assert est.linear_solver == "SuperLU"
    assert any("read as SuperLU" in note for note in est.notes)


def test_missing_fits_file_gives_no_estimate(monkeypatch):
    monkeypatch.setattr(estimate_module, "_packaged_fits", lambda: None)

    est = estimate_from_inputs(_driven_config(), TWO_TETS)

    assert not est.available
    assert "missing" in _reason(est)
    assert est.machine_memory_gib == DEFAULT_MACHINE_MEMORY_GIB


# --- the simulation ----------------------------------------------------------


def _sim_with_config(tmp_path, monkeypatch, **driven):
    """A meshless driven sim whose real ``write_config`` has run."""
    sim = pa.DrivenSim(solver={"driven": {"fmin": 2e9, "fmax": 8e9, **driven}})
    sim.set_output_dir(tmp_path)
    monkeypatch.setattr(type(sim), "_resolve_stack", lambda self: LayerStack())
    sim._last_ports = []
    (tmp_path / "palace.msh").write_text("mesh")
    sim._last_mesh_result = MeshResult(
        mesh_path=tmp_path / "palace.msh",
        output_dir=tmp_path,
        mesh_stats={
            "tetrahedra": 1000,
            "topology": {"edges": 1400, "triangular_faces": 2100},
        },
        groups={
            "volumes": {"airbox": {"phys_group": 1}},
            "conductor_surfaces": {},
            "pec_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {"absorbing": {"phys_group": [2]}},
        },
    )
    sim.write_config(validate_mesh=False)
    return sim


def test_estimate_reads_what_write_config_wrote(tmp_path, monkeypatch):
    sim = _sim_with_config(tmp_path, monkeypatch, num_points=7, adaptive_tol=0)
    files = {p.name: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}

    est = sim.estimate_resources()

    assert est.available
    assert (est.solver, est.linear_solver) == ("driven_uniform", "MUMPS")
    # order 2: 2 * 1400 edges + 2 * 2100 faces, nothing inside the tetrahedra
    assert est.dofs == 2 * 1400 + 2 * 2100
    assert est.dofs_source == "mesh_topology"
    group = next(
        g
        for g in _shipped_fits()["groups"]
        if (g["solver"], g["linear_solver"]) == ("driven_uniform", "MUMPS")
    )
    # no excited port: one excitation; 2..8 GHz in 1 GHz steps: 7 points
    expected = group["time"]["b"] * est.dofs ** group["time"]["q"] * 7
    assert est.time_s == pytest.approx(expected)
    assert est.memory_gib == pytest.approx(
        group["memory"]["a"] * est.dofs ** group["memory"]["p"]
    )
    # the module function is the same call, and neither writes anything
    assert pa.estimate_resources(sim) == est
    assert {p.name: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == files


def test_estimate_follows_the_written_adaptive_settings(tmp_path, monkeypatch):
    sim = _sim_with_config(
        tmp_path, monkeypatch, num_points=7, adaptive_tol=0.02, adaptive_max_samples=9
    )

    est = sim.estimate_resources()

    assert est.solver == "driven_adaptive"
    assert any("cap of 9" in note for note in est.notes)


def test_estimate_takes_the_machine_from_the_caller(tmp_path, monkeypatch):
    sim = _sim_with_config(tmp_path, monkeypatch)

    est = sim.estimate_resources(machine_memory_gib=0.001)

    assert est.machine_memory_gib == 0.001
    assert est.exceeds_machine


def test_estimate_without_mesh_or_config_says_what_is_missing(tmp_path):
    sim = pa.DrivenSim()
    assert "call mesh() first" in _reason(sim.estimate_resources())

    sim.set_output_dir(tmp_path)
    sim._last_mesh_result = MeshResult(mesh_path=tmp_path / "palace.msh")
    assert "call write_config() first" in _reason(sim.estimate_resources())


# --- the submit path ---------------------------------------------------------


@pytest.fixture
def submit_sim(tmp_path, monkeypatch):
    """A sim whose mesh is big enough to warn, with the cloud calls mocked."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "palace.msh").write_text("mesh")
    (source / "nested").mkdir()
    (source / "nested" / "material.json").write_text("{}")
    sim = pa.DrivenSim()
    sim.set_output_dir(source)
    sim._last_mesh_result = MeshResult(
        mesh_path=source / "palace.msh",
        mesh_stats={"tetrahedra": 4_000_000},
    )

    def write_config(_self):
        path = source / "config.json"
        path.write_text(json.dumps(_driven_config(adaptive=True)))
        sim._last_mesh_result.config_path = path
        return path

    monkeypatch.setattr(pa.DrivenSim, "write_config", write_config)
    (tmp_path / "upload").mkdir()
    monkeypatch.setattr(
        "gsim.palace.base.tempfile.mkdtemp", lambda **_kw: str(tmp_path / "upload")
    )
    upload = Mock(return_value="new-job")
    monkeypatch.setattr(gcloud, "upload", upload)
    monkeypatch.setattr(gcloud, "start", Mock())
    return sim, source, tmp_path / "upload", upload


def _tree(path):
    return {
        str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()
    }


def test_upload_logs_the_estimate_and_warns_above_the_machine(submit_sim, caplog):
    sim, _source, _copied, _upload = submit_sim

    with caplog.at_level(logging.INFO, logger=LOGGER):
        assert sim.upload(verbose=False) == "new-job"

    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert any(
        "Palace resource estimate (driven_adaptive, MUMPS)" in m for m in messages
    )
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.name == LOGGER and r.levelno == logging.WARNING
    ]
    assert any("above the machine memory of 194.55 GiB" in m for m in warned)


def test_run_prepares_the_upload_through_the_estimate(submit_sim, monkeypatch):
    sim, _source, _copied, _upload = submit_sim
    calls = []
    original = pa.DrivenSim.estimate_resources

    def spy(self, **kwargs):
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(pa.DrivenSim, "estimate_resources", spy)

    sim.run(wait=False, verbose="quiet")

    assert calls == [{}]


def test_the_estimate_does_not_change_what_is_uploaded(submit_sim, monkeypatch):
    sim, source, copied, upload = submit_sim

    sim.upload(verbose=False)
    with_estimate = _tree(copied)
    source_after = _tree(source)
    hash_with = upload.call_args.kwargs["input_hash"]
    for item in copied.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    monkeypatch.setattr(pa.DrivenSim, "_log_resource_estimate", lambda self: None)
    sim.upload(verbose=False)

    assert _tree(copied) == with_estimate
    assert upload.call_args.kwargs["input_hash"] == hash_with
    assert _tree(source) == source_after
    assert set(with_estimate) == {
        "palace.msh",
        "config.json",
        str(Path("nested") / "material.json"),
    }


@pytest.mark.parametrize("failure", [RuntimeError("boom"), ValueError("bad fits")])
def test_a_failing_estimate_never_stops_the_upload(submit_sim, monkeypatch, failure):
    sim, _source, copied, upload = submit_sim

    def broken(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(estimate_module, "estimate_from_inputs", broken)

    assert sim.upload(verbose=False) == "new-job"
    upload.assert_called_once()
    assert (copied / "config.json").is_file()


def test_upload_of_a_sim_without_an_estimate_is_quiet(submit_sim, caplog):
    sim, _source, _copied, _upload = submit_sim
    sim._last_mesh_result.mesh_stats = {}

    with caplog.at_level(logging.INFO, logger=LOGGER):
        sim.upload(verbose=False)

    assert [r for r in caplog.records if r.name == LOGGER] == []
