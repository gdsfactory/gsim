"""Dry-run requests must stay separate from full simulations and their cache."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gsim import fdtd, gcloud
from gsim.hashing import compute_input_hash


def test_dry_run_artifacts_do_not_change_full_run_settings(tmp_path, fdtd_pdk_module):
    simulation = fdtd.Simulation(
        pdk=fdtd_pdk_module, default_port="o1", mesh_size_nm=750
    )
    simulation.geometry("straight", settings={"length": 2})
    simulation.solver(max_timesteps=1234)
    dry_path = tmp_path / "dry"
    full_path = tmp_path / "full"

    simulation.write(dry_path, dry_run=True)
    simulation.write(full_path)
    dry = json.loads((dry_path / "config.json").read_text())
    full = json.loads((full_path / "config.json").read_text())

    assert dry["run"]["dry_run"] is True
    assert dry["run"]["max_timesteps"] == 1
    assert full["run"]["dry_run"] is False
    assert full["run"]["max_timesteps"] == simulation.solver.max_timesteps == 1234
    assert compute_input_hash(dry_path, "fdtd") != compute_input_hash(full_path, "fdtd")


@pytest.mark.parametrize("check_cache", [False, True])
def test_cloud_dry_run_followed_by_full_run(monkeypatch, check_cache):
    simulation = fdtd.Simulation()
    requested_modes = []
    hashes = []

    def write(directory: Path, *, dry_run=False):
        requested_modes.append(dry_run)
        (directory / "config.json").write_text(
            json.dumps({"run": {"dry_run": dry_run}})
        )

    def upload(directory, _job_type, **_kwargs):
        hashes.append(compute_input_hash(directory, "fdtd"))
        return "job-test"

    monkeypatch.setattr(simulation, "write", write)
    monkeypatch.setattr(gcloud, "upload", upload)
    monkeypatch.setattr(gcloud, "start", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        gcloud,
        "check_cache_for_dir",
        lambda directory, job_type: (compute_input_hash(directory, job_type), None),
    )
    preview = SimpleNamespace(
        dry_run=True,
        convergence={"timesteps_run": 0},
        excitation_type="eigenmode",
        source_mode=object(),
    )
    monkeypatch.setattr(gcloud, "wait_for_results", lambda *_args, **_kwargs: preview)

    assert (
        simulation.run(dry_run=True, check_cache=check_cache, verbose="quiet")
        is preview
    )
    simulation.run(wait=False, check_cache=check_cache, verbose="quiet")
    assert requested_modes == [True, False]
    assert hashes[0] != hashes[1]
    assert not simulation._dry_run


@pytest.mark.parametrize(
    "result",
    [
        SimpleNamespace(dry_run=False),
        SimpleNamespace(dry_run=True, convergence={"timesteps_run": 1}),
        SimpleNamespace(
            dry_run=True,
            convergence={"timesteps_run": 0},
            excitation_type="eigenmode",
            source_mode=None,
        ),
    ],
)
def test_rejects_cloud_results_without_a_mode_preview(monkeypatch, result):
    simulation = fdtd.Simulation()
    simulation._job_id = "old-backend"
    simulation._dry_run = True
    monkeypatch.setattr(gcloud, "wait_for_results", lambda *_args, **_kwargs: result)
    with pytest.raises(RuntimeError, match="Update the FDTD cloud image"):
        simulation.wait_for_results(verbose="quiet")


def test_dry_run_cache_hit_still_validates_backend(monkeypatch):
    simulation = fdtd.Simulation()
    simulation._job_id = "cached"
    monkeypatch.setattr(simulation, "write", lambda directory, **kwargs: None)
    monkeypatch.setattr(
        gcloud, "check_cache_for_dir", lambda *_args: ("hash", "cached")
    )
    monkeypatch.setattr(
        gcloud,
        "wait_for_results",
        lambda *_args, **_kwargs: SimpleNamespace(dry_run=False),
    )
    with pytest.raises(RuntimeError, match="zero timesteps"):
        simulation.run(dry_run=True, check_cache=True, verbose="quiet")
