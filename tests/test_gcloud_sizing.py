"""Cloud submission transports sizing once, while cached results bypass upload."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gsim import gcloud


def _inputs(tmp_path, solver="fdtd"):
    tmp_path.joinpath("config.json").write_text("{}")
    metadata = {"schema_version": 1, "solver": solver, "grid": {"cell_count": 1}}
    tmp_path.joinpath("metadata.json").write_text(json.dumps(metadata))
    return metadata


@pytest.mark.parametrize("entry_point", ["upload", "upload_simulation_dir", "run"])
@pytest.mark.parametrize("solver", ["palace", "fdtd"])
def test_submission_forwards_metadata_and_exact_manifest(
    tmp_path, monkeypatch, entry_point, solver
):
    metadata = _inputs(tmp_path, solver)
    payloads = []

    def upload_simulation(
        path, job_definition, sizing_metadata=None, input_manifest=None
    ):
        assert path.is_dir()
        assert job_definition is not None
        payloads.append((sizing_metadata, input_manifest))
        return SimpleNamespace(job_id="submitted")

    monkeypatch.setattr(gcloud.sim, "upload_simulation", upload_simulation)
    if entry_point == "run":
        job = SimpleNamespace(
            job_id="submitted", job_name="submission", status="completed", exit_code=0
        )
        monkeypatch.setattr(gcloud.sim, "start_simulation", lambda _job: job)
        monkeypatch.setattr(gcloud.sim, "wait_for_simulation", lambda _job: job)
        monkeypatch.setattr(
            gcloud.sim, "download_results", lambda *_args, **_kwargs: {}
        )
        gcloud.run_simulation(
            tmp_path, solver, verbose=False, parent_dir=tmp_path.parent
        )
    elif entry_point == "upload":
        assert gcloud.upload(tmp_path, solver, verbose=False) == "submitted"
    else:
        assert gcloud.upload_simulation_dir(tmp_path, solver).job_id == "submitted"

    assert payloads[0][0] == metadata
    assert set(payloads[0][1]) == {"config.json", "metadata.json"}
    assert payloads[0][1]["config.json"] == (
        "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    )


@pytest.mark.parametrize("partial_support", [False, True])
def test_old_sdk_warns_and_submits_with_fixed_resources(
    tmp_path, monkeypatch, caplog, partial_support
):
    _inputs(tmp_path)

    def old_upload(path, job_definition):
        assert path.is_dir()
        assert job_definition is not None
        return SimpleNamespace(job_id="fixed")

    def partial_upload(path, job_definition, sizing_metadata=None):
        assert path.is_dir()
        assert job_definition is not None
        assert sizing_metadata is None
        return SimpleNamespace(job_id="fixed")

    monkeypatch.setattr(
        gcloud.sim,
        "upload_simulation",
        partial_upload if partial_support else old_upload,
    )
    assert gcloud.upload(tmp_path, "fdtd", verbose=False) == "fixed"
    assert "dynamic sizing unavailable" in caplog.text.lower()
    assert "fixed resources" in caplog.text.lower()


def test_missing_metadata_preserves_fixed_resource_submission(tmp_path, monkeypatch):
    tmp_path.joinpath("config.json").write_text("{}")

    def fixed_upload(path, job_definition):
        assert path.is_dir()
        assert job_definition is not None
        return SimpleNamespace(job_id="legacy")

    monkeypatch.setattr(gcloud.sim, "upload_simulation", fixed_upload)
    assert gcloud.upload(tmp_path, "fdtd", verbose=False) == "legacy"


@pytest.mark.parametrize("cached", [False, True])
def test_fdtd_cache_paths_reuse_results_or_forward_sizing(
    tmp_path, monkeypatch, cached
):
    from gsim import fdtd

    simulation = fdtd.Simulation()
    metadata = _inputs(tmp_path)
    monkeypatch.setattr(simulation, "_prepare_upload_dir", lambda: tmp_path)
    monkeypatch.setattr(
        gcloud.sim,
        "check_cache",
        lambda **_kwargs: SimpleNamespace(cached=cached, job_id="cached"),
        raising=False,
    )
    submissions = []
    starts = []

    def upload_simulation(
        path, job_definition, input_hash=None, sizing_metadata=None, input_manifest=None
    ):
        assert path.is_dir()
        assert job_definition is not None
        submissions.append((sizing_metadata, input_manifest, input_hash))
        return SimpleNamespace(job_id="fresh")

    monkeypatch.setattr(gcloud.sim, "upload_simulation", upload_simulation)
    monkeypatch.setattr(
        gcloud, "start", lambda job_id, **_kwargs: starts.append(job_id)
    )

    assert simulation.run(check_cache=True, wait=False, verbose="quiet") == (
        "cached" if cached else "fresh"
    )
    if cached:
        assert submissions == []
        assert starts == []
    else:
        assert submissions[0][0] == metadata
        assert "metadata.json" in submissions[0][1]
        assert submissions[0][2].startswith("sha256:")
        assert starts == ["fresh"]
