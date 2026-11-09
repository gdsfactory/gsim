"""Check the shared cloud-job lifecycle once, against a fake gcloud."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from gsim import fdtd, gcloud
from gsim.common.cloud import CloudJobMixin
from gsim.hashing import compute_input_hash
from gsim.meep import Simulation as MeepSimulation
from gsim.palace import BoundaryModeSim, DrivenSim, EigenmodeSim, ElectrostaticSim


class _Sim(CloudJobMixin):
    _cloud_job_type = "toy"

    def __init__(self, staging: Path) -> None:
        self.staging = staging

    def _prepare_upload_dir(self) -> Path:
        self.staging.mkdir()
        (self.staging / "config.json").write_text("{}")
        return self.staging


@pytest.fixture
def cloud(monkeypatch):
    """Replace every gcloud call the lifecycle makes."""
    calls = Mock()
    calls.upload.return_value = "new-job"
    calls.wait_for_results.return_value = "results"
    calls.get_status.return_value = "running"
    calls.check_cache_for_dir.side_effect = lambda path, job_type: (
        compute_input_hash(path, job_type),
        calls.cached_job,
    )
    calls.cached_job = None
    for name in ("upload", "start", "wait_for_results", "get_status"):
        monkeypatch.setattr(gcloud, name, getattr(calls, name))
    monkeypatch.setattr(gcloud, "check_cache_for_dir", calls.check_cache_for_dir)
    return calls


@pytest.mark.parametrize("cache", ["off", "hit", "miss"])
@pytest.mark.parametrize("wait", [False, True])
def test_run_reuses_cached_jobs_and_records_the_job(
    cloud, tmp_path, capsys, cache, wait
):
    cloud.cached_job = "old-job" if cache == "hit" else None
    simulation = _Sim(tmp_path / "staging")

    result = simulation.run(
        tmp_path, check_cache=cache != "off", wait=wait, poll_interval=0.5
    )

    job_id = "old-job" if cache == "hit" else "new-job"
    assert result == ("results" if wait else job_id)
    assert simulation.get_status() == "running"
    assert simulation._config_dir == tmp_path / "staging"
    assert simulation._input_hash == compute_input_hash(tmp_path / "staging", "toy")
    assert simulation._estimated_runtime_seconds == gcloud.estimate_runtime_seconds(
        tmp_path / "staging"
    )
    assert cloud.upload.call_count == cloud.start.call_count == (cache != "hit")
    if cache != "hit":
        assert cloud.upload.call_args.kwargs == {
            "verbose": False,
            "input_hash": simulation._input_hash,
        }
    assert ("Cache hit: reusing job old-job" in capsys.readouterr().out) == (
        cache == "hit"
    )
    if wait:
        cloud.wait_for_results.assert_called_once_with(
            job_id,
            verbose="status",
            parent_dir=tmp_path,
            poll_interval=0.5,
            estimated_runtime_seconds=simulation._estimated_runtime_seconds,
        )
    else:
        cloud.wait_for_results.assert_not_called()


def test_upload_does_not_start(cloud, tmp_path):
    simulation = _Sim(tmp_path / "staging")
    assert simulation.upload(verbose=False) == "new-job"
    cloud.start.assert_not_called()
    simulation.start(verbose=False)
    cloud.start.assert_called_once_with("new-job", verbose=False)


def test_a_failed_start_keeps_the_uploaded_job_for_a_retry(cloud, tmp_path):
    cloud.start.side_effect = [RuntimeError("start refused"), None]
    simulation = _Sim(tmp_path / "staging")
    with pytest.raises(RuntimeError, match="start refused"):
        simulation.run(wait=False)
    assert simulation._job_id == "new-job"
    assert simulation._config_dir == tmp_path / "staging"
    assert (tmp_path / "staging" / "config.json").is_file()
    assert simulation.get_status() == "running"
    simulation.start(verbose=False)
    assert cloud.upload.call_count == 1
    assert [call.args for call in cloud.start.call_args_list] == [
        ("new-job",),
        ("new-job",),
    ]


@pytest.mark.parametrize("check_cache", [False, True])
@pytest.mark.parametrize("entry", ["upload", "run"])
def test_failed_upload_removes_the_staging_copy(cloud, tmp_path, check_cache, entry):
    cloud.upload.side_effect = RuntimeError("offline")
    simulation = _Sim(tmp_path / "staging")
    submit = (
        simulation.upload
        if entry == "upload"
        else lambda: simulation.run(check_cache=check_cache)
    )
    with pytest.raises(RuntimeError, match="offline"):
        submit()
    assert not (tmp_path / "staging").exists()
    assert simulation._job_id is None


@pytest.mark.parametrize(
    "simulation_class",
    [
        DrivenSim,
        EigenmodeSim,
        ElectrostaticSim,
        BoundaryModeSim,
        MeepSimulation,
        fdtd.Simulation,
    ],
)
@pytest.mark.parametrize("method", ["start", "get_status", "wait_for_results"])
def test_every_solver_refuses_job_calls_before_a_job_exists(simulation_class, method):
    with pytest.raises(ValueError, match=r"upload|No job submitted"):
        getattr(simulation_class(), method)()
