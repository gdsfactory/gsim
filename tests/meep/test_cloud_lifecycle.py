"""Exercise cloud orchestration while generating real canonical MEEP inputs."""

import json
import shutil
from unittest.mock import Mock

import pytest

from gsim import gcloud
from gsim.hashing import compute_input_hash
from gsim.meep import Simulation
from tests.meep.test_xy_bounds import _bounded_simulation


@pytest.mark.parametrize("cache", ["off", "hit", "miss"])
@pytest.mark.parametrize("wait", [False, True])
@pytest.mark.parametrize("verbose", ["quiet", "full"])
def test_cloud_run_serializes_real_inputs_and_reuses_completed_jobs(
    monkeypatch, tmp_path, cache, wait, verbose
):
    simulation = _bounded_simulation()
    simulation.solver(dispersion="false")
    upload = Mock(return_value="new-job")
    start = Mock()
    result = object()
    download = Mock(return_value=result)
    monkeypatch.setattr(gcloud, "upload", upload)
    monkeypatch.setattr(gcloud, "start", start)
    monkeypatch.setattr(gcloud, "wait_for_results", download)
    monkeypatch.setattr(gcloud, "get_status", lambda _job: "running")
    monkeypatch.setattr(
        gcloud,
        "check_cache_for_dir",
        lambda path, solver: (
            compute_input_hash(path, solver),
            "cached-job" if cache == "hit" else None,
        ),
    )
    try:
        actual = simulation.run(
            check_cache=cache != "off", wait=wait, verbose=verbose, parent_dir=tmp_path
        )
        job_id = "cached-job" if cache == "hit" else "new-job"
        assert actual == (result if wait else job_id)
        assert simulation.get_status() == "running"
        assert upload.call_count == start.call_count == (cache != "hit")
        directory = simulation._config_dir
        config = json.loads((directory / "sim_config.json").read_text())
        assert config["meep_verbosity"] == (2 if verbose == "full" else 0)
        assert (directory / "layout.gds").is_file()
        assert "import meep" in (directory / "run_meep.py").read_text()
        assert simulation._input_hash == compute_input_hash(directory, "meep")
        assert download.call_count == wait
        if wait:
            assert download.call_args.args == (job_id,)
            assert download.call_args.kwargs["parent_dir"] == tmp_path
    finally:
        if simulation._config_dir is not None:
            shutil.rmtree(simulation._config_dir)


@pytest.mark.parametrize("method", ["start", "get_status", "wait_for_results"])
def test_cloud_operations_require_a_submitted_job(method):
    with pytest.raises(ValueError):
        getattr(Simulation(), method)()
