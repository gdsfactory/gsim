"""Check upload copies, cache reuse and lifecycle errors without cloud jobs."""

from unittest.mock import Mock

import numpy as np
import pytest

from gsim import gcloud
from gsim.hashing import compute_input_hash
from gsim.palace import DrivenSim
from gsim.palace.results import SParams


@pytest.fixture
def prepared_sim(monkeypatch, tmp_path):
    simulation = DrivenSim()
    source = tmp_path / "source"
    source.mkdir()
    (source / "palace.msh").write_text("mesh")
    (source / "nested").mkdir()
    (source / "nested" / "material.json").write_text("{}")
    simulation.set_output_dir(source)

    def write_config(_self):
        path = source / "config.json"
        path.write_text('{"Problem": {"Type": "Driven"}}')
        return path

    monkeypatch.setattr(DrivenSim, "write_config", write_config)
    monkeypatch.setattr(
        "gsim.palace.base.tempfile.mkdtemp", lambda **_kw: str(tmp_path / "upload")
    )
    # mkdtemp normally creates its directory before returning it.
    (tmp_path / "upload").mkdir()
    return simulation, source, tmp_path / "upload"


@pytest.mark.parametrize("cache", ["off", "hit", "miss"])
@pytest.mark.parametrize("wait", [False, True])
def test_cloud_lifecycle_preserves_inputs_and_reuses_cached_jobs(
    monkeypatch, prepared_sim, cache, wait, tmp_path
):
    simulation, source, copied = prepared_sim
    upload = Mock(return_value="new-job")
    start = Mock()
    results = SParams(freq=np.array([1.0]), data={}, port_names=[])
    download = Mock(return_value=results)
    monkeypatch.setattr(gcloud, "upload", upload)
    monkeypatch.setattr(gcloud, "start", start)
    monkeypatch.setattr(gcloud, "wait_for_results", download)
    monkeypatch.setattr(gcloud, "get_status", lambda _job: "running")
    monkeypatch.setattr(
        gcloud,
        "check_cache_for_dir",
        lambda *_args: ("cached-hash", "old-job" if cache == "hit" else None),
    )
    result = simulation.run(
        check_cache=cache != "off", wait=wait, verbose="quiet", parent_dir=tmp_path
    )
    job_id = "old-job" if cache == "hit" else "new-job"
    assert result == (results if wait else job_id)
    assert simulation.get_status() == "running"
    assert upload.call_count == start.call_count == (cache != "hit")
    assert (copied / "nested" / "material.json").read_bytes() == (
        source / "nested" / "material.json"
    ).read_bytes()
    assert (copied / "config.json").read_bytes() == (
        source / "config.json"
    ).read_bytes()
    if cache == "off":
        assert upload.call_args.kwargs["input_hash"] == compute_input_hash(
            source, "palace"
        )
    if wait:
        download.assert_called_once_with(
            job_id,
            verbose="quiet",
            parent_dir=tmp_path,
            estimated_runtime_seconds=gcloud.estimate_runtime_seconds(copied),
        )
    else:
        download.assert_not_called()


def test_failed_upload_cleans_temporary_copy_and_preserves_source(
    monkeypatch, prepared_sim
):
    simulation, source, copied = prepared_sim
    monkeypatch.setattr(gcloud, "upload", Mock(side_effect=RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        simulation.upload()
    assert not copied.exists()
    assert (source / "palace.msh").read_text() == "mesh"


@pytest.mark.parametrize(
    "method", ["upload", "start", "get_status", "wait_for_results"]
)
def test_cloud_lifecycle_rejects_missing_prerequisites(method):
    with pytest.raises(ValueError):
        getattr(DrivenSim(), method)()
