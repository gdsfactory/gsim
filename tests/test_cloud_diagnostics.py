"""Check cloud failure artifacts and terminal detection at SDK boundaries."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gsim import gcloud
from tests.test_gcloud_start import FakeJob


@pytest.mark.parametrize("artifacts", ["directory", "no-log", "download-error"])
def test_failed_job_retains_download_diagnostics(tmp_path, monkeypatch, artifacts):
    directory = tmp_path / "artifacts"
    directory.mkdir()
    (directory / "solver.log").write_text("solver failed")
    (directory / ".hidden").write_text("hidden")
    (directory / "nested").mkdir()
    (directory / "nested" / "config.json").write_text("{}")
    downloader = (
        Mock(side_effect=OSError("network offline"))
        if artifacts == "download-error"
        else Mock(
            return_value={"output": directory}
            if artifacts == "directory"
            else {"output": directory / "nested" / "config.json"}
        )
    )
    monkeypatch.setattr(gcloud.sim, "download_results", downloader)
    job = FakeJob(status="failed", exit_code=1, download_urls={"output": "unused"})
    with pytest.raises(RuntimeError) as error:
        gcloud._handle_failed_job(job, tmp_path, verbose=True)
    assert {
        "directory": "solver failed",
        "no-log": "No log file",
        "download-error": "network offline",
    }[artifacts] in str(error.value)


@pytest.mark.parametrize("environment", ["jupyter", "tty", "pipe", "no-ipython"])
def test_output_detection(monkeypatch, environment):
    if environment == "no-ipython":
        monkeypatch.setitem(sys.modules, "IPython", None)
    else:
        monkeypatch.setattr(
            "IPython.get_ipython",
            lambda: SimpleNamespace(config={"IPKernelApp": {}})
            if environment == "jupyter"
            else None,
        )
    monkeypatch.setattr(sys.stdout, "isatty", lambda: environment == "tty")
    assert gcloud._output_mode() == (
        "pipe" if environment == "no-ipython" else environment
    )


def test_unsupported_log_streaming_falls_back_to_status(tmp_path, monkeypatch):
    job = FakeJob(job_def_name="custom", status="completed")
    monkeypatch.setattr(gcloud.sim, "get_job", Mock(return_value=job))
    monkeypatch.setattr(gcloud.sim, "download_results", Mock(return_value={}))
    monkeypatch.setattr(gcloud, "_get_job_logs_callable", lambda: None)
    with pytest.warns(UserWarning, match="Log streaming is not supported"):
        result = gcloud.wait_for_results("job-1", verbose="full", parent_dir=tmp_path)
    assert result.files == {}
