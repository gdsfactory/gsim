"""Exercise cloud retries and log pagination at the SDK boundary."""

import re
from datetime import timedelta
from unittest.mock import Mock

import httpx
import pytest

from gsim import gcloud
from tests.test_gcloud_start import FakeJob


@pytest.mark.parametrize("failure", ["timeout", "connect", 503, 403, 400])
def test_polling_retries_transient_errors_and_preserves_other_failures(
    monkeypatch, tmp_path, capsys, failure
):
    request = httpx.Request("GET", "https://example.test/jobs/job-1")
    error = (
        httpx.TimeoutException("timeout", request=request)
        if failure == "timeout"
        else httpx.ConnectError("offline", request=request)
        if failure == "connect"
        else httpx.HTTPStatusError(
            "HTTP failure", request=request, response=httpx.Response(failure)
        )
    )
    get_job = Mock(
        side_effect=[
            FakeJob(status="queued"),
            error,
            FakeJob(status="running"),
            FakeJob(job_def_name="custom", status="completed"),
        ]
    )
    pages = Mock(
        side_effect=[
            {
                "items": [{"message": "[2026-10-16T00:00:00.000Z] solving"}],
                "page_info": {"next_cursor": "page-2"},
            },
            {
                "items": [{"message": "finished"}],
                "page_info": {"next_cursor": "page-3"},
            },
            {"items": [], "page_info": {}},
        ]
    )
    output = tmp_path / "solver.log"
    output.write_text("result")
    monkeypatch.setattr(gcloud.sim, "get_job", get_job)
    download = Mock(return_value={"log": output})
    monkeypatch.setattr(gcloud.sim, "download_results", download)
    monkeypatch.setattr(gcloud, "_get_job_logs_callable", lambda: pages)
    monkeypatch.setattr(gcloud.time, "sleep", lambda _seconds: None)
    if failure in {403, 400}:
        with pytest.raises(httpx.HTTPStatusError) as caught:
            gcloud.wait_for_results("job-1", verbose="full", parent_dir=tmp_path)
        assert caught.value is error
        download.assert_not_called()
        return
    result = gcloud.wait_for_results("job-1", verbose="full", parent_dir=tmp_path)
    assert result.files == {"solver.log": output}
    assert get_job.call_count == 4
    assert [call.kwargs["cursor"] for call in pages.call_args_list] == [
        None,
        "page-2",
        "page-3",
    ]
    text = capsys.readouterr().out
    assert "  solving\n  finished\n" in text
    assert "2026-10-16T" not in text


@pytest.mark.parametrize("mode", ["pipe", "tty", "jupyter"])
def test_multi_job_progress_keeps_finished_elapsed_time(monkeypatch, capsys, mode):
    monkeypatch.setattr(gcloud, "_output_mode", lambda: mode)
    monkeypatch.setattr(gcloud.time, "monotonic", lambda: 200)
    cleared = Mock()
    monkeypatch.setattr("IPython.display.clear_output", cleared)
    jobs = {"a": FakeJob(status="running"), "b": FakeJob(status="failed")}
    assert (
        gcloud._print_status_table(
            jobs,
            {"a": 0, "b": 0},
            prev_lines=3,
            end_times={"b": 65},
            final=True,
        )
        == 3
    )
    output = capsys.readouterr().out
    assert all(
        value in output
        for value in [
            "Waiting for 2 jobs",
            "Job in progress...",
            "Job failed",
            "1m 05s",
        ]
    )
    assert "%" not in output
    assert ("\033[3A" in output) == (mode == "tty")
    assert cleared.call_count == (mode == "jupyter")


@pytest.mark.parametrize("mode", ["pipe", "tty", "jupyter"])
@pytest.mark.parametrize("estimate", [None, 240.0])
def test_mixed_job_descriptions_align_elapsed_columns(
    monkeypatch, capsys, mode, estimate
):
    monkeypatch.setattr(gcloud, "_output_mode", lambda: mode)
    monkeypatch.setattr(gcloud.time, "monotonic", lambda: 120)
    monkeypatch.setattr("IPython.display.clear_output", Mock())
    jobs = {
        status: FakeJob(job_name=f"meep-{status}", status=status)
        for status in ("created", "queued", "running", "completed", "failed")
    }
    jobs["queued"].job_name = "meep-job-with-a-name-longer-than-thirty-characters"
    gcloud._print_status_table(
        jobs, dict.fromkeys(jobs, 0.0), final=True, estimated_runtime_seconds=estimate
    )

    output = re.sub(r"\033\[[0-9]*[AK]", "", capsys.readouterr().out)
    rows = [line for line in output.splitlines() if "elapsed" in line]
    assert len(rows) == len(jobs)
    assert len({row.index("elapsed") for row in rows}) == 1


@pytest.mark.parametrize("job_count", [1, 2])
@pytest.mark.parametrize("terminal", ["completed", "failed"])
def test_tty_redraw_erases_shorter_terminal_rows(
    monkeypatch, capsys, job_count, terminal
):
    monkeypatch.setattr(gcloud, "_output_mode", lambda: "tty")
    monkeypatch.setattr(gcloud.time, "monotonic", lambda: 120)
    jobs = {
        f"job-{i}": FakeJob(job_name="meep-" + "long-name-" * 5, status="running")
        for i in range(job_count)
    }
    starts = dict.fromkeys(jobs, 0.0)
    lines = gcloud._print_status_table(jobs, starts, estimated_runtime_seconds=240)
    previous = capsys.readouterr().out
    for job in jobs.values():
        job.status = terminal
    gcloud._print_status_table(jobs, starts, prev_lines=lines, final=True)
    redraw = capsys.readouterr().out

    # Every row is erased before writing, including rows longer than 80 columns.
    assert len(previous.splitlines()[-1]) > 80
    if job_count == 1:
        assert "\033[1A" not in redraw
    else:
        assert redraw.startswith(f"\033[{lines}A")
        redraw = redraw.removeprefix(f"\033[{lines}A")
    rows = redraw.split("\n")[:-1]
    assert len(rows) == lines
    assert all(row.startswith("\r\033[2K") for row in rows)
    assert redraw.count(f"Job {terminal}") == job_count
    assert "ETA" not in redraw


@pytest.mark.parametrize("finished", [False, True])
def test_job_summary_includes_resources_duration_and_artifacts(capsys, finished):
    job = FakeJob(download_urls={"solver.log": "unused"})
    if finished:
        job.started_at = job.created_at
        job.finished_at = job.created_at + timedelta(seconds=125)
        job.output_size_bytes = 2 * 1024**2
    gcloud.print_job_summary(job)
    output = capsys.readouterr().out
    for value in [
        "solver.log",
        "2.0 CPU / 4 GB",
        "2m 5s" if finished else "N/A",
        "2.00 MB" if finished else "0.0 KB",
    ]:
        assert value in output


@pytest.mark.parametrize("status", [403, 503])
def test_upload_explains_denied_access_and_preserves_server_errors(
    monkeypatch, tmp_path, status
):
    error = httpx.HTTPStatusError(
        "denied",
        request=httpx.Request("POST", "https://example.test/jobs"),
        response=httpx.Response(status),
    )
    monkeypatch.setattr(gcloud.sim, "upload_simulation", Mock(side_effect=error))
    expected = (
        gcloud.CloudSimulationNotEnabledError
        if status == 403
        else httpx.HTTPStatusError
    )
    with pytest.raises(expected) as caught:
        gcloud.upload_simulation_dir(tmp_path, "palace")
    if status == 403:
        assert "not enabled for your account" in str(caught.value)
        assert caught.value.__cause__ is error
    else:
        assert caught.value is error


@pytest.mark.parametrize("available", [False, True])
def test_log_outages_keep_the_resume_cursor(monkeypatch, available):
    provider = Mock(side_effect=RuntimeError("logs unavailable")) if available else None
    monkeypatch.setattr(gcloud, "_get_job_logs_callable", lambda: provider)
    assert gcloud._fetch_logs("job-1", "resume-here") == ([], "resume-here")
