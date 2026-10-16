"""Tests for the non-blocking start + polling API in gsim.gcloud."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from gdsfactoryplus.sim import PreJob, SimStatus

# ---------------------------------------------------------------------------
# Lightweight SDK fake — uses real SimStatus so comparisons work
# ---------------------------------------------------------------------------


@dataclass
class FakeJob:
    """Minimal stand-in for the SDK Job object used in tests."""

    id: str = "job-abc123"
    job_name: str = "palace-abc123"
    job_def_name: str = "prod-palace-simulation"
    status: str | SimStatus = SimStatus.COMPLETED
    exit_code: int | None = 0
    download_urls: dict | None = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status_reason: str | None = None
    detail_reason: str | None = None
    output_size_bytes: int = 0
    requested_cpu: float = 2.0
    requested_memory_mb: int = 4096


# ---------------------------------------------------------------------------
# _extract_solver_from_job
# ---------------------------------------------------------------------------


class TestExtractSolverFromJob:
    """Tests for _extract_solver_from_job helper."""

    def test_palace_prod(self):
        """Extracts 'palace' from prod job definition name."""
        from gsim.gcloud import _extract_solver_from_job

        job = MagicMock(job_def_name="prod-palace-simulation")
        assert _extract_solver_from_job(job) == "palace"

    def test_meep_prod(self):
        """Extracts 'meep' from prod job definition name."""
        from gsim.gcloud import _extract_solver_from_job

        job = MagicMock(job_def_name="prod-meep-simulation")
        assert _extract_solver_from_job(job) == "meep"

    def test_femwell(self):
        """Extracts 'femwell' from dev job definition name."""
        from gsim.gcloud import _extract_solver_from_job

        job = MagicMock(job_def_name="dev-femwell-simulation")
        assert _extract_solver_from_job(job) == "femwell"

    def test_fdtd(self):
        """Extracts 'fdtd' from dev job definition name."""
        from gsim.gcloud import _extract_solver_from_job

        job = MagicMock(job_def_name="dev-fdtd-simulation")
        assert _extract_solver_from_job(job) == "fdtd"

    def test_plain_name(self):
        """Extracts solver from plain name without prefix."""
        from gsim.gcloud import _extract_solver_from_job

        job = MagicMock(job_def_name="meep")
        assert _extract_solver_from_job(job) == "meep"

    def test_unknown(self):
        """Returns None for unrecognized solver names."""
        from gsim.gcloud import _extract_solver_from_job

        job = MagicMock(job_def_name="unknown-solver")
        assert _extract_solver_from_job(job) is None

    def test_empty(self):
        """Returns None for empty job_def_name."""
        from gsim.gcloud import _extract_solver_from_job

        job = MagicMock(job_def_name="")
        assert _extract_solver_from_job(job) is None

    def test_none_attr(self):
        """Returns None when job_def_name is None."""
        from gsim.gcloud import _extract_solver_from_job

        job = MagicMock(job_def_name=None)
        assert _extract_solver_from_job(job) is None


# ---------------------------------------------------------------------------
# Job definition compatibility
# ---------------------------------------------------------------------------


class TestJobDefinitionCompatibility:
    """Tests for solver names introduced after older SDK releases."""

    @patch("gsim.gcloud.sim")
    def test_fdtd_uses_string_when_sdk_enum_is_missing(self, mock_sim):
        """FDTD works with SDK 1.8.x without modifying its enum."""
        from enum import Enum

        from gsim.gcloud import _get_job_definition

        class LegacyJobDefinition(Enum):
            MEEP = "meep"
            PALACE = "palace"

        mock_sim.JobDefinition = LegacyJobDefinition

        assert _get_job_definition("fdtd") == "fdtd"

    @patch("gsim.gcloud.sim")
    def test_existing_sdk_enum_is_preserved(self, mock_sim):
        """Existing solver definitions still use the SDK enum."""
        from enum import Enum

        from gsim.gcloud import _get_job_definition

        class LegacyJobDefinition(Enum):
            MEEP = "meep"
            PALACE = "palace"

        mock_sim.JobDefinition = LegacyJobDefinition

        assert _get_job_definition("meep") is LegacyJobDefinition.MEEP


# ---------------------------------------------------------------------------
# register_result_parser
# ---------------------------------------------------------------------------


class TestResultParserRegistry:
    """Tests for register_result_parser and _RESULT_PARSERS."""

    def test_register_and_lookup(self):
        """Registered parser is stored and callable."""
        from gsim.gcloud import _RESULT_PARSERS, register_result_parser

        sentinel = object()
        register_result_parser("test_solver", lambda _r: sentinel)
        assert "test_solver" in _RESULT_PARSERS
        assert _RESULT_PARSERS["test_solver"](None) is sentinel  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        del _RESULT_PARSERS["test_solver"]

    def test_overwrite(self):
        """Later registration overwrites earlier one."""
        from gsim.gcloud import _RESULT_PARSERS, register_result_parser

        register_result_parser("overwrite_test", lambda _r: 1)
        register_result_parser("overwrite_test", lambda _r: 2)
        assert _RESULT_PARSERS["overwrite_test"](None) == 2  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
        del _RESULT_PARSERS["overwrite_test"]


# ---------------------------------------------------------------------------
# SDK layout compatibility
# ---------------------------------------------------------------------------


class TestSdkLayoutCompatibility:
    """Tests for SDK helpers that moved between 1.x and 2.x."""

    @patch("gsim.gcloud.sim")
    def test_get_job_logs_from_sdk_1_module(self, mock_sim):
        """SDK 1.x exposes the log fetcher directly on sim."""
        from gsim.gcloud import _get_job_logs_callable

        fetch_logs = MagicMock()
        mock_sim._get_job_logs = fetch_logs

        assert _get_job_logs_callable() is fetch_logs

    @patch("gsim.gcloud.sim")
    def test_get_job_logs_from_sdk_2_web_module(self, mock_sim):
        """SDK 2.x exposes the log fetcher from the nested web module."""
        from gsim.gcloud import _get_job_logs_callable

        fetch_logs = MagicMock()
        mock_sim._get_job_logs = None
        mock_sim.web._get_job_logs = fetch_logs

        assert _get_job_logs_callable() is fetch_logs


# ---------------------------------------------------------------------------
# upload()
# ---------------------------------------------------------------------------


class TestUpload:
    """Tests for upload()."""

    @patch("gsim.gcloud.upload_simulation_dir")
    def test_returns_job_id(self, mock_upload_dir, tmp_path):
        """Upload returns the job_id from the SDK."""
        from gsim.gcloud import upload

        mock_upload_dir.return_value = PreJob(job_id="job-xyz", job_name="palace-xyz")
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.json").write_text("{}")

        job_id = upload(config_dir, "palace", verbose=True)
        assert job_id == "job-xyz"
        mock_upload_dir.assert_called_once_with(config_dir, "palace")

    def test_missing_dir(self, tmp_path):
        """Upload raises FileNotFoundError for missing directory."""
        from gsim.gcloud import upload

        with pytest.raises(FileNotFoundError):
            upload(tmp_path / "nonexistent", "palace", verbose=False)


# ---------------------------------------------------------------------------
# start()
# ---------------------------------------------------------------------------


class TestStart:
    """Tests for start()."""

    @patch("gsim.gcloud.sim")
    def test_returns_job_name(self, mock_sim):
        """Start returns the job_name from the started job."""
        from gsim.gcloud import start

        fake_job = FakeJob(job_name="palace-started")
        mock_sim.start_simulation.return_value = fake_job

        name = start("job-abc", verbose=True)
        assert name == "palace-started"
        call_args = mock_sim.start_simulation.call_args
        pre_job = call_args[0][0]
        assert pre_job.job_id == "job-abc"


# ---------------------------------------------------------------------------
# get_status()
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Tests for get_status()."""

    @patch("gsim.gcloud.sim")
    def test_returns_status_string(self, mock_sim):
        """Returns lowercase status string."""
        from gsim.gcloud import get_status

        fake_job = FakeJob(status=SimStatus.RUNNING)
        mock_sim.get_job.return_value = fake_job

        status = get_status("job-abc")
        assert status == "running"
        mock_sim.get_job.assert_called_once_with("job-abc")

    @patch("gsim.gcloud.sim")
    def test_returns_sdk_2_string_status(self, mock_sim):
        """SDK 2.x string statuses are returned without enum conversion."""
        from gsim.gcloud import get_status

        mock_sim.get_job.return_value = FakeJob(status="running")

        assert get_status("job-abc") == "running"


# ---------------------------------------------------------------------------
# wait_for_results() — single job
# ---------------------------------------------------------------------------


class TestWaitForResultsSingle:
    """Tests for wait_for_results with a single job."""

    @patch("gsim.gcloud._output_mode", new=lambda: "pipe")
    @patch("gsim.gcloud.sim")
    def test_status_display_includes_progress_bar(self, mock_sim, capsys):
        """Status mode renders a lifecycle progress estimate and elapsed time."""
        from gsim.gcloud import _print_status_table

        mock_sim.SimStatus = SimStatus
        job = FakeJob(job_name="meep-progress", status=SimStatus.RUNNING)

        _print_status_table({"job-1": job}, {"job-1": 0.0}, final=True)

        output = capsys.readouterr().out
        assert "[##########..........]  50%" in output
        assert "running" in output
        assert "elapsed" in output

    @patch("gsim.gcloud._output_mode", new=lambda: "pipe")
    @patch("gsim.gcloud.time.monotonic", return_value=120.0)
    def test_status_display_estimates_remaining_runtime(self, mock_monotonic, capsys):
        """Input-size runtime estimates drive running progress and ETA output."""
        from gsim.gcloud import _print_status_table

        job = FakeJob(job_name="meep-progress", status=SimStatus.RUNNING)
        _print_status_table(
            {"job-1": job},
            {"job-1": 0.0},
            final=True,
            estimated_runtime_seconds=240.0,
        )

        output = capsys.readouterr().out
        assert "[##########..........]  50%" in output
        assert "ETA 2m 00s" in output
        assert mock_monotonic.called

    @patch("gsim.gcloud._output_mode", new=lambda: "pipe")
    def test_status_display_never_moves_backwards(self, capsys):
        """Progress rises through queued, starting and running (issue #340)."""
        from gsim.gcloud import _print_status_table

        shown: dict[str, int] = {}
        percents = []
        for status in (SimStatus.QUEUED, SimStatus.STARTING, SimStatus.RUNNING):
            job = FakeJob(job_name="meep-progress", status=status)
            _print_status_table(
                {"job-1": job}, {"job-1": 0.0}, final=True, shown_progress=shown
            )
            percents.append(shown["job-1"])

        output = capsys.readouterr().out
        assert percents == [15, 30, 50]
        assert "starting" in output

    @patch("gsim.gcloud._output_mode", new=lambda: "pipe")
    @patch("gsim.gcloud.time.monotonic", return_value=30.0)
    def test_running_estimate_does_not_undercut_startup(self, mock_monotonic, capsys):
        """An early running estimate below the startup value holds the bar."""
        from gsim.gcloud import _print_status_table

        shown = {"job-1": 30}
        job = FakeJob(job_name="meep-progress", status=SimStatus.RUNNING)
        _print_status_table(
            {"job-1": job},
            {"job-1": 0.0},
            final=True,
            estimated_runtime_seconds=300.0,
            shown_progress=shown,
        )

        assert shown["job-1"] == 30
        assert mock_monotonic.called
        assert " 30%" in capsys.readouterr().out

    @patch("gsim.gcloud.sim")
    def test_already_completed(self, mock_sim, tmp_path):
        """Completed job downloads and parses results immediately."""
        from gsim.gcloud import _RESULT_PARSERS, wait_for_results

        fake_job = FakeJob(
            id="job-1",
            job_name="palace-done",
            job_def_name="prod-palace-simulation",
            status=SimStatus.COMPLETED,
            exit_code=0,
        )
        mock_sim.SimStatus = SimStatus
        mock_sim.get_job.return_value = fake_job

        output_file = tmp_path / "result.csv"
        output_file.write_text("data")
        mock_sim.download_results.return_value = {"output": output_file}

        _RESULT_PARSERS["palace"] = lambda rr: {"parsed": True, "files": rr.files}
        try:
            result = wait_for_results("job-1", verbose="quiet", parent_dir=tmp_path)
            assert result["parsed"] is True
            assert "result.csv" in result["files"]
        finally:
            del _RESULT_PARSERS["palace"]

    @patch("gsim.gcloud.sim")
    def test_already_completed_with_sdk_2_string_status(self, mock_sim, tmp_path):
        """SDK 2.x string terminal statuses are recognized."""
        from gsim.gcloud import wait_for_results

        mock_sim.SimStatus = SimStatus
        mock_sim.get_job.return_value = FakeJob(
            job_def_name="unknown", status="completed"
        )
        output_file = tmp_path / "result.csv"
        output_file.write_text("data")
        mock_sim.download_results.return_value = {"output": output_file}

        result = wait_for_results("job-1", verbose="quiet", parent_dir=tmp_path)

        assert "result.csv" in result.files

    @patch("gsim.gcloud.sim")
    def test_failed_before_start_reports_backend_reason(self, mock_sim, tmp_path):
        """Pre-start failures raise their backend reason without downloading."""
        from gsim.gcloud import wait_for_results

        mock_sim.SimStatus = SimStatus
        mock_sim.get_job.return_value = FakeJob(
            status="failed",
            exit_code=None,
            download_urls={"results": "https://example.com/missing-results.tar.gz"},
            status_reason="CannotPullContainerError",
            detail_reason="no space left on device",
        )

        with pytest.raises(RuntimeError) as exc_info:
            wait_for_results("job-1", verbose="quiet", parent_dir=tmp_path)

        message = str(exc_info.value)
        assert "Simulation failed before producing an exit code" in message
        assert "Status: failed" in message
        assert "Reason: CannotPullContainerError" in message
        assert "Details: no space left on device" in message
        mock_sim.download_results.assert_not_called()

    @patch("gsim.gcloud.sim")
    def test_list_input(self, mock_sim, tmp_path):
        """wait_for_results(*[id]) works like wait_for_results(id)."""
        from gsim.gcloud import wait_for_results

        fake_job = FakeJob(id="job-1", job_name="meep-x", status=SimStatus.COMPLETED)
        mock_sim.SimStatus = SimStatus
        mock_sim.get_job.return_value = fake_job

        output_file = tmp_path / "s_parameters.csv"
        output_file.write_text("data")
        mock_sim.download_results.return_value = {"output": output_file}

        result = wait_for_results(["job-1"], verbose="status", parent_dir=tmp_path)
        # Single element list -> single result, not a list
        assert not isinstance(result, list)

    def test_empty_raises(self):
        """Raises ValueError when no job_ids provided."""
        from gsim.gcloud import wait_for_results

        with pytest.raises(ValueError, match="At least one job_id"):
            wait_for_results(verbose="quiet")


class TestRuntimeEstimate:
    """Tests for the pre-run input-size runtime heuristic."""

    def test_larger_mesh_has_a_larger_runtime_estimate(self, tmp_path):
        """Mesh byte size is used as a solver-agnostic size proxy."""
        from gsim.gcloud import estimate_runtime_seconds

        small = tmp_path / "small"
        large = tmp_path / "large"
        small.mkdir()
        large.mkdir()
        (small / "mesh.msh").write_bytes(b"0" * 1024)
        (large / "mesh.msh").write_bytes(b"0" * (10 * 1024 * 1024))

        assert estimate_runtime_seconds(large) > estimate_runtime_seconds(small)


# ---------------------------------------------------------------------------
# wait_for_results() — multiple jobs
# ---------------------------------------------------------------------------


class TestWaitForResultsMulti:
    """Tests for wait_for_results with multiple jobs."""

    @patch("gsim.gcloud.time.sleep")
    @patch("gsim.gcloud.sim")
    def test_mixed_statuses(self, mock_sim, mock_sleep, tmp_path):  # noqa: ARG002
        """Polls until all jobs complete, then returns results."""
        from gsim.gcloud import wait_for_results

        mock_sim.SimStatus = SimStatus

        job1 = FakeJob(id="job-1", job_name="palace-1", status=SimStatus.COMPLETED)
        job2_running = FakeJob(
            id="job-2", job_name="meep-2", status=SimStatus.RUNNING, exit_code=None
        )
        job2_done = FakeJob(id="job-2", job_name="meep-2", status=SimStatus.COMPLETED)

        poll_count = {"job-2": 0}

        def fake_get_job(jid):
            if jid == "job-1":
                return job1
            poll_count["job-2"] += 1
            return job2_done if poll_count["job-2"] > 1 else job2_running

        mock_sim.get_job.side_effect = fake_get_job

        dl_count = [0]

        def fake_download(_job, *, output_dir):  # noqa: ARG001
            dl_count[0] += 1
            f = tmp_path / f"out{dl_count[0]}.csv"
            f.write_text("x")
            return {"output": f}

        mock_sim.download_results.side_effect = fake_download

        results = wait_for_results(
            "job-1", "job-2", verbose="status", parent_dir=tmp_path
        )
        assert isinstance(results, list)
        assert len(results) == 2

    @patch("gsim.gcloud.time.sleep")
    @patch("gsim.gcloud.sim")
    def test_list_of_ids(self, mock_sim, mock_sleep, tmp_path):  # noqa: ARG002
        """wait_for_results(*[id1, id2]) returns a list."""
        from gsim.gcloud import wait_for_results

        mock_sim.SimStatus = SimStatus

        job1 = FakeJob(id="j1", job_name="p-1", status=SimStatus.COMPLETED)
        job2 = FakeJob(id="j2", job_name="p-2", status=SimStatus.COMPLETED)
        mock_sim.get_job.side_effect = lambda jid: job1 if jid == "j1" else job2

        dl_count = [0]

        def fake_download(_job, *, output_dir):  # noqa: ARG001
            dl_count[0] += 1
            f = tmp_path / f"dl{dl_count[0]}.csv"
            f.write_text("x")
            return {"output": f}

        mock_sim.download_results.side_effect = fake_download

        results = wait_for_results(*["j1", "j2"], verbose="quiet", parent_dir=tmp_path)
        assert isinstance(results, list)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# run_simulation() backward compat
# ---------------------------------------------------------------------------


class TestRunSimulationBackwardCompat:
    """Tests for backward-compatible run_simulation()."""

    @patch("gsim.gcloud.sim")
    def test_run_simulation_still_works(self, mock_sim, tmp_path):
        """Legacy run_simulation returns RunResult with files."""
        from gsim.gcloud import run_simulation

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.json").write_text("{}")

        pre_job = PreJob(job_id="job-bc", job_name="palace-bc")
        mock_sim.upload_simulation.return_value = pre_job
        mock_sim.JobDefinition.PALACE = "palace"

        started_job = FakeJob(
            id="job-bc", job_name="palace-bc", status=SimStatus.RUNNING
        )
        mock_sim.start_simulation.return_value = started_job

        finished_job = FakeJob(
            id="job-bc", job_name="palace-bc", status=SimStatus.COMPLETED
        )
        mock_sim.wait_for_simulation.return_value = finished_job

        out_file = tmp_path / "result.csv"
        out_file.write_text("data")
        mock_sim.download_results.return_value = {"output": out_file}

        result = run_simulation(
            config_dir=config_dir,
            job_type="palace",
            verbose=True,
            parent_dir=tmp_path,
        )
        assert result.job_name == "palace-bc"
        assert "result.csv" in result.files

    @patch("gsim.gcloud.sim")
    def test_pre_start_failure_reports_backend_reason(self, mock_sim, tmp_path):
        """Legacy API reports infrastructure failures without downloading."""
        from gsim.gcloud import run_simulation

        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "config.json").write_text("{}")

        pre_job = PreJob(job_id="job-bc", job_name="palace-bc")
        mock_sim.upload_simulation.return_value = pre_job
        mock_sim.JobDefinition.PALACE = "palace"

        started_job = FakeJob(
            id="job-bc", job_name="palace-bc", status=SimStatus.RUNNING
        )
        mock_sim.start_simulation.return_value = started_job
        mock_sim.wait_for_simulation.return_value = FakeJob(
            id="job-bc",
            job_name="palace-bc",
            status="failed",
            exit_code=None,
            download_urls={"results": "https://example.com/missing-results.tar.gz"},
            status_reason="CannotPullContainerError",
            detail_reason="no space left on device",
        )

        with pytest.raises(RuntimeError, match="CannotPullContainerError") as exc_info:
            run_simulation(
                config_dir=config_dir,
                job_type="palace",
                verbose=False,
                parent_dir=tmp_path,
            )

        assert "Details: no space left on device" in str(exc_info.value)
        mock_sim.download_results.assert_not_called()


# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------


class TestModuleLevelExports:
    """Tests for gsim top-level exports."""

    def test_gsim_exports_get_status(self):
        """gsim.get_status is accessible."""
        import gsim

        assert hasattr(gsim, "get_status")
        assert callable(gsim.get_status)

    def test_gsim_exports_wait_for_results(self):
        """gsim.wait_for_results is accessible."""
        import gsim

        assert hasattr(gsim, "wait_for_results")
        assert callable(gsim.wait_for_results)


# ---------------------------------------------------------------------------
# _handle_failed_job
# ---------------------------------------------------------------------------


class TestHandleFailedJob:
    """Tests for the error raised when a cloud job fails."""

    def _fail(self, tmp_path, **kwargs):
        """Run _handle_failed_job on a failed job and return the message."""
        from gsim.gcloud import _handle_failed_job

        job = FakeJob(status=SimStatus.FAILED, **kwargs)
        with pytest.raises(RuntimeError) as exc:
            _handle_failed_job(job, tmp_path, verbose=False)
        return str(exc.value)

    def test_reports_missing_artifacts(self, tmp_path):
        """An empty download set is stated, not left as an unexplained blank."""
        message = self._fail(tmp_path, exit_code=1, download_urls={})
        assert "exit code 1" in message
        assert "No output artifacts" in message

    def test_explains_signal_exit_code(self, tmp_path):
        """128+N exit codes get a signal explanation instead of a bare number."""
        message = self._fail(tmp_path, exit_code=134, download_urls={})
        assert "SIGABRT" in message

    def test_explains_oom_kill(self, tmp_path):
        """SIGKILL is called out as a memory limit."""
        assert "memory limit" in self._fail(tmp_path, exit_code=137, download_urls={})

    def test_no_note_for_plain_failure(self, tmp_path):
        """An ordinary non-signal exit code gets no signal note."""
        assert "SIGABRT" not in self._fail(tmp_path, exit_code=1, download_urls={})

    def test_surfaces_unconventionally_named_log(self, tmp_path):
        """Any *.log among the artifacts is shown when the known names are absent."""
        log = tmp_path / "solver-run.log"
        log.write_text("line one\nfatal: mesh is degenerate\n", encoding="utf-8")

        with patch("gsim.gcloud.sim.download_results", return_value={"log": log}):
            message = self._fail(tmp_path, exit_code=1, download_urls={"a": "url"})

        assert "solver-run.log" in message
        assert "fatal: mesh is degenerate" in message
