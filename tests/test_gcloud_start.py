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
    status: SimStatus = SimStatus.COMPLETED
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
        assert _RESULT_PARSERS["test_solver"](None) is sentinel  # type: ignore[arg-type]
        del _RESULT_PARSERS["test_solver"]

    def test_overwrite(self):
        """Later registration overwrites earlier one."""
        from gsim.gcloud import _RESULT_PARSERS, register_result_parser

        register_result_parser("overwrite_test", lambda _r: 1)
        register_result_parser("overwrite_test", lambda _r: 2)
        assert _RESULT_PARSERS["overwrite_test"](None) == 2  # type: ignore[arg-type]
        del _RESULT_PARSERS["overwrite_test"]


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

        job_id = upload(config_dir, "palace", verbose=False)
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

        name = start("job-abc", verbose=False)
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


# ---------------------------------------------------------------------------
# wait_for_results() — single job
# ---------------------------------------------------------------------------


class TestWaitForResultsSingle:
    """Tests for wait_for_results with a single job."""

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
            result = wait_for_results("job-1", verbose=False, parent_dir=tmp_path)
            assert result["parsed"] is True
            assert "result.csv" in result["files"]
        finally:
            del _RESULT_PARSERS["palace"]

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

        result = wait_for_results(*["job-1"], verbose=False, parent_dir=tmp_path)
        # Single element list → single result, not a list
        assert not isinstance(result, list)

    def test_empty_raises(self):
        """Raises ValueError when no job_ids provided."""
        from gsim.gcloud import wait_for_results

        with pytest.raises(ValueError, match="At least one job_id"):
            wait_for_results(verbose=False)


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

        results = wait_for_results("job-1", "job-2", verbose=False, parent_dir=tmp_path)
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

        results = wait_for_results(*["j1", "j2"], verbose=False, parent_dir=tmp_path)
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
            verbose=False,
            parent_dir=tmp_path,
        )
        assert result.job_name == "palace-bc"
        assert "result.csv" in result.files


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
