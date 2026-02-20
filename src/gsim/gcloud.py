"""GDSFactory+ cloud simulation interface.

This module provides an interface to run simulations on
the GDSFactory+ cloud infrastructure.

Usage:
    from gsim import gcloud

    # Blocking (default): upload + start + wait + download
    result = gcloud.run_simulation("./sim", job_type="palace")

    # Fine-grained control:
    job_id = gcloud.upload("./sim", job_type="palace")
    gcloud.start(job_id)
    gcloud.get_status(job_id)
    result = gcloud.wait_for_results(job_id)

    # Multi-job polling:
    results = gcloud.wait_for_results(id1, id2, id3)

    # Or use solver-specific wrappers:
    from gsim import palace as pa
    result = pa.run_simulation("./sim")
"""

from __future__ import annotations

import contextlib
import importlib
import io
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gdsfactoryplus import sim

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal


@dataclass
class RunResult:
    """Result of a cloud simulation run.

    Attributes:
        sim_dir: Root directory (``{job_type}_{job_name}/``).
        files: Flat mapping of filename → Path inside ``output/``.
        job_name: Cloud job identifier.
    """

    sim_dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    job_name: str = ""


# ---------------------------------------------------------------------------
# Result parser registry
# ---------------------------------------------------------------------------

_RESULT_PARSERS: dict[str, Callable[[RunResult], Any]] = {}


def register_result_parser(solver: str, parser: Callable[[RunResult], Any]) -> None:
    """Register a result parser for a solver type.

    Args:
        solver: Solver name (e.g. ``"meep"``, ``"palace"``).
        parser: Callable that takes a :class:`RunResult` and returns
            a solver-specific result object.
    """
    _RESULT_PARSERS[solver] = parser


def _extract_solver_from_job(job) -> str | None:
    """Extract the solver name from a Job's ``job_def_name``.

    Handles formats like ``"prod-meep-simulation"`` → ``"meep"``,
    ``"prod-palace-simulation"`` → ``"palace"``, or plain ``"meep"``.
    """
    name = getattr(job, "job_def_name", "") or ""
    # Try known solver names in the definition string
    for solver in ("meep", "palace", "femwell"):
        if solver in name.lower():
            return solver
    return None


def _get_result_parser(solver: str) -> Callable[[RunResult], Any] | None:
    """Look up a result parser, auto-importing the solver module if needed."""
    if solver not in _RESULT_PARSERS:
        # Auto-import gsim.{solver} to trigger registration
        with contextlib.suppress(ImportError):
            importlib.import_module(f"gsim.{solver}")
    return _RESULT_PARSERS.get(solver)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _flatten_results(raw_results: dict) -> dict[str, Path]:
    """Flatten gdsfactoryplus download results to a filename → Path dict.

    The SDK may return directories (extracted archives) or individual files.
    This walks everything and returns a flat mapping.
    """
    flat: dict[str, Path] = {}
    for result_path in raw_results.values():
        if result_path.is_dir():
            for file_path in result_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    flat[file_path.name] = file_path
        else:
            flat[result_path.name] = result_path
    return flat


def _handle_failed_job(job, output_dir: Path, verbose: bool) -> None:
    """Handle a failed simulation job by downloading logs and raising informative error.

    Args:
        job: The finished Job object with non-zero exit code
        output_dir: Directory to download logs to
        verbose: Whether to print progress

    Raises:
        RuntimeError: Always raised with detailed error information
    """
    error_parts = [
        f"Simulation failed with exit code {job.exit_code}",
        f"Status: {job.status.value}",
    ]

    if job.status_reason:
        error_parts.append(f"Reason: {job.status_reason}")
    if job.detail_reason:
        error_parts.append(f"Details: {job.detail_reason}")

    # Try to download logs even though job failed
    try:
        if job.download_urls:
            if verbose:
                print("Downloading logs from failed job...")  # noqa: T201

            raw_results = sim.download_results(job, output_dir=output_dir)
            all_files = _flatten_results(raw_results)

            # Look for log files and display them
            log_files = ["palace.log", "stdout.log", "stderr.log", "output.log"]
            for log_name in log_files:
                if log_name in all_files:
                    content = all_files[log_name].read_text()
                    error_parts.append(f"\n--- {log_name} (last 100 lines) ---")
                    lines = content.strip().split("\n")
                    error_parts.append("\n".join(lines[-100:]))
                    break

            if verbose and all_files:
                print(f"Logs downloaded to {output_dir}")  # noqa: T201

    except Exception as e:
        error_parts.append(f"(Failed to download logs: {e})")

    raise RuntimeError("\n".join(error_parts))


def _get_job_definition(job_type: str):
    """Get JobDefinition enum value by name."""
    job_type_upper = job_type.upper()
    if not hasattr(sim.JobDefinition, job_type_upper):
        valid = [e.name for e in sim.JobDefinition]
        raise ValueError(f"Unknown job type '{job_type}'. Valid types: {valid}")
    return getattr(sim.JobDefinition, job_type_upper)


def _download_job(job, parent_dir: str | Path | None, verbose: bool) -> RunResult:
    """Download results from a finished job.

    Creates ``sim-data-{job_name}/`` directory structure and downloads
    output files.

    Args:
        job: Finished Job object from the SDK.
        parent_dir: Where to create the sim directory (default: cwd).
        verbose: Print progress messages.

    Returns:
        RunResult with sim_dir, files, and job_name.

    Raises:
        RuntimeError: If the job failed (non-zero exit code).
    """
    root = Path(parent_dir) if parent_dir else Path.cwd()
    sim_dir = root / f"sim-data-{job.job_name}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Check status
    if job.exit_code is not None and job.exit_code != 0:
        _handle_failed_job(job, sim_dir, verbose)

    # Download directly into sim_dir
    raw_results = sim.download_results(job, output_dir=sim_dir)
    files = _flatten_results(raw_results)

    if verbose and files:
        print(f"Downloaded {len(files)} files to {sim_dir}")  # noqa: T201

    return RunResult(sim_dir=sim_dir, files=files, job_name=job.job_name)


def _parse_result(job, run_result: RunResult) -> Any:
    """Apply the registered result parser for this job's solver type.

    Falls back to the raw RunResult if no parser is registered.
    """
    solver = _extract_solver_from_job(job)
    if solver is None:
        return run_result

    parser = _get_result_parser(solver)
    if parser is None:
        return run_result

    return parser(run_result)


# ---------------------------------------------------------------------------
# Public API — fine-grained control
# ---------------------------------------------------------------------------


def upload(
    config_dir: str | Path,
    job_type: str,
    *,
    verbose: bool = True,
) -> str:
    """Upload simulation files to the cloud. Does NOT start execution.

    Args:
        config_dir: Directory containing simulation config files.
        job_type: Simulation type (e.g. ``"palace"``, ``"meep"``).
        verbose: Print progress messages.

    Returns:
        ``job_id`` string that can be passed to :func:`start`,
        :func:`get_status`, or :func:`wait_for_results`.
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    if verbose:
        print("Uploading simulation... ", end="", flush=True)  # noqa: T201

    pre_job = upload_simulation_dir(config_dir, job_type)

    if verbose:
        print(f"done (job_id: {pre_job.job_id})")  # noqa: T201

    return pre_job.job_id


def start(job_id: str, *, verbose: bool = True) -> str:
    """Start cloud execution for a previously uploaded job.

    Args:
        job_id: Job identifier returned by :func:`upload`.
        verbose: Print progress messages.

    Returns:
        The ``job_name`` (human-readable label).
    """
    from gdsfactoryplus.sim import PreJob

    pre_job = PreJob(job_id=job_id, job_name="")
    job = sim.start_simulation(pre_job)

    if verbose:
        print(f"Job started: {job.job_name}")  # noqa: T201

    return job.job_name


def get_status(job_id: str) -> str:
    """Get the current status of a cloud job.

    Args:
        job_id: Job identifier.

    Returns:
        Status string — one of ``"created"``, ``"queued"``,
        ``"running"``, ``"completed"``, ``"failed"``.
    """
    job = sim.get_job(job_id)
    return job.status.value


def wait_for_results(
    *job_ids: str,
    verbose: bool = True,
    parent_dir: str | Path | None = None,
    poll_interval: float = 5.0,
) -> Any:
    """Wait for one or more jobs to finish, then download and parse results.

    Accepts job IDs as positional args or a single list/tuple::

        wait_for_results(id1, id2)
        wait_for_results([id1, id2])

    For a single job, returns the parsed result directly.
    For multiple jobs, returns a list of results (same order as input).

    Args:
        *job_ids: One or more job ID strings, or a single list/tuple of IDs.
        verbose: Print progress messages.
        parent_dir: Where to create sim-data directories (default: cwd).
        poll_interval: Seconds between status polls (default 5.0).

    Returns:
        Parsed result (single job) or list of parsed results (multiple jobs).
    """
    # Support both varargs and a single list/tuple
    if len(job_ids) == 1 and isinstance(job_ids[0], (list, tuple)):
        job_ids = tuple(job_ids[0])

    if not job_ids:
        raise ValueError("At least one job_id is required")

    # Fetch initial job objects
    jobs: dict[str, Any] = {jid: sim.get_job(jid) for jid in job_ids}
    now = time.monotonic()
    start_times: dict[str, float] = dict.fromkeys(job_ids, now)
    end_times: dict[str, float] = {}
    terminal = {sim.SimStatus.COMPLETED, sim.SimStatus.FAILED}

    # Freeze timer for any jobs already finished
    for jid, job in jobs.items():
        if job.status in terminal:
            end_times[jid] = now

    # Track how many lines we printed last time (for overwriting multi-job)
    prev_lines = 0

    # Poll until all jobs reach a terminal state
    while not all(j.status in terminal for j in jobs.values()):
        if verbose:
            prev_lines = _print_status_table(
                jobs, start_times, prev_lines, end_times=end_times
            )
        time.sleep(poll_interval)
        for jid, job in jobs.items():
            if job.status not in terminal:
                jobs[jid] = sim.get_job(jid)
                # Freeze timer when job reaches terminal state
                if jobs[jid].status in terminal:
                    end_times[jid] = time.monotonic()

    # Final status display (with newline to finish the line)
    if verbose:
        _print_status_table(
            jobs, start_times, prev_lines, end_times=end_times, final=True
        )

    # Download + parse all
    results = []
    for jid in job_ids:
        job = jobs[jid]
        run_result = _download_job(job, parent_dir, verbose)
        results.append(_parse_result(job, run_result))

    return results[0] if len(job_ids) == 1 else results


def _output_mode() -> str:
    """Detect the output environment.

    Returns ``"jupyter"`` inside a Jupyter/IPython kernel (notebook or
    nbconvert), ``"tty"`` when stdout is a terminal, or ``"pipe"``
    otherwise (plain CI, redirected output).
    """
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        if ipy is not None and "IPKernelApp" in ipy.config:
            return "jupyter"
    except ImportError:
        pass

    import sys

    if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
        return "tty"

    return "pipe"


def _print_status_table(
    jobs: dict[str, Any],
    start_times: dict[str, float],
    prev_lines: int = 0,
    *,
    end_times: dict[str, float] | None = None,
    final: bool = False,
) -> int:
    """Print job status, updating in place.

    * **Jupyter / nbconvert** — uses ``clear_output(wait=True)`` so every
      poll replaces the previous output.  nbconvert only captures the
      *last* state, giving one clean line in rendered docs.
    * **TTY (terminal)** — uses carriage-return / ANSI cursor-up to overwrite.
    * **Pipe / plain CI** — only prints the final status.

    Returns the number of lines printed (for the TTY path to erase).
    """
    import sys

    _end_times = end_times or {}
    mode = _output_mode()

    # Pipe: only print at the end
    if mode == "pipe" and not final:
        return 0

    # Jupyter: clear previous cell output before printing
    if mode == "jupyter":
        from IPython.display import clear_output

        clear_output(wait=True)

    # TTY: move cursor up to overwrite previous output
    if mode == "tty" and prev_lines > 0:
        sys.stdout.write(f"\033[{prev_lines}A")

    def _elapsed(jid: str) -> str:
        t = _end_times.get(jid, time.monotonic()) - start_times[jid]
        mins, secs = divmod(int(t), 60)
        return f"{mins}m {secs:02d}s"

    lines_printed = 0
    n = len(jobs)

    if n == 1:
        jid, job = next(iter(jobs.items()))
        msg = f"  {job.job_name or jid}  {job.status.value}  {_elapsed(jid)}"
        if mode == "tty":
            sys.stdout.write(f"\r{msg:<60s}")
            if final:
                sys.stdout.write("\n")
        else:
            print(msg)  # noqa: T201
        sys.stdout.flush()
        return 1

    # Multi-job: header + one line per job
    print(f"Waiting for {n} jobs...")  # noqa: T201
    lines_printed += 1
    for jid, job in jobs.items():
        line = f"  {job.job_name or jid:<30s} {job.status.value:<12s} {_elapsed(jid)}"
        print(line)  # noqa: T201
        lines_printed += 1

    sys.stdout.flush()
    return lines_printed


# ---------------------------------------------------------------------------
# Public API — legacy / backward-compatible
# ---------------------------------------------------------------------------


def upload_simulation_dir(input_dir: str | Path, job_type: str):
    """Upload a simulation directory for cloud execution.

    Args:
        input_dir: Directory containing simulation files
        job_type: Simulation type (e.g., "palace")

    Returns:
        PreJob object from gdsfactoryplus
    """
    input_dir = Path(input_dir)
    job_definition = _get_job_definition(job_type)
    return sim.upload_simulation(path=input_dir, job_definition=job_definition)


def run_simulation(
    config_dir: str | Path,
    job_type: Literal["palace", "meep"] = "palace",
    verbose: bool = True,
    on_started: Callable | None = None,
    parent_dir: str | Path | None = None,
) -> RunResult:
    """Run a simulation on GDSFactory+ cloud (blocking).

    This function handles the complete workflow:
    1. Uploads simulation files from *config_dir*
    2. Starts the simulation job
    3. Creates a structured directory ``sim-data-{job_name}/``
       with ``input/`` (config files) and ``output/`` (results) sub-dirs
    4. Waits for completion
    5. Downloads results into ``output/``

    Args:
        config_dir: Directory containing the simulation config files.
        job_type: Type of simulation (default: "palace").
        verbose: Print progress messages (default True).
        on_started: Optional callback called with job object when simulation starts.
        parent_dir: Where to create the sim directory.
            Defaults to the current working directory.

    Returns:
        RunResult with sim_dir, files dict, and job_name.

    Raises:
        RuntimeError: If simulation fails

    Example:
        >>> result = gcloud.run_simulation("./sim", job_type="palace")
        Uploading simulation... done
        Job started: palace-abc123
        Waiting for completion... done (2m 34s)
        Downloading results... done
        >>> print(result.sim_dir)
        sim-data-palace-abc123/
    """
    config_dir = Path(config_dir)

    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    # Upload
    if verbose:
        print("Uploading simulation... ", end="", flush=True)  # noqa: T201

    pre_job = upload_simulation_dir(config_dir, job_type)

    if verbose:
        print("done")  # noqa: T201

    # Start
    job = sim.start_simulation(pre_job)

    if verbose:
        print(f"Job started: {job.job_name}")  # noqa: T201

    if on_started:
        on_started(job)

    # Create structured directory
    root = Path(parent_dir) if parent_dir else Path.cwd()
    sim_dir = root / f"sim-data-{job.job_name}"
    input_dir = sim_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Move config files into input/
    for item in list(config_dir.iterdir()):
        shutil.move(str(item), str(input_dir / item.name))
    # Remove now-empty config_dir (may fail if it was CWD, etc.)
    shutil.rmtree(config_dir, ignore_errors=True)

    # Wait (suppress per-poll prints from gdsfactoryplus SDK)
    with contextlib.redirect_stdout(io.StringIO()):
        finished_job = sim.wait_for_simulation(job)
    if verbose:
        created = finished_job.created_at.strftime("%H:%M:%S")
        from datetime import datetime

        now = datetime.now(finished_job.created_at.tzinfo).strftime("%H:%M:%S")
        print(  # noqa: T201
            f"Created: {created} | Now: {now} | Status: {finished_job.status.value}"
        )

    # Check status
    if finished_job.exit_code != 0:
        _handle_failed_job(finished_job, sim_dir, verbose)

    # Download directly into sim_dir (SDK creates results/ subdirectory)
    raw_results = sim.download_results(finished_job, output_dir=sim_dir)
    files = _flatten_results(raw_results)

    if verbose and files:
        print(f"Downloaded {len(files)} files to {sim_dir}")  # noqa: T201

    return RunResult(sim_dir=sim_dir, files=files, job_name=job.job_name)


def print_job_summary(job) -> None:
    """Print a formatted summary of a simulation job.

    Args:
        job: Job object from gdsfactoryplus
    """
    if job.started_at and job.finished_at:
        delta = job.finished_at - job.started_at
        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        duration = f"{minutes}m {seconds}s"
    else:
        duration = "N/A"

    size_kb = job.output_size_bytes / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.2f} MB"
    files = list(job.download_urls.keys()) if job.download_urls else []

    print(f"{'Job:':<12} {job.job_name}")  # noqa: T201
    print(f"{'Status:':<12} {job.status.value} (exit {job.exit_code})")  # noqa: T201
    print(f"{'Duration:':<12} {duration}")  # noqa: T201
    mem_gb = job.requested_memory_mb // 1024
    print(f"{'Resources:':<12} {job.requested_cpu} CPU / {mem_gb} GB")  # noqa: T201
    print(f"{'Output:':<12} {size_str}")  # noqa: T201
    print(f"{'Files:':<12} {len(files)} files")  # noqa: T201
    for f in files:
        print(f"             - {f}")  # noqa: T201
