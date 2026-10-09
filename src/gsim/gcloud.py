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
import functools
import importlib
import io
import json
import logging
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from gdsfactoryplus import sim

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def _status_value(status: Any) -> str:
    """Return a status string for SDK 1.x enums and SDK 2.x strings."""
    return str(getattr(status, "value", status))


def _job_failed(job: Any) -> bool:
    """Return whether a finished job reports a failed outcome.

    Infrastructure failures can happen before a container starts, in which case
    the SDK reports ``status=failed`` without an exit code.  Solver failures have
    a non-zero exit code, so preserve that signal as well.
    """
    status = _status_value(getattr(job, "status", "")).casefold()
    exit_code = getattr(job, "exit_code", None)
    return status == "failed" or (exit_code is not None and exit_code != 0)


def _get_job_logs_callable() -> Callable[..., dict[str, Any]] | None:
    """Return the SDK log fetcher across the 1.x module and 2.x package layouts."""
    get_logs = getattr(sim, "_get_job_logs", None)
    if get_logs is not None:
        return cast("Callable[..., dict[str, Any]]", get_logs)

    sim_web = getattr(sim, "web", None)
    return cast(
        "Callable[..., dict[str, Any]] | None",
        getattr(sim_web, "_get_job_logs", None),
    )


def _is_transient_error(exc: Exception) -> bool:
    """Return True if *exc* is a transient HTTP/network error worth retrying."""
    try:
        from httpx import ConnectError, HTTPStatusError, TimeoutException
    except ImportError:  # pragma: no cover
        return False

    if isinstance(exc, (TimeoutException, ConnectError)):
        return True
    return isinstance(exc, HTTPStatusError) and exc.response.status_code >= 500


def _is_forbidden_error(exc: Exception) -> bool:
    """Return True if *exc* is an HTTP 403 Forbidden error."""
    try:
        from httpx import HTTPStatusError
    except ImportError:  # pragma: no cover
        return False

    return isinstance(exc, HTTPStatusError) and exc.response.status_code == 403


__all__ = [
    "CloudSimulationNotEnabledError",
    "RunResult",
    "check_cache",
    "check_cache_for_dir",
    "get_status",
    "print_job_summary",
    "register_result_parser",
    "run_simulation",
    "start",
    "upload",
    "upload_simulation_dir",
    "wait_for_results",
]


@dataclass
class RunResult:
    """Result of a cloud simulation run.

    Attributes:
        sim_dir: Root directory (``{job_type}_{job_name}/``).
        files: Flat mapping of filename -> Path inside ``output/``.
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

    Handles formats like ``"prod-meep-simulation"`` -> ``"meep"``,
    ``"prod-palace-simulation"`` -> ``"palace"``, or plain ``"meep"``.
    """
    name = getattr(job, "job_def_name", "") or ""
    # Try known solver names in the definition string
    for solver in ("meep", "palace", "femwell", "fdtd"):
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
    """Flatten gdsfactoryplus download results to a filename -> Path dict.

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


# Container exit codes in the 128+N range mean the process died on signal N.
# Worth spelling out: these look like solver errors but are usually resource
# limits, and the solver rarely gets a chance to write a log.
_SIGNAL_EXIT_CODES = {
    134: "exit 134 is SIGABRT — an assertion failure or out-of-memory abort. "
    "Try a coarser mesh, fewer frequency points, or a larger instance.",
    137: "exit 137 is SIGKILL — almost always the container hitting its "
    "memory limit. Try a coarser mesh or a larger instance.",
    139: "exit 139 is SIGSEGV — a solver crash. Please report the mesh and "
    "config that reproduce it.",
}


def _handle_failed_job(job, output_dir: Path, verbose: bool) -> None:
    """Handle a failed simulation job by downloading logs and raising informative error.

    Args:
        job: The finished Job object with non-zero exit code
        output_dir: Directory to download logs to
        verbose: Whether to print progress

    Raises:
        RuntimeError: Always raised with detailed error information
    """
    exit_code = getattr(job, "exit_code", None)
    if exit_code is None:
        error_parts = ["Simulation failed before producing an exit code"]
    else:
        error_parts = [f"Simulation failed with exit code {exit_code}"]

    error_parts.append(f"Status: {_status_value(job.status)}")

    status_reason = getattr(job, "status_reason", None)
    detail_reason = getattr(job, "detail_reason", None)
    if status_reason:
        error_parts.append(f"Reason: {status_reason}")
    if detail_reason:
        error_parts.append(f"Details: {detail_reason}")

    if exit_code in _SIGNAL_EXIT_CODES:
        error_parts.append(f"Note: {_SIGNAL_EXIT_CODES[exit_code]}")

    # Try to download logs even though job failed
    try:
        if exit_code is not None and not job.download_urls:
            error_parts.append(
                "No output artifacts were uploaded, so no solver log is "
                "available — the container most likely died before writing "
                "any output."
            )
        elif exit_code is not None:
            if verbose:
                print("Downloading logs from failed job...")  # noqa: T201

            raw_results = sim.download_results(job, output_dir=output_dir)
            all_files = _flatten_results(raw_results)

            # Look for log files and display them. Fall back to any *.log so a
            # solver that names its log differently is still surfaced.
            log_files = ["palace.log", "stdout.log", "stderr.log", "output.log"]
            log_name = next(
                (name for name in log_files if name in all_files),
                next(
                    (name for name in sorted(all_files) if name.endswith(".log")),
                    None,
                ),
            )
            if log_name is not None:
                content = all_files[log_name].read_text()
                error_parts.append(f"\n--- {log_name} (last 100 lines) ---")
                lines = content.strip().split("\n")
                error_parts.append("\n".join(lines[-100:]))
            else:
                error_parts.append(
                    f"No log file among the downloaded artifacts "
                    f"({', '.join(sorted(all_files)) or 'none'})."
                )

            if verbose and all_files:
                print(f"Logs downloaded to {output_dir}")  # noqa: T201

    except Exception as e:
        error_parts.append(f"(Failed to download logs: {type(e).__name__}: {e})")

    raise RuntimeError("\n".join(error_parts))


def _get_job_definition(job_type: str):
    """Get the SDK job definition, with support for newer gsim solvers."""
    normalized_job_type = job_type.casefold()
    job_type_upper = normalized_job_type.upper()
    if not hasattr(sim.JobDefinition, job_type_upper):
        # gdsfactoryplus 1.8.x accepts strings but its enum predates FDTD.
        # Keep that SDK usable until users are ready to upgrade it.
        if normalized_job_type == "fdtd":
            return normalized_job_type
        valid = [e.name.casefold() for e in sim.JobDefinition]
        valid.append("fdtd")
        raise ValueError(f"Unknown job type '{job_type}'. Valid types: {valid}")
    return getattr(sim.JobDefinition, job_type_upper)


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------


def _sdk_accepts(func: Any, param: str) -> bool:
    """Return True if *func* accepts a keyword argument named *param*.

    Used to stay compatible with SDK versions released before the caching
    parameters existed.
    """
    import inspect

    try:
        return param in inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins / C funcs
        return False


@functools.cache
def _warn_cache_unsupported() -> None:
    """Warn once per process that the installed SDK cannot look up the cache."""
    logger.warning(
        "Installed gdsfactoryplus has no check_cache(); cache lookups are "
        "disabled. Upgrade gdsfactoryplus to reuse completed jobs without "
        "re-uploading their inputs."
    )


def check_cache(job_type: str, input_hash: str) -> str | None:
    """Look up a previously completed job with identical inputs.

    Never raises: a cache lookup is an optimization, so an unsupported SDK,
    a transient network error, or a server error all degrade to a miss and
    the caller submits the job normally. Those degraded paths are logged at
    ``WARNING`` — silently returning a miss makes a permanently broken
    lookup indistinguishable from a cold cache.

    Args:
        job_type: Solver name, e.g. ``"meep"`` or ``"palace"``.
        input_hash: Value from :func:`gsim.hashing.compute_input_hash`.

    Returns:
        ``job_id`` of the cached job, or ``None`` on a miss.
    """
    sdk_check = getattr(sim, "check_cache", None)
    if sdk_check is None:
        _warn_cache_unsupported()
        return None

    try:
        result = sdk_check(
            job_definition=_get_job_definition(job_type),
            input_hash=input_hash,
        )
    except Exception as exc:
        logger.warning(
            "Cache lookup for %s failed (%s: %s); submitting normally",
            job_type,
            type(exc).__name__,
            exc,
        )
        return None

    if not getattr(result, "cached", False):
        logger.debug("Cache miss for %s %s", job_type, input_hash)
        return None
    return getattr(result, "job_id", None)


def check_cache_for_dir(input_dir: str | Path, job_type: str) -> tuple[str, str | None]:
    """Hash a prepared input directory and look it up in the cloud cache.

    Args:
        input_dir: Directory holding the files that would be uploaded.
        job_type: Solver name, e.g. ``"meep"`` or ``"palace"``.

    Returns:
        ``(input_hash, job_id)`` where ``job_id`` is ``None`` on a cache
        miss. The hash is returned either way so the caller can pass it to
        :func:`upload` and populate the cache for the next run.
    """
    from gsim.hashing import compute_input_hash

    input_hash = compute_input_hash(input_dir, job_type)
    return input_hash, check_cache(job_type, input_hash)


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
        RuntimeError: If the job failed or has a non-zero exit code.
    """
    root = Path(parent_dir) if parent_dir else Path.cwd()
    sim_dir = root / f"sim-data-{job.job_name}"
    sim_dir.mkdir(parents=True, exist_ok=True)

    # Check status
    if _job_failed(job):
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
    input_hash: str | None = None,
) -> str:
    """Upload simulation files to the cloud. Does NOT start execution.

    Args:
        config_dir: Directory containing simulation config files.
        job_type: Simulation type (e.g. ``"palace"``, ``"meep"``).
        verbose: Print progress messages.
        input_hash: Optional cache key from
            :func:`gsim.hashing.compute_input_hash`. Recorded server-side so
            a later identical run can be served from cache. Ignored by SDK
            versions that predate the caching API.

    Returns:
        ``job_id`` string that can be passed to :func:`start`,
        :func:`get_status`, or :func:`wait_for_results`.
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    if verbose:
        print("Uploading simulation... ", end="", flush=True)  # noqa: T201

    # Only forward input_hash when set, keeping the call shape unchanged for
    # callers that don't use caching.
    extra = {"input_hash": input_hash} if input_hash is not None else {}
    pre_job = upload_simulation_dir(config_dir, job_type, **extra)

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
    return _status_value(job.status)


def _fetch_logs(job_id: str, cursor: str | None) -> tuple[list[str], str | None]:
    """Fetch a page of log messages. Returns (messages, next_cursor)."""
    _get_logs = _get_job_logs_callable()
    if _get_logs is None:
        return [], cursor
    _logs_unavailable = getattr(sim, "LogsNotAvailableError", Exception)
    try:
        page = _get_logs(job_id, cursor=cursor, limit=50)
    except _logs_unavailable:
        return [], cursor
    except Exception:
        return [], cursor
    else:
        # Strip leading timestamps like "[2026-03-18T15:05:33.217Z] "
        msgs = [
            re.sub(r"^\[\d{4}-\d{2}-\d{2}T[\d:.]+Z\]\s*", "", entry.get("message", ""))
            for entry in page["items"]
        ]
        next_cursor = page["page_info"].get("next_cursor") or cursor
        return msgs, next_cursor


def _fetch_and_print_logs(
    job_id: str,
    cursor: str | None,
    *,
    drain: bool = False,
) -> str | None:
    """Fetch a page of logs and print them. Returns the next cursor.

    When *drain* is True, fetch all remaining pages (for final flush).
    """
    while True:
        msgs, cursor = _fetch_logs(job_id, cursor)
        for msg in msgs:
            print(f"  {msg}")  # noqa: T201
        if not drain or not msgs:
            break
    return cursor


def estimate_runtime_seconds(input_dir: str | Path) -> float:
    """Estimate cloud solver runtime from the generated mesh/input size.

    Mesh size is the best solver-agnostic proxy available before execution:
    finer meshes produce larger discretized systems and generally take longer.
    The estimate deliberately excludes cloud-queue time.  It is a heuristic,
    not a scheduling guarantee, and is refined as the solver runs.
    """
    directory = Path(input_dir)
    sizes = [
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in {".msh", ".mesh"}
    ]
    input_bytes = sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file()
    )
    size_bytes = max([input_bytes, *sizes])
    size_mb = size_bytes / (1024 * 1024)
    return 60.0 + 45.0 * max(size_mb, 0.1) ** 0.75


def wait_for_results(
    *job_ids: str,
    verbose: Literal["quiet", "status", "full"] = "status",
    parent_dir: str | Path | None = None,
    poll_interval: float = 5.0,
    estimated_runtime_seconds: float | None = None,
) -> Any:
    """Wait for one or more jobs to finish, then download and parse results.

    Accepts job IDs as positional args or a single list/tuple::

        wait_for_results(id1, id2)
        wait_for_results([id1, id2])

    For a single job, returns the parsed result directly.
    For multiple jobs, returns a list of results (same order as input).

    Args:
        *job_ids: One or more job ID strings, or a single list/tuple of IDs.
        verbose: Output mode:
            ``"quiet"`` — no output.
            ``"status"`` — status line only (default).
            ``"full"`` — stream solver logs live (timestamps stripped).
        parent_dir: Where to create sim-data directories (default: cwd).
        poll_interval: Seconds between status polls (default 5.0).
        estimated_runtime_seconds: Expected solver runtime, excluding queue
            time. When provided, status output includes an estimated percent
            complete and remaining time.

    Returns:
        Parsed result (single job) or list of parsed results (multiple jobs).
    """
    # Support both varargs and a single list/tuple
    if len(job_ids) == 1 and isinstance(job_ids[0], (list, tuple)):
        job_ids = tuple(job_ids[0])

    if not job_ids:
        raise ValueError("At least one job_id is required")

    if verbose == "full" and _get_job_logs_callable() is None:
        import warnings

        warnings.warn(
            "Log streaming is not supported by this gdsfactoryplus SDK. "
            "Falling back to verbose='status'.",
            stacklevel=2,
        )
        verbose = "status"

    # Fetch initial job objects
    jobs: dict[str, Any] = {jid: sim.get_job(jid) for jid in job_ids}
    now = time.monotonic()
    start_times: dict[str, float] = dict.fromkeys(job_ids, now)
    end_times: dict[str, float] = {}
    terminal = {
        _status_value(sim.SimStatus.COMPLETED),
        _status_value(sim.SimStatus.FAILED),
    }

    # Freeze timer for any jobs already finished
    for jid, job in jobs.items():
        if _status_value(job.status) in terminal:
            end_times[jid] = now

    # Track how many lines we printed last time (for overwriting multi-job)
    prev_lines = 0

    # Log cursors for streaming (one per job)
    log_cursors: dict[str, str | None] = dict.fromkeys(job_ids, None)

    # Poll until all jobs reach a terminal state
    while not all(_status_value(j.status) in terminal for j in jobs.values()):
        if verbose == "status":
            prev_lines = _print_status_table(
                jobs,
                start_times,
                prev_lines,
                end_times=end_times,
                estimated_runtime_seconds=estimated_runtime_seconds,
            )

        time.sleep(poll_interval)

        for jid, job in jobs.items():
            if _status_value(job.status) not in terminal:
                try:
                    jobs[jid] = sim.get_job(jid)
                except Exception as exc:
                    if _is_transient_error(exc):
                        logger.debug("Transient error polling job %s: %s", jid, exc)
                        continue
                    raise
                # Stream logs when running
                if verbose == "full" and _status_value(
                    jobs[jid].status
                ) == _status_value(sim.SimStatus.RUNNING):
                    log_cursors[jid] = _fetch_and_print_logs(jid, log_cursors[jid])
                # Freeze timer when job reaches terminal state
                if _status_value(jobs[jid].status) in terminal:
                    end_times[jid] = time.monotonic()

    # Final log fetch — drain all remaining pages
    if verbose == "full":
        for jid in job_ids:
            _fetch_and_print_logs(jid, log_cursors[jid], drain=True)

    # Final status display — skip clear_output when logs were streamed
    if verbose == "status":
        _print_status_table(
            jobs,
            start_times,
            prev_lines,
            end_times=end_times,
            final=True,
            estimated_runtime_seconds=estimated_runtime_seconds,
        )

    # Download + parse all
    results = []
    for jid in job_ids:
        job = jobs[jid]
        run_result = _download_job(job, parent_dir, verbose != "quiet")
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
    estimated_runtime_seconds: float | None = None,
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

    def _progress(jid: str, job: Any) -> tuple[int, str]:
        """Return a conservative lifecycle-based completion estimate.

        The cloud API reports lifecycle states but not solver iteration counts.
        When an input-size estimate is available, use its elapsed fraction for
        the running phase; otherwise show only conservative lifecycle progress.
        """
        status = _status_value(job.status).casefold()
        if status == "completed":
            return 100, "complete"
        if status == "failed":
            return 100, "failed"
        if status == "running":
            if estimated_runtime_seconds is not None:
                elapsed = time.monotonic() - start_times[jid]
                percent = min(99, round(100 * elapsed / estimated_runtime_seconds))
                remaining = max(0, estimated_runtime_seconds - elapsed)
                mins, secs = divmod(round(remaining), 60)
                return percent, f"running, ETA {mins}m {secs:02d}s"
            return 50, "running"
        if status == "queued":
            return 15, "queued"
        return 5, status or "starting"

    def _bar(percent: int, width: int = 20) -> str:
        filled = round(width * percent / 100)
        return f"[{'#' * filled}{'.' * (width - filled)}] {percent:3d}%"

    lines_printed = 0
    n = len(jobs)

    if n == 1:
        jid, job = next(iter(jobs.items()))
        percent, phase = _progress(jid, job)
        msg = (
            f"  {job.job_name or jid}  {_bar(percent)}  {phase}"
            f"  elapsed {_elapsed(jid)}"
        )
        if mode == "tty":
            sys.stdout.write(f"\r{msg:<80s}")
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
        percent, phase = _progress(jid, job)
        line = (
            f"  {job.job_name or jid:<30s} {_bar(percent)} {phase:<12s}"
            f" elapsed {_elapsed(jid)}"
        )
        print(line)  # noqa: T201
        lines_printed += 1

    sys.stdout.flush()
    return lines_printed


# ---------------------------------------------------------------------------
# Public API — legacy / backward-compatible
# ---------------------------------------------------------------------------


class CloudSimulationNotEnabledError(Exception):
    """Raised when the user's account does not have cloud simulation enabled."""


def upload_simulation_dir(
    input_dir: str | Path,
    job_type: str,
    *,
    input_hash: str | None = None,
):
    """Upload a simulation directory for cloud execution.

    Args:
        input_dir: Directory containing simulation files
        job_type: Simulation type (e.g., "palace")
        input_hash: Optional cache key recorded with the job. Silently
            dropped when the installed SDK does not support it.

    Palace and FDTD metadata is read from root-level ``metadata.json`` and
    forwarded with the exact staged-file manifest when the SDK supports both
    fields. An older SDK warns and submits with fixed resources.

    Returns:
        PreJob object from gdsfactoryplus

    Raises:
        CloudSimulationNotEnabledError: If the account lacks cloud simulation access.
    """
    input_dir = Path(input_dir)
    job_definition = _get_job_definition(job_type)
    kwargs: dict[str, Any] = {}
    if input_hash is not None:
        if _sdk_accepts(sim.upload_simulation, "input_hash"):
            kwargs["input_hash"] = input_hash
        else:
            logger.debug(
                "Installed gdsfactoryplus does not accept input_hash; "
                "uploading without a cache key"
            )
    metadata_path = input_dir / "metadata.json"
    if job_type.casefold() in {"palace", "fdtd"} and metadata_path.is_file():
        if all(
            _sdk_accepts(sim.upload_simulation, name)
            for name in ("sizing_metadata", "input_manifest")
        ):
            from gsim.hashing import compute_input_manifest

            metadata = json.loads(metadata_path.read_text(encoding="utf8"))
            if (
                not isinstance(metadata, dict)
                or metadata.get("solver") != job_type.casefold()
            ):
                raise ValueError(
                    "Sizing metadata solver must match the submitted job type"
                )
            kwargs["sizing_metadata"] = metadata
            kwargs["input_manifest"] = compute_input_manifest(input_dir)
        else:
            logger.warning(
                "Installed gdsfactoryplus cannot forward sizing metadata and input "
                "manifest; dynamic sizing unavailable, submitting with fixed "
                "resources. "
                "Upgrade gdsfactoryplus to enable dynamic sizing."
            )
    try:
        return sim.upload_simulation(
            path=input_dir, job_definition=job_definition, **kwargs
        )
    except Exception as exc:
        if _is_forbidden_error(exc):
            raise CloudSimulationNotEnabledError(
                "Cloud simulation is not enabled for your account.\n"
                "Please contact support@gdsfactory.com or visit https://gdsfactory.com "
                "to enable cloud simulation access."
            ) from exc
        raise


def run_simulation(
    config_dir: str | Path,
    job_type: Literal["palace", "meep", "fdtd"] = "palace",
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
            f"Created: {created} | Now: {now} | "
            f"Status: {_status_value(finished_job.status)}"
        )

    # Check status
    if _job_failed(finished_job):
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
    print(  # noqa: T201
        f"{'Status:':<12} {_status_value(job.status)} (exit {job.exit_code})"
    )
    print(f"{'Duration:':<12} {duration}")  # noqa: T201
    mem_gb = job.requested_memory_mb // 1024
    print(f"{'Resources:':<12} {job.requested_cpu} CPU / {mem_gb} GB")  # noqa: T201
    print(f"{'Output:':<12} {size_str}")  # noqa: T201
    print(f"{'Files:':<12} {len(files)} files")  # noqa: T201
    for f in files:
        print(f"             - {f}")  # noqa: T201
