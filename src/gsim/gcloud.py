"""GDSFactory+ cloud simulation interface.

This module provides an interface to run simulations on
the GDSFactory+ cloud infrastructure.

Usage:
    from gsim import gcloud

    # Run simulation (uploads, starts, waits, downloads)
    results = gcloud.run_simulation("./sim", job_type="palace")

    # Or use solver-specific wrappers:
    from gsim import palace as pa
    results = pa.run_simulation("./sim")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gdsfactoryplus import sim

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal


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

            # Flatten results to find all files
            all_files: dict[str, Path] = {}
            for result_path in raw_results.values():
                if result_path.is_dir():
                    for file_path in result_path.rglob("*"):
                        if file_path.is_file() and not file_path.name.startswith("."):
                            all_files[file_path.name] = file_path
                else:
                    all_files[result_path.name] = result_path

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
    output_dir: str | Path,
    job_type: Literal["palace"] = "palace",
    verbose: bool = True,
    on_started: Callable | None = None,
) -> dict[str, Path]:
    """Run a simulation on GDSFactory+ cloud.

    This function handles the complete workflow:
    1. Uploads simulation files
    2. Starts the simulation job
    3. Waits for completion
    4. Downloads results

    Args:
        output_dir: Directory containing the simulation files
        job_type: Type of simulation (default: "palace")
        verbose: Print progress messages (default True)
        on_started: Optional callback called with job object when simulation starts

    Returns:
        Dict mapping result filename to local Path.

    Raises:
        RuntimeError: If simulation fails

    Example:
        >>> results = gcloud.run_simulation("./sim", job_type="palace")
        Uploading simulation... done
        Job started: palace-abc123
        Waiting for completion... done (2m 34s)
        Downloading results... done
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Upload
    if verbose:
        print("Uploading simulation... ", end="", flush=True)  # noqa: T201

    pre_job = upload_simulation_dir(output_dir, job_type)

    if verbose:
        print("done")  # noqa: T201

    # Start
    job = sim.start_simulation(pre_job)

    if verbose:
        print(f"Job started: {job.job_name}")  # noqa: T201

    if on_started:
        on_started(job)

    # Wait
    finished_job = sim.wait_for_simulation(job)

    # Check status
    if finished_job.exit_code != 0:
        _handle_failed_job(finished_job, output_dir, verbose)

    # Download
    raw_results = sim.download_results(
        finished_job, output_dir=f"sim-data-{finished_job.job_name}"
    )

    # Flatten results: gdsfactoryplus returns extracted directories,
    # but we want a dict of filename -> Path for individual files
    results: dict[str, Path] = {}
    for result_path in raw_results.values():
        if result_path.is_dir():
            # Recursively find all files in the extracted directory
            for file_path in result_path.rglob("*"):
                if file_path.is_file() and not file_path.name.startswith("."):
                    results[file_path.name] = file_path
        else:
            results[result_path.name] = result_path

    if verbose and results:
        # Find common parent directory for display
        first_path = next(iter(results.values()))
        print(f"Downloaded {len(results)} files to {first_path.parent}")  # noqa: T201

    return results


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
