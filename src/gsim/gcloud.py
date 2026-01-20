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

import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from gdsfactoryplus import sim  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal


def _get_job_definition(job_type: str):
    """Get JobDefinition enum value by name."""
    job_type_upper = job_type.upper()
    if not hasattr(sim.JobDefinition, job_type_upper):
        valid = [e.name for e in sim.JobDefinition]
        raise ValueError(f"Unknown job type '{job_type}'. Valid types: {valid}")
    return getattr(sim.JobDefinition, job_type_upper)


def upload_simulation_dir(input_dir: str | Path, job_type: str):
    """Zip all files in a directory and upload for simulation.

    Args:
        input_dir: Directory containing simulation files
        job_type: Simulation type (e.g., "palace")

    Returns:
        PreJob object from gdsfactoryplus
    """
    input_dir = Path(input_dir)
    zip_path = Path("_gsim_upload.zip")

    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for file in input_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(input_dir))

        job_definition = _get_job_definition(job_type)
        pre_job = sim.upload_simulation(path=zip_path, job_definition=job_definition)
    finally:
        if zip_path.exists():
            zip_path.unlink()

    return pre_job


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
        raise RuntimeError(
            f"Simulation failed with exit code {finished_job.exit_code}. "
            f"Status: {finished_job.status.value}"
        )

    # Download
    results = sim.download_results(finished_job)

    if verbose and results:
        first_path = next(iter(results.values()))
        download_dir = first_path.parent
        print(f"Downloaded {len(results)} files to {download_dir}")  # noqa: T201

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
