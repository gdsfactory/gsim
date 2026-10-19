"""Cloud lifecycle mixin for an FDTD simulation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from gsim.fdtd.models import SimulationArtifacts


class CloudWorkflowMixin:
    """Upload, start, monitor, and retrieve a simulation through gcloud."""

    _job_id: str | None
    _input_hash: str | None
    _config_dir: Path | None
    _dry_run: bool = False

    if TYPE_CHECKING:

        def write(
            self, output_dir: str | Path, *, dry_run: bool = False
        ) -> SimulationArtifacts:
            """Write the simulation artifacts consumed by the cloud workflow."""
            ...

    def _prepare_upload_dir(self, *, dry_run: bool = False) -> Path:
        """Generate upload artifacts in a fresh temporary directory."""
        import tempfile

        directory = Path(tempfile.mkdtemp(prefix="fdtd_"))
        self.write(directory, dry_run=dry_run)
        self._config_dir = directory
        self._dry_run = dry_run
        return directory

    def upload(self, *, verbose: bool = True, dry_run: bool = False) -> str:
        """Generate and upload artifacts without starting the cloud job."""
        from gsim import gcloud
        from gsim.hashing import compute_input_hash

        directory = self._prepare_upload_dir(dry_run=dry_run)
        self._estimated_runtime_seconds = gcloud.estimate_runtime_seconds(directory)
        self._input_hash = compute_input_hash(directory, "fdtd")
        self._job_id = gcloud.upload(
            directory,
            "fdtd",
            verbose=verbose,
            input_hash=self._input_hash,
        )
        return self._job_id

    def start(self, *, verbose: bool = True) -> None:
        """Start the previously uploaded cloud job."""
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("Call upload() first")
        gcloud.start(self._job_id, verbose=verbose)

    def get_status(self) -> str:
        """Return the current cloud job status."""
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("No job submitted yet")
        return gcloud.get_status(self._job_id)

    def wait_for_results(
        self,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        parent_dir: str | Path | None = None,
        poll_interval: float = 5.0,
    ) -> Any:
        """Wait for, download, and parse the current cloud job."""
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("No job submitted yet")
        result = gcloud.wait_for_results(
            self._job_id,
            verbose=verbose,
            parent_dir=parent_dir,
            poll_interval=poll_interval,
            estimated_runtime_seconds=getattr(self, "_estimated_runtime_seconds", None),
        )
        if self._dry_run and (
            not getattr(result, "dry_run", False)
            or result.convergence.get("timesteps_run") != 0
            or (result.excitation_type == "eigenmode" and result.source_mode is None)
        ):
            raise RuntimeError(
                "The cloud FDTD backend did not return a dry run with zero "
                "timesteps. Update the FDTD cloud image to one supporting "
                "run.dry_run and source-mode export."
            )
        return result

    def run(
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: Literal["quiet", "status", "full"] = "status",
        wait: bool = True,
        check_cache: bool = False,
        poll_interval: float = 5.0,
        dry_run: bool = False,
    ) -> Any:
        """Run cloud FDTD, or build and solve source modes without time stepping.

        ``dry_run=True`` returns grid and source-mode information. A subsequent
        normal run uses the original solver settings and a separate cache entry.
        """
        from gsim import gcloud

        if check_cache:
            directory = self._prepare_upload_dir(dry_run=dry_run)
            self._estimated_runtime_seconds = gcloud.estimate_runtime_seconds(directory)
            self._input_hash, cached_job_id = gcloud.check_cache_for_dir(
                directory, "fdtd"
            )
            if cached_job_id is not None:
                self._job_id = cached_job_id
                if verbose != "quiet":
                    print(f"Cache hit: reusing job {cached_job_id}")  # noqa: T201
                if not wait:
                    return self._job_id
                return self.wait_for_results(
                    verbose=verbose,
                    parent_dir=parent_dir,
                    poll_interval=poll_interval,
                )
            self._job_id = gcloud.upload(
                directory,
                "fdtd",
                verbose=False,
                input_hash=self._input_hash,
            )
        else:
            self.upload(verbose=False, dry_run=dry_run)
        self.start(verbose=verbose != "quiet")
        if not wait:
            return self._job_id
        return self.wait_for_results(
            verbose=verbose,
            parent_dir=parent_dir,
            poll_interval=poll_interval,
        )


__all__ = ["CloudWorkflowMixin"]
