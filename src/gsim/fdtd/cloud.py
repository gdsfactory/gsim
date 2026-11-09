"""FDTD dry-run selection and validation on the shared cloud lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gsim.common.cloud import CloudJobMixin, Verbosity


class CloudWorkflowMixin(CloudJobMixin):
    """Keep preview and full jobs distinct while sharing upload/cache handling."""

    _dry_run: bool = False

    def upload(self, *, verbose: bool = True, dry_run: bool = False) -> str:
        """Upload inputs for a full simulation or source-mode preview."""
        self._dry_run = dry_run
        return super().upload(verbose=verbose)

    def run(
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: Verbosity = "status",
        wait: bool = True,
        check_cache: bool = False,
        poll_interval: float = 5.0,
        dry_run: bool = False,
    ) -> Any:
        """Run cloud FDTD, or inspect grid/source modes without time stepping."""
        self._dry_run = dry_run
        return super().run(
            parent_dir,
            verbose=verbose,
            wait=wait,
            check_cache=check_cache,
            poll_interval=poll_interval,
        )

    def wait_for_results(
        self,
        *,
        verbose: Verbosity = "status",
        parent_dir: str | Path | None = None,
        poll_interval: float = 5.0,
    ) -> Any:
        """Retrieve results and reject backends that ignored a dry-run request."""
        result = super().wait_for_results(
            verbose=verbose,
            parent_dir=parent_dir,
            poll_interval=poll_interval,
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


__all__ = ["CloudWorkflowMixin"]
