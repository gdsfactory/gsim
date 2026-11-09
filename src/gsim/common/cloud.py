"""One cloud-job lifecycle for every solver that runs on GDSFactory+ cloud.

A simulation names its job type and writes its inputs to a fresh staging
directory. Everything after that — runtime estimate, input hash, cache lookup,
upload, start, status, waiting, and removing the staging copy when an upload
fails — happens here, the same way for Palace, MEEP, and FDTD.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple

Verbosity = Literal["quiet", "status", "full"]


class Submission(NamedTuple):
    """A directory of solver inputs handed to the cloud as one job."""

    job_id: str
    input_hash: str
    estimated_runtime_seconds: float
    cached: bool
    """``True`` when a finished job with identical inputs was reused."""


def upload_inputs(
    directory: Path,
    job_type: str,
    *,
    check_cache: bool = False,
    verbose: bool = False,
) -> Submission:
    """Upload a staging directory of solver inputs, without starting the job.

    The input hash always travels with the upload, so a later identical run can
    be served from the cache. With ``check_cache``, a finished job with
    byte-identical inputs is reused instead and nothing is uploaded; a lookup
    failure degrades to a normal upload.

    Args:
        directory: Staging copy of the inputs. Removed if the upload fails, so
            it must never be the user's own directory.
        job_type: Solver name, e.g. ``"palace"``, ``"meep"``, or ``"fdtd"``.
        check_cache: Reuse a finished job with identical inputs when one exists.
        verbose: Print upload progress.

    Returns:
        The :class:`Submission`; ``cached`` says whether it reused a job.
    """
    from gsim import gcloud
    from gsim.hashing import compute_input_hash

    try:
        estimate = gcloud.estimate_runtime_seconds(directory)
        if check_cache:
            input_hash, cached_job_id = gcloud.check_cache_for_dir(directory, job_type)
            if cached_job_id is not None:
                return Submission(cached_job_id, input_hash, estimate, cached=True)
        else:
            input_hash = compute_input_hash(directory, job_type)
        job_id = gcloud.upload(
            directory, job_type, verbose=verbose, input_hash=input_hash
        )
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return Submission(job_id, input_hash, estimate, cached=False)


def submit(
    directory: Path,
    job_type: str,
    *,
    check_cache: bool = False,
    verbose: Verbosity = "status",
) -> Submission:
    """Upload and start a staging directory of solver inputs.

    Like :func:`upload_inputs`, but starts the uploaded job. A reused cached
    job is already finished, so it is announced instead of started.
    """
    submission = upload_inputs(directory, job_type, check_cache=check_cache)
    _start_or_announce(submission, verbose=verbose)
    return submission


def _start_or_announce(submission: Submission, *, verbose: Verbosity) -> None:
    """Start an uploaded job, or announce the finished one a cache hit reused."""
    from gsim import gcloud

    if submission.cached:
        if verbose != "quiet":
            print(f"Cache hit: reusing job {submission.job_id}")  # noqa: T201
    else:
        gcloud.start(submission.job_id, verbose=verbose != "quiet")


class CloudJobMixin:
    """Upload, start, monitor, and fetch a simulation's cloud job.

    A simulation names its ``_cloud_job_type`` and writes its inputs in
    ``_prepare_upload_dir``; this mixin owns the job and its state.
    """

    _cloud_job_type: ClassVar[str]

    _job_id: str | None = None
    _input_hash: str | None = None
    _config_dir: Path | None = None
    _estimated_runtime_seconds: float | None = None

    def _prepare_upload_dir(self) -> Path:
        """Write this simulation's cloud inputs to a fresh staging directory."""
        raise NotImplementedError

    def _record(self, directory: Path, submission: Submission) -> None:
        """Remember the staging directory and the job it became."""
        self._config_dir = directory
        self._job_id = submission.job_id
        self._input_hash = submission.input_hash
        self._estimated_runtime_seconds = submission.estimated_runtime_seconds

    def upload(self, *, verbose: bool = True) -> str:
        """Write the inputs and upload them. Does NOT start execution.

        Args:
            verbose: Print progress messages.

        Returns:
            ``job_id`` string for use with :meth:`start`, :meth:`get_status`,
            or :func:`gsim.wait_for_results`.
        """
        directory = self._prepare_upload_dir()
        submission = upload_inputs(directory, self._cloud_job_type, verbose=verbose)
        self._record(directory, submission)
        return submission.job_id

    def start(self, *, verbose: bool = True) -> None:
        """Start cloud execution for this simulation's uploaded job.

        Raises:
            ValueError: If :meth:`upload` has not been called.
        """
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("Call upload() first")
        gcloud.start(self._job_id, verbose=verbose)

    def get_status(self) -> str:
        """Get the current status of this simulation's cloud job.

        Returns:
            Status string (``"created"``, ``"queued"``, ``"running"``,
            ``"completed"``, ``"failed"``).

        Raises:
            ValueError: If no job has been submitted yet.
        """
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("No job submitted yet")
        return gcloud.get_status(self._job_id)

    def wait_for_results(
        self,
        *,
        verbose: Verbosity = "status",
        parent_dir: str | Path | None = None,
        poll_interval: float = 5.0,
    ) -> Any:
        """Wait for this simulation's cloud job, then download and parse results.

        Args:
            verbose: ``"quiet"`` no output, ``"status"`` status line,
                ``"full"`` stream solver logs.
            parent_dir: Where to create the sim-data directory.
            poll_interval: Seconds between status polls.

        Returns:
            The solver's parsed results.

        Raises:
            ValueError: If no job has been submitted yet.
        """
        from gsim import gcloud

        if self._job_id is None:
            raise ValueError("No job submitted yet")
        return gcloud.wait_for_results(
            self._job_id,
            verbose=verbose,
            parent_dir=parent_dir,
            poll_interval=poll_interval,
            estimated_runtime_seconds=self._estimated_runtime_seconds,
        )

    def run(
        self,
        parent_dir: str | Path | None = None,
        *,
        verbose: Verbosity = "status",
        wait: bool = True,
        check_cache: bool = False,
        poll_interval: float = 5.0,
    ) -> Any:
        """Run the simulation on GDSFactory+ cloud.

        Args:
            parent_dir: Where to create the sim directory.
                Defaults to the current working directory.
            verbose: ``"quiet"`` no output, ``"status"`` status line,
                ``"full"`` stream solver logs.
            wait: If ``True`` (default), block until results are ready.
                If ``False``, upload + start and return the ``job_id``.
            check_cache: If ``True``, look for a completed cloud job with
                byte-identical inputs and reuse its results instead of
                submitting. A lookup failure degrades to a normal submit.
            poll_interval: Seconds between status polls while waiting.

        Returns:
            The solver's parsed results when ``wait=True``, or the ``job_id``
            string when ``wait=False``.
        """
        directory = self._prepare_upload_dir()
        submission = upload_inputs(
            directory, self._cloud_job_type, check_cache=check_cache
        )
        # Recorded before the start, so a start that fails leaves the uploaded
        # job reachable through start() and get_status() for a retry.
        self._record(directory, submission)
        _start_or_announce(submission, verbose=verbose)
        if not wait:
            return submission.job_id
        return self.wait_for_results(
            verbose=verbose, parent_dir=parent_dir, poll_interval=poll_interval
        )


__all__ = ["CloudJobMixin", "Submission", "submit", "upload_inputs"]
