"""Validate available Palace result evidence without rerunning simulations.

The report keeps solver convergence, mesh-error estimation, adaptive sampling,
process exit/OOM evidence, input provenance, and caller-selected mode checks
independent. Missing observations are ``unknown``. A passed check only describes
its supplied evidence; no aggregate physical-validity flag is provided.
"""

from __future__ import annotations

import cmath
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from gsim.palace._validation_log import (
    adaptive_checks,
    clean_lines,
    port_mode_samples,
    solver_checks,
)
from gsim.palace._validation_models import (
    AdaptiveCheck,
    ModeExpectation,
    ModeSample,
    SolutionCheck,
    SolutionProvenance,
    SolutionReport,
)

__all__ = [
    "AdaptiveCheck",
    "ModeExpectation",
    "ModeSample",
    "SolutionCheck",
    "SolutionProvenance",
    "SolutionReport",
    "validate_solution",
]


def _validate_expectation(expectation: ModeExpectation) -> None:
    """Reject nonfinite references or invalid comparison tolerances."""
    for tolerance in (
        expectation.frequency_rtol,
        expectation.wave_number_rtol,
        expectation.wave_number_atol,
        expectation.voltage_ratio_atol,
    ):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(
                "Mode comparison tolerances must be finite and nonnegative."
            )
    for reference in (expectation.wave_number, expectation.voltage_ratio):
        if reference is not None and not cmath.isfinite(reference):
            raise ValueError("Mode reference values must be finite.")
    if expectation.frequency_ghz is not None and (
        not math.isfinite(expectation.frequency_ghz) or expectation.frequency_ghz <= 0
    ):
        raise ValueError("Expected frequency must be finite and positive.")


def _check_mode(
    expectation: ModeExpectation, samples: Sequence[ModeSample]
) -> SolutionCheck:
    """Compare every selected sample with a caller's mode and physical references."""
    _validate_expectation(expectation)
    matching = [
        sample
        for sample in samples
        if sample.port == expectation.port
        and (
            expectation.frequency_ghz is None
            or (
                sample.frequency_ghz is not None
                and math.isclose(
                    sample.frequency_ghz,
                    expectation.frequency_ghz,
                    rel_tol=expectation.frequency_rtol,
                )
            )
        )
    ]
    name = f"mode[port={expectation.port},frequency={expectation.frequency_ghz}]"
    if not matching:
        return SolutionCheck(name, "unknown", ("No matching port-mode samples.",))
    failures, missing = [], []
    for sample in matching:
        label = f"port {sample.port}, mode {sample.mode}, {sample.frequency_ghz} GHz"
        if sample.mode != expectation.mode:
            failures.append(f"{label}: expected mode {expectation.mode}.")
        if not cmath.isfinite(sample.wave_number):
            failures.append(f"{label}: nonfinite wave number.")
        elif expectation.wave_number is not None and not cmath.isclose(
            sample.wave_number,
            expectation.wave_number,
            rel_tol=expectation.wave_number_rtol,
            abs_tol=expectation.wave_number_atol,
        ):
            failures.append(
                f"{label}: wave number {sample.wave_number} misses reference."
            )
        if expectation.voltage_ratio is not None:
            if sample.voltage_ratio is None:
                missing.append(f"{label}: voltage parity ratio unavailable.")
            elif not cmath.isfinite(sample.voltage_ratio) or not cmath.isclose(
                sample.voltage_ratio,
                expectation.voltage_ratio,
                rel_tol=0,
                abs_tol=expectation.voltage_ratio_atol,
            ):
                failures.append(f"{label}: voltage parity ratio misses reference.")
    return SolutionCheck(
        name,
        "failed" if failures else "unknown" if missing else "passed",
        tuple(failures + missing) or (f"All {len(matching)} matching samples passed.",),
    )


def _provenance(
    metadata: Mapping[str, object] | None,
    input_files: Mapping[str, str | Path] | None,
    expected_input_sha256: Mapping[str, str] | None,
) -> SolutionProvenance:
    """Hash supplied inputs and compare only against an explicitly supplied manifest."""
    revision = (metadata or {}).get("GitTag")
    hashes, missing, failures = {}, [], []
    for name, path in (input_files or {}).items():
        try:
            with Path(path).open("rb") as stream:
                hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        except OSError as error:
            missing.append(
                f"Input {name!r} is unavailable: {type(error).__name__}: {error}"
            )
    for name, expected in (expected_input_sha256 or {}).items():
        if name not in hashes:
            missing.append(f"No current digest for recorded input {name!r}.")
        elif hashes[name] != expected.lower():
            failures.append(f"Input {name!r} differs from the recorded SHA-256 digest.")
    check = SolutionCheck(
        "inputs",
        "failed"
        if failures
        else "unknown"
        if missing or not expected_input_sha256
        else "passed",
        tuple(failures + missing)
        or (
            "Compared supplied files with the recorded manifest."
            if expected_input_sha256
            else "No recorded input manifest supplied.",
        ),
    )
    return SolutionProvenance(
        revision if isinstance(revision, str) else None, hashes, check
    )


def validate_solution(
    log: str,
    *,
    estimator_disabled: bool = False,
    expected_excitations: Sequence[int] = (),
    mode_expectations: Sequence[ModeExpectation] = (),
    mode_samples: Sequence[ModeSample] | None = None,
    exit_code: int | None = None,
    oom_killed: bool | None = None,
    metadata: Mapping[str, object] | None = None,
    input_files: Mapping[str, str | Path] | None = None,
    expected_input_sha256: Mapping[str, str] | None = None,
) -> SolutionReport:
    """Build a report from a Palace log and optional independent evidence.

    Args:
        log: Full solver log text. Unrecognized or absent evidence is unknown.
        estimator_disabled: Set only when the actual config has
            ``Solver.Linear.EstimatorMaxIts=0``. An exact zero-iteration PCG
            and unit-residual warning pair is skipped only in an estimator
            section or the adaptive offline phase, never a field correction.
        expected_excitations: Excitation IDs required to have adaptive outcomes.
            A missing or truncated outcome is unknown, not converged.
        mode_expectations: Optional mode IDs, complex wave numbers, and parity
            references to check. No application-specific limits are assumed.
        mode_samples: Explicit observations, for example from BoundaryMode
            tables and voltage integrals. When supplied, these replace the
            driven log's mode observations, including an explicitly empty list.
        exit_code: Recorded process exit status; zero does not pass other checks.
        oom_killed: Recorded OOM evidence, independent of exit status.
        metadata: Decoded ``palace.json``; its ``GitTag`` is recorded verbatim.
        input_files: Logical input names mapped to current local files to hash.
            Missing, unreadable and directory entries are unknown with a reason.
        expected_input_sha256: Optional independently recorded input manifest.

    Returns:
        Independent checks and raw mode samples. Even an empty ``failures``
        tuple does not establish that unknown or omitted checks passed.
    """
    lines = clean_lines(log)
    core, estimator = solver_checks(lines, estimator_disabled=estimator_disabled)
    samples = (
        tuple(mode_samples) if mode_samples is not None else port_mode_samples(lines)
    )
    return SolutionReport(
        core=core,
        estimator=estimator,
        exit_status=SolutionCheck(
            "exit_status",
            "unknown"
            if exit_code is None
            else "passed"
            if exit_code == 0
            else "failed",
            () if exit_code is None else (f"Recorded exit code: {exit_code}.",),
        ),
        oom=SolutionCheck(
            "oom",
            "unknown" if oom_killed is None else "failed" if oom_killed else "passed",
            () if oom_killed is None else (f"Recorded OOM killed: {oom_killed}.",),
        ),
        adaptive=adaptive_checks(lines, expected_excitations),
        mode_checks=tuple(
            _check_mode(expected, samples) for expected in mode_expectations
        ),
        mode_samples=samples,
        provenance=_provenance(metadata, input_files, expected_input_sha256),
    )
