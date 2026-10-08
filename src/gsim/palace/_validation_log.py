"""Conservative parsing of Palace solver and adaptive sampling log evidence."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from gsim.palace._validation_models import AdaptiveCheck, ModeSample, SolutionCheck

_NUMBER = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf|nan)"
_EXCITATION = re.compile(r"(?:Sweeping|Adding) excitation index (\d+)")
_FREQUENCY = re.compile(rf"wave ports for \u03c9/2\u03c0 = ({_NUMBER}) GHz")
_MODE = re.compile(rf"Port (\d+), mode (\d+): k\u2099 = ({_NUMBER})\s*({_NUMBER})i")
_FAILURE = re.compile(r"did not converge|failed to converge|diverged", re.IGNORECASE)
_SUCCESS = re.compile(
    r"(?:solver|eigensolve|Quasi-Newton) converged(?: in | \()", re.IGNORECASE
)
_ZERO_PCG = re.compile(r"^PCG solver did NOT converge in 0 iterations$")
_UNIT_RESIDUAL = re.compile(
    rf"^Linear solver did not converge, norm\(Ax-b\)/norm\(b\) = ({_NUMBER})(?: |$)"
)


def clean_lines(log: str) -> list[str]:
    """Strip terminal color codes while preserving source line numbers."""
    return [re.sub(r"\x1b\[[0-9;]*m", "", line).strip() for line in log.splitlines()]


def solver_checks(
    lines: Sequence[str], *, estimator_disabled: bool
) -> tuple[SolutionCheck, SolutionCheck]:
    """Separate auxiliary PCG evidence from core and field-correction failures."""
    core_success, core_failure, estimator_success, estimator_failure, skipped = (
        [],
        [],
        [],
        [],
        [],
    )
    estimating = offline = correction = False
    skip_indices: set[int] = set()
    for index, line in enumerate(lines):
        if index in skip_indices:
            continue
        if "Beginning PROM construction offline phase" in line:
            offline = True
        if "online phase" in line.lower():
            offline = False
        if "Updating solution error estimates" in line:
            estimating, correction = True, False
        elif re.search(r"field.*correction|correcting.*field", line, re.IGNORECASE):
            estimating, correction = False, True
        elif re.search(
            r"\b(?:GMRES|FGMRES|eigensolve|eigenvalue)\b", line, re.IGNORECASE
        ):
            estimating = False
        elif (
            _EXCITATION.search(line)
            or "Calculating boundary modes" in line
            or line.startswith(("It ", "Greedy iteration"))
        ):
            estimating, correction = False, False
        evidence = f"line {index + 1}: {line}"
        # Adaptive offline estimation omits the ordinary estimator header.
        # Exempt only this exact paired warning, with explicit caller opt-in.
        if (
            estimator_disabled
            and (estimating or offline)
            and not correction
            and _ZERO_PCG.fullmatch(line)
        ):
            following = index + 1
            while following < len(lines) and lines[following] in ("", "--> Warning!"):
                following += 1
            match = (
                _UNIT_RESIDUAL.match(lines[following])
                if following < len(lines)
                else None
            )
            if match and float(match[1]) == 1.0:
                skipped.append(evidence)
                skip_indices.add(following)
                continue
        if _FAILURE.search(line):
            target = estimator_failure if estimating else core_failure
            target.append(evidence)
        elif _SUCCESS.search(line):
            target = estimator_success if estimating else core_success
            target.append(evidence)
    core = SolutionCheck(
        "core",
        "failed" if core_failure else "passed" if core_success else "unknown",
        tuple(core_failure or core_success),
    )
    if estimator_failure:
        estimator = SolutionCheck("estimator", "failed", tuple(estimator_failure))
    elif estimator_disabled:
        estimator = SolutionCheck(
            "estimator", "skipped", ("Caller declares EstimatorMaxIts=0.", *skipped)
        )
    else:
        estimator = SolutionCheck(
            "estimator",
            "passed" if estimator_success else "unknown",
            tuple(estimator_success),
        )
    return core, estimator


def adaptive_checks(
    lines: Sequence[str], expected_excitations: Sequence[int]
) -> tuple[AdaptiveCheck, ...]:
    """Preserve each adaptive outcome and require its full printed training grid."""
    results: list[AdaptiveCheck] = []
    excitation = None
    observed: set[int] = set()
    for index, line in enumerate(lines):
        if match := _EXCITATION.search(line):
            excitation = int(match[1])
            observed.add(excitation)
        converged = re.search(
            r"Adaptive sampling converged with (\d+) frequency samples", line
        )
        maximum = "Adaptive sampling reached maximum" in line
        if not (converged or maximum):
            continue
        count = int(converged[1]) if converged else None
        frequencies: list[float] = []
        collecting = complete = False
        for next_line in lines[index + 1 :]:
            if _EXCITATION.search(next_line) or "Adaptive sampling" in next_line:
                break
            if "Sample errors:" in next_line:
                complete = collecting
                break
            if "Sampled frequencies (GHz):" in next_line:
                collecting = True
                next_line = next_line.split(":", maxsplit=1)[1]
            if collecting:
                try:
                    frequencies.extend(
                        float(value) for value in next_line.split(",") if value.strip()
                    )
                except ValueError:
                    break
        status = "failed" if maximum else "passed"
        evidence = [f"line {index + 1}: {line}"]
        if not maximum and (not complete or len(frequencies) != count):
            status = "unknown"
            evidence.append("Training frequencies are missing or incomplete.")
        if any(not math.isfinite(value) or value <= 0 for value in frequencies):
            status = "failed"
            evidence.append("Training frequencies must be finite and positive.")
        results.append(
            AdaptiveCheck(
                excitation,
                SolutionCheck(f"adaptive[{excitation}]", status, tuple(evidence)),
                count,
                tuple(frequencies),
            )
        )
    # Only infer adaptive expectations when an adaptive outcome was observed.
    required = set(expected_excitations) | (observed if results else set())
    results.extend(
        AdaptiveCheck(
            missing,
            SolutionCheck(
                f"adaptive[{missing}]",
                "unknown",
                ("No adaptive outcome for this excitation.",),
            ),
        )
        for missing in sorted(required - {result.excitation for result in results})
    )
    return tuple(results)


def port_mode_samples(lines: Sequence[str]) -> tuple[ModeSample, ...]:
    """Keep all printed port solves, including frequency and excitation rewinds."""
    samples = []
    frequency = excitation = None
    for line in lines:
        if "online phase" in line.lower():
            excitation = None
        if match := _EXCITATION.search(line):
            excitation = int(match[1])
        if match := _FREQUENCY.search(line):
            frequency = float(match[1])
        if match := _MODE.search(line):
            samples.append(
                ModeSample(
                    port=int(match[1]),
                    mode=int(match[2]),
                    wave_number=complex(float(match[3]), float(match[4])),
                    frequency_ghz=frequency,
                    excitation=excitation,
                )
            )
    return tuple(samples)
