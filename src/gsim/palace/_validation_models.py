"""Evidence records for Palace result validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CheckStatus = Literal["passed", "failed", "unknown", "skipped"]


@dataclass(frozen=True)
class SolutionCheck:
    """One evidence-based check; unknown never implies success."""

    name: str
    status: CheckStatus
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdaptiveCheck:
    """Adaptive training outcome for an excitation, with frequencies in GHz."""

    excitation: int | None
    check: SolutionCheck
    sample_count: int | None = None
    frequencies_ghz: tuple[float, ...] = ()


@dataclass(frozen=True)
class ModeSample:
    """One selected port mode; wave number is complex and measured in 1/m.

    ``voltage_ratio`` is an optional caller-measured complex ratio between two
    consistently oriented voltage integrals. Its interpretation is geometry
    dependent; no CPW or slotline parity is assumed.
    """

    port: int
    mode: int
    wave_number: complex
    frequency_ghz: float | None = None
    excitation: int | None = None
    voltage_ratio: complex | None = None


@dataclass(frozen=True)
class ModeExpectation:
    """Caller-selected mode identity and optional physical reference values.

    Every matching sample, including repeated solves, must satisfy the checks.
    ``frequency_rtol`` accounts for rounded log frequencies, not solver error.
    Supply physical tolerances explicitly when using reference values.
    """

    port: int
    mode: int
    frequency_ghz: float | None = None
    frequency_rtol: float = 5e-4
    wave_number: complex | None = None
    wave_number_rtol: float = 0.0
    wave_number_atol: float = 0.0
    voltage_ratio: complex | None = None
    voltage_ratio_atol: float = 0.0


@dataclass(frozen=True)
class SolutionProvenance:
    """Available solver revision and hashes of the supplied current input files.

    Hashes alone do not establish which inputs a remote solver actually used.
    ``input_check`` compares them with an optional independently saved manifest.
    """

    solver_revision: str | None
    input_sha256: dict[str, str] = field(default_factory=dict)
    input_check: SolutionCheck = SolutionCheck("inputs", "unknown")


@dataclass(frozen=True)
class SolutionReport:
    """Independent checks, without claiming blanket physical validity.

    A passed ``core`` means recognized convergence evidence with no recognized
    core failure. It does not establish mesh accuracy, complete log coverage,
    port identity, or convergence of unreported eigenpairs.
    """

    core: SolutionCheck
    estimator: SolutionCheck
    exit_status: SolutionCheck
    oom: SolutionCheck
    adaptive: tuple[AdaptiveCheck, ...]
    mode_checks: tuple[SolutionCheck, ...]
    mode_samples: tuple[ModeSample, ...]
    provenance: SolutionProvenance

    @property
    def failures(self) -> tuple[SolutionCheck, ...]:
        """Return explicit failures; an empty tuple can still contain unknowns."""
        checks = (
            self.core,
            self.estimator,
            self.exit_status,
            self.oom,
            self.provenance.input_check,
            *(item.check for item in self.adaptive),
            *self.mode_checks,
        )
        return tuple(check for check in checks if check.status == "failed")
