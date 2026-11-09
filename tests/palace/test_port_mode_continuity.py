"""Regressions for the wave-port mode continuity check (gsim#375, gsim#408).

The data files are trimmed excerpts of three real generic-PDK CPW driven logs
(two ports, 31 frequencies from 1 to 10 GHz, two excitations): an old Palace
(cfa430a) run with three spurious port solves, a clean run of the same Palace,
and a clean run of the fixed Palace (palace#1019) on the same input.
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from gsim.palace.validation import (
    CONTINUITY_RTOL,
    MAX_EVIDENCE,
    MIN_DISTINCT_FREQUENCIES,
    ModeExpectation,
    ModeSample,
    check_port_mode_continuity,
    validate_solution,
)

DATA = Path(__file__).parent / "data"
SPURIOUS = DATA / "port_modes_old_palace_spurious.txt"
CLEAN_LOGS = [
    "port_modes_old_palace_clean.txt",
    "port_modes_new_palace_clean.txt",
]
C0 = 299_792_458.0


def _read(name: str | Path) -> str:
    path = name if isinstance(name, Path) else DATA / name
    return path.read_text(encoding="utf-8")


def _k(frequency_ghz: float, n_eff: float, loss: float = 1e-4) -> complex:
    k0 = 2 * math.pi * frequency_ghz * 1e9 / C0
    return complex(n_eff * k0, -loss * n_eff * k0)


FREQUENCIES = tuple(range(1, 13))


def _samples(
    n_eff,
    *,
    frequencies=FREQUENCIES,
    ports=(1, 2),
    excitation: int | None = 1,
) -> list[ModeSample]:
    """Synthetic samples; ``n_eff(port, f)`` gives the effective index."""
    return [
        ModeSample(
            port=port,
            mode=1,
            wave_number=_k(f, n_eff(port, f)),
            frequency_ghz=float(f),
            excitation=excitation,
        )
        for f in frequencies
        for port in ports
    ]


def _status(checks, name):
    return {check.name: check for check in checks}[name]


def test_spurious_old_palace_log_warns_and_names_the_bad_points():
    report = validate_solution(_read(SPURIOUS))
    check = _status(report.port_mode_checks, "port_mode_continuity")
    assert check.status == "warning"
    assert check.evidence[0] == "3 of 124 port solves flagged."
    listed = "\n".join(check.evidence[1:])
    assert len(check.evidence) == 4  # the summary and exactly the three points
    # Excitation 2: both ports at 1 GHz, port 2 again at 1.3 GHz; port 1 at
    # 1.3 GHz was a good solve and must not be named.
    assert "port 1, mode 1, excitation 2: n_eff = 2.0087 at 1 GHz" in listed
    assert "port 2, mode 1, excitation 2: n_eff = 2.2669 at 1 GHz" in listed
    assert "port 2, mode 1, excitation 2: n_eff = 3.3796 at 1.3 GHz" in listed
    assert "neighbour median 1.71" in listed
    assert "excitation 1" not in listed
    # A warning is not a failure, and the other checks are untouched.
    assert report.failures == ()
    assert report.warnings == (check,)
    assert report.mode_checks == ()
    assert len(report.mode_samples) == 124


def test_spurious_log_port_comparison_is_opt_in():
    text = _read(SPURIOUS)
    default = validate_solution(text)
    assert [c.name for c in default.port_mode_checks] == ["port_mode_continuity"]
    report = validate_solution(text, ports_share_cross_section=True)
    assert [c.name for c in report.port_mode_checks] == [
        "port_mode_continuity",
        "port_mode_ports",
    ]
    ports = _status(report.port_mode_checks, "port_mode_ports")
    assert ports.status == "warning"
    assert ports.evidence[0] == "2 of 62 port pairs flagged."
    assert "ports 1 and 2 differ by 11.39 % at 1 GHz" in ports.evidence[1]
    assert "ports 1 and 2 differ by 49.36 % at 1.3 GHz" in ports.evidence[2]


@pytest.mark.parametrize("name", CLEAN_LOGS)
def test_clean_logs_pass(name):
    text = _read(name)
    for share in (False, True):
        report = validate_solution(text, ports_share_cross_section=share)
        assert {c.name for c in report.port_mode_checks} == (
            {"port_mode_continuity", "port_mode_ports"}
            if share
            else {"port_mode_continuity"}
        )
        assert all(c.status == "passed" for c in report.port_mode_checks)
        assert report.warnings == ()
        assert report.failures == ()
    continuity = _status(report.port_mode_checks, "port_mode_continuity")
    assert continuity.evidence[0].startswith("4 series, 124 solves")


@pytest.mark.parametrize("name", CLEAN_LOGS)
def test_clean_logs_keep_a_wide_margin_below_the_threshold(name):
    """Real scatter is at most 0.05 %; a limit ten times tighter still passes."""
    tight = check_port_mode_continuity(
        validate_solution(_read(name)).mode_samples, rtol=CONTINUITY_RTOL / 10
    )
    assert tight[0].status == "passed"


def test_fixtures_hold_only_the_lines_the_parser_needs():
    # Fixtures hold only what the parser needs: no solver or GMRES output.
    for name in (SPURIOUS.name, *CLEAN_LOGS):
        lines = _read(name).splitlines()
        assert len(lines) == 188
        assert all(
            line.startswith(("Sweeping", "Calculating boundary modes", " Port "))
            for line in lines
        )


def test_too_few_points_is_unknown_not_passed():
    few = _samples(lambda p, f: 1.7, frequencies=range(1, MIN_DISTINCT_FREQUENCIES))
    (check,) = check_port_mode_continuity(few)
    assert check.status == "unknown"
    assert "only 5 distinct frequencies" in check.evidence[0]
    # Even an obviously wrong point cannot be judged on a series that short.
    wrong = [*few[:-1], ModeSample(2, 1, _k(5.0, 3.0), 5.0, 1)]
    assert check_port_mode_continuity(wrong)[0].status == "unknown"
    enough = _samples(
        lambda p, f: 1.7, frequencies=range(1, MIN_DISTINCT_FREQUENCIES + 1)
    )
    assert check_port_mode_continuity(enough)[0].status == "passed"


def test_repeated_frequencies_do_not_count_as_distinct():
    repeated = _samples(lambda p, f: 1.7, frequencies=[1.0, 2.0, 3.0] * 4)
    assert check_port_mode_continuity(repeated)[0].status == "unknown"


def test_short_series_is_reported_next_to_a_judged_one():
    long = _samples(lambda p, f: 1.7, ports=(1,))
    short = [ModeSample(2, 1, _k(f, 1.7), float(f), 1) for f in (1, 2)]
    (check,) = check_port_mode_continuity([*long, *short])
    assert check.status == "passed"
    assert any("port 2" in line and "not judged" in line for line in check.evidence)


def test_single_port_gives_unknown_port_comparison():
    one = _samples(lambda p, f: 1.7, ports=(1,))
    continuity, ports = check_port_mode_continuity(one, compare_ports=True)
    assert continuity.status == "passed"
    assert ports.name == "port_mode_ports"
    assert ports.status == "unknown"
    assert "Only one port" in ports.evidence[0]


def test_ports_on_different_frequency_lists_are_unknown():
    a = _samples(lambda p, f: 1.7, ports=(1,), frequencies=range(1, 9))
    b = _samples(
        lambda p, f: 1.7, ports=(2,), frequencies=[x + 0.5 for x in range(1, 9)]
    )
    _, ports = check_port_mode_continuity([*a, *b], compare_ports=True)
    assert ports.status == "unknown"
    assert "share no solve frequency" in ports.evidence[0]


def test_ports_with_different_cross_sections_are_only_compared_on_request():
    samples = _samples(lambda p, f: 1.7 if p == 1 else 1.9)
    assert check_port_mode_continuity(samples)[0].status == "passed"
    _, ports = check_port_mode_continuity(samples, compare_ports=True)
    assert ports.status == "warning"
    assert ports.evidence[0] == "12 of 12 port pairs flagged."
    assert len(ports.evidence) == 1 + MAX_EVIDENCE + 1
    assert ports.evidence[-1] == f"... and {12 - MAX_EVIDENCE} more."


def test_eigenmode_and_unrelated_logs_are_ignored():
    eigenmode = (
        "Eigenmode log\n"
        "Calculating boundary modes at wave ports\n"
        " Port 1, mode 1: k\u2099 = 3.587e+01-3.389e-03i m\u207b\u00b9\n"
        " Port 2, mode 1: k\u2099 = 3.587e+01-3.389e-03i m\u207b\u00b9\n"
        "Quasi-Newton converged in 7 iterations\n"
    )
    report = validate_solution(eigenmode)
    assert len(report.mode_samples) == 2
    assert report.port_mode_checks == ()
    assert report.warnings == ()
    assert validate_solution("").port_mode_checks == ()
    assert validate_solution("GMRES solver converged in 11 iterations").warnings == ()


def test_existing_mode_expectations_are_unaffected_by_the_new_check():
    report = validate_solution(
        _read(SPURIOUS),
        mode_expectations=[ModeExpectation(port=1, mode=1, frequency_ghz=2.5)],
    )
    assert [check.status for check in report.mode_checks] == ["passed"]
    assert report.port_mode_checks[0].status == "warning"


def test_nonfinite_wave_number_is_flagged_even_on_a_short_series():
    samples = _samples(lambda p, f: 1.7, ports=(1,), frequencies=range(1, 4))
    samples.append(ModeSample(1, 1, complex(math.nan, 0), 4.0, 1))
    (check,) = check_port_mode_continuity(samples)
    assert check.status == "warning"
    assert "nonfinite wave number at 4 GHz" in check.evidence[1]


def test_a_nonfinite_neighbour_does_not_hide_a_real_jump():
    samples = _samples(
        lambda p, f: 2.4 if f == 6 else 1.7, ports=(1,), frequencies=range(1, 13)
    )
    samples.append(ModeSample(1, 1, complex(math.inf, 0), 6.5, 1))
    (check,) = check_port_mode_continuity(samples)
    assert check.status == "warning"
    assert any("n_eff = 2.4000 at 6 GHz" in line for line in check.evidence)


def test_result_does_not_depend_on_sample_order():
    samples = _samples(lambda p, f: 2.5 if (p, f) == (2, 5) else 1.7)
    shuffled = samples.copy()
    random.Random(7).shuffle(shuffled)  # noqa: S311 - test order shuffle, not crypto
    assert check_port_mode_continuity(shuffled, compare_ports=True) == (
        check_port_mode_continuity(samples, compare_ports=True)
    )


def test_series_are_separate_per_excitation_and_mode():
    # n_eff differs between excitations (e.g. a different mode index), but each
    # series is smooth, so nothing is flagged.
    a = _samples(lambda p, f: 1.7, excitation=1)
    b = _samples(lambda p, f: 2.4, excitation=2)
    c = _samples(lambda p, f: 3.1, excitation=None)
    (check,) = check_port_mode_continuity([*a, *b, *c])
    assert check.status == "passed"
    assert check.evidence[0].startswith("6 series, 72 solves")


def test_slow_dispersion_is_not_flagged_but_a_step_is():
    ramp = lambda p, f: 1.5 + 0.003 * f  # noqa: E731 - 0.2 % per step
    assert check_port_mode_continuity(_samples(ramp))[0].status == "passed"
    step = lambda p, f: ramp(p, f) + (0.2 if (p, f) == (1, 7) else 0.0)  # noqa: E731
    (check,) = check_port_mode_continuity(_samples(step))
    assert check.status == "warning"
    assert check.evidence[0] == "1 of 24 port solves flagged."


def test_two_spurious_neighbours_are_tolerated_by_the_median():
    samples = _samples(
        lambda p, f: 2.4 if f in (6, 7) else 1.7, ports=(1,), frequencies=range(1, 13)
    )
    (check,) = check_port_mode_continuity(samples)
    assert check.status == "warning"
    assert check.evidence[0] == "2 of 12 port solves flagged."


def test_thresholds_are_adjustable_and_validated():
    samples = _samples(lambda p, f: 1.7 * (1.03 if f == 5 else 1.0), ports=(1,))
    assert check_port_mode_continuity(samples)[0].status == "warning"
    assert check_port_mode_continuity(samples, rtol=0.05)[0].status == "passed"
    for bad in (-0.1, math.nan, math.inf):
        with pytest.raises(ValueError, match="nonnegative"):
            check_port_mode_continuity(samples, rtol=bad)
        with pytest.raises(ValueError, match="nonnegative"):
            check_port_mode_continuity(samples, port_rtol=bad)


def test_explicit_samples_replace_the_log_and_are_checked_too():
    samples = _samples(lambda p, f: 2.5 if (p, f) == (1, 5) else 1.7)
    report = validate_solution(_read(SPURIOUS), mode_samples=samples)
    assert report.port_mode_checks[0].status == "warning"
    assert report.port_mode_checks[0].evidence[0] == "1 of 24 port solves flagged."
    # Explicit samples without a frequency (BoundaryMode tables) are not driven
    # solves: nothing to check.
    bare = [ModeSample(port=1, mode=2, wave_number=50 - 1j)]
    assert validate_solution("", mode_samples=bare).port_mode_checks == ()
    assert validate_solution(_read(SPURIOUS), mode_samples=[]).port_mode_checks == ()
