"""Regressions for the review of the port-mode continuity check.

1. A neighbour median of zero must not raise ``ZeroDivisionError``.
2. Port agreement must compare every port pair that has common frequencies,
   not only the pairs that include the lowest-numbered port.
"""

from __future__ import annotations

import math

import pytest

from gsim.palace.validation import (
    CONTINUITY_RTOL,
    MAX_EVIDENCE,
    ModeSample,
    check_port_mode_continuity,
    validate_solution,
)

C0 = 299_792_458.0


def _k(frequency_ghz: float, n_eff: float) -> complex:
    return complex(n_eff * 2 * math.pi * frequency_ghz * 1e9 / C0, 0.0)


def _series(n_effs, *, port=1, first_ghz=1, excitation=1) -> list[ModeSample]:
    return [
        ModeSample(port, 1, _k(f, n), float(f), excitation)
        for f, n in enumerate(n_effs, start=first_ghz)
    ]


def _by_name(checks, name):
    return {check.name: check for check in checks}[name]


# -- zero neighbour median ---------------------------------------------------


def test_five_zero_neighbours_and_a_nonzero_sample_is_flagged_not_raised():
    (check,) = check_port_mode_continuity(_series([0.0] * 5 + [1.7]))
    assert check.status == "warning"
    assert check.evidence[0] == "1 of 6 port solves flagged."
    assert len(check.evidence) == 2
    assert "n_eff = 1.7000 at 6 GHz" in check.evidence[1]
    assert "neighbour median 0 " in check.evidence[1]
    assert "absolute difference 1.7 > 0.01" in check.evidence[1]


@pytest.mark.parametrize("position", range(6))
def test_one_nonzero_sample_at_any_position_never_raises(position):
    n_effs = [0.0] * 6
    n_effs[position] = 1.7
    (check,) = check_port_mode_continuity(_series(n_effs))
    assert check.status == "warning"
    assert f"n_eff = 1.7000 at {position + 1} GHz" in "\n".join(check.evidence)


def test_zero_median_uses_the_tolerance_as_an_absolute_limit():
    small = CONTINUITY_RTOL / 2
    large = CONTINUITY_RTOL * 2
    assert check_port_mode_continuity(_series([0.0] * 5 + [small]))[0].status == (
        "passed"
    )
    assert check_port_mode_continuity(_series([0.0] * 5 + [large]))[0].status == (
        "warning"
    )
    # The same absolute rule follows a caller-supplied tolerance.
    assert (
        check_port_mode_continuity(_series([0.0] * 5 + [large]), rtol=0.1)[0].status
        == "passed"
    )


def test_all_zero_series_is_not_flagged():
    (check,) = check_port_mode_continuity(_series([0.0] * 8))
    assert check.status == "passed"


def test_zero_median_with_negative_neff_is_flagged_by_magnitude():
    (check,) = check_port_mode_continuity(_series([0.0] * 5 + [-1.7]))
    assert check.status == "warning"
    assert "absolute difference 1.7" in check.evidence[1]


def test_zero_median_does_not_hide_a_second_series():
    good = _series([1.7] * 8, port=2)
    bad = _series([0.0] * 5 + [1.7], port=1)
    (check,) = check_port_mode_continuity([*bad, *good])
    assert check.status == "warning"
    assert check.evidence[0] == "1 of 14 port solves flagged."
    assert "port 1" in check.evidence[1]


def test_zero_median_reaches_the_report_through_validate_solution():
    omega = "\u03c9/2\u03c0"
    k_n = "k\u2099"
    inverse_metre = "m\u207b\u00b9"
    lines = ["Sweeping excitation index 1 (1/1):"]
    for index in range(6):
        value = "0.000e+00+0.000e+00i" if index < 5 else "6.000e+01-1.000e-03i"
        lines.append(
            f"Calculating boundary modes at wave ports for {omega} = "
            f"{index + 1:.3e} GHz (4.820e-03)"
        )
        lines.append(f" Port 1, mode 1: {k_n} = {value} {inverse_metre}")
        lines.append("")
    report = validate_solution("\n".join(lines))
    (check,) = report.port_mode_checks
    assert check.name == "port_mode_continuity"
    assert check.status == "warning"
    assert report.warnings == (check,)
    assert report.failures == ()


# -- all port pairs ----------------------------------------------------------


def _three_ports(n_port2, n_port3):
    """Port 1 at 1-6 GHz; ports 2 and 3 at 7-12 GHz only."""
    return [
        *_series([1.7] * 6, port=1, first_ghz=1),
        *_series([n_port2] * 6, port=2, first_ghz=7),
        *_series([n_port3] * 6, port=3, first_ghz=7),
    ]


def test_disagreeing_ports_without_the_lowest_port_are_flagged():
    _, ports = check_port_mode_continuity(_three_ports(1.7, 1.9), compare_ports=True)
    assert ports.name == "port_mode_ports"
    assert ports.status == "warning"
    assert ports.evidence[0] == "6 of 6 port pairs flagged."
    listed = "\n".join(ports.evidence)
    assert "ports 2 and 3 differ by 10.53 % at 7 GHz" in listed
    assert "ports 1" not in listed


def test_agreeing_ports_without_the_lowest_port_pass_and_note_the_gap():
    _, ports = check_port_mode_continuity(_three_ports(1.7, 1.7), compare_ports=True)
    assert ports.status == "passed"
    assert ports.evidence[0] == "6 port pairs agree within 0.5 %."
    notes = "\n".join(ports.evidence[1:])
    assert "ports 1 and 2 share no solve frequency" in notes
    assert "ports 1 and 3 share no solve frequency" in notes


def test_pairs_not_involving_the_first_port_are_compared_when_all_overlap():
    samples = [
        *_series([1.7] * 6, port=1),
        *_series([1.7] * 6, port=2),
        *_series([1.9] * 6, port=3),
    ]
    _, ports = check_port_mode_continuity(samples, compare_ports=True)
    assert ports.status == "warning"
    # Pairs (1, 3) and (2, 3) disagree, pair (1, 2) agrees: 12 of 18 flagged.
    assert ports.evidence[0] == "12 of 18 port pairs flagged."
    assert "worst pair: mode 1, excitation 1:" in ports.evidence[1]


def test_single_outlier_flags_every_pair_that_contains_it():
    samples = [
        *_series([1.7] * 6, port=1),
        *_series([1.7] * 6, port=2),
        *_series([1.7] * 6, port=3),
    ]
    samples[-1] = ModeSample(3, 1, _k(6.0, 2.5), 6.0, 1)
    _, ports = check_port_mode_continuity(samples, compare_ports=True)
    assert ports.status == "warning"
    assert ports.evidence[0] == "2 of 18 port pairs flagged."


def test_worst_pair_is_named_even_when_the_listed_points_omit_it():
    ports_n_eff = {1: 1.70, 2: 1.71, 3: 1.80, 4: 2.40}
    samples = [
        s
        for port, n_eff in ports_n_eff.items()
        for s in _series([n_eff] * 8, port=port)
    ]
    _, ports = check_port_mode_continuity(samples, compare_ports=True)
    assert ports.status == "warning"
    assert ports.evidence[1].startswith(
        "worst pair: mode 1, excitation 1: ports 1 and 4 differ by 29.17 %"
    )
    # Summary, worst pair, MAX_EVIDENCE points, and the remainder line.
    assert len(ports.evidence) == 1 + 1 + MAX_EVIDENCE + 1
    # The first listed points all belong to pair (1, 2), not to pair (1, 4).
    listed = "\n".join(ports.evidence[2:])
    assert "ports 1 and 4" not in listed


def test_pair_comparison_is_per_mode_and_excitation():
    a = _series([1.7] * 6, port=2, excitation=1)
    b = _series([1.9] * 6, port=3, excitation=2)
    # Same frequencies, but different excitations are never compared.
    _, ports = check_port_mode_continuity([*a, *b], compare_ports=True)
    assert ports.status == "unknown"


def test_two_port_result_is_unchanged_by_pair_enumeration():
    samples = [*_series([1.7] * 6, port=1), *_series([1.9] * 6, port=2)]
    _, ports = check_port_mode_continuity(samples, compare_ports=True)
    assert ports.status == "warning"
    assert ports.evidence[0] == "6 of 6 port pairs flagged."
    assert "worst pair" not in "\n".join(ports.evidence)
    assert ports.evidence[1].startswith(
        "mode 1, excitation 1: ports 1 and 2 differ by 10.53 % at 1 GHz"
    )
