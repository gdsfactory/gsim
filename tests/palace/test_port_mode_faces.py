"""Port-mode checks for faces that carry several wave-port entries.

A multimode wave port puts one Palace ``WavePort`` entry per mode on the same
face, with the active entry (lowest index) on the port's own index and the
other modes after all ports. The driven log prints one ``Port <index>, mode
<mode>`` line per entry. The checks must not mistake entries that follow
different modes for ports that disagree, and ``port_mode_faces`` must catch
modes of one face that coincide or come out in the wrong order.
"""

from __future__ import annotations

import math

import pytest

from gsim.palace._validation_log import port_mode_samples
from gsim.palace.ports.modes import PortModeEntry, faces_by_index
from gsim.palace.validation import (
    FACE_MODE_RTOL,
    MAX_EVIDENCE,
    ModeSample,
    check_port_mode_continuity,
    validate_solution,
)

C0 = 299_792_458.0
FREQUENCIES = tuple(range(1, 13))
#: Palace index -> face: two faces, modes 1 on the ports' own indices.
FACES = {1: "o1", 2: "o2", 3: "o1", 4: "o2", 5: "o1", 6: "o2"}
#: Palace index -> mode, as the sidecar of two ports with modes (1, 2, 3) has it.
MODES = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}


def _k(frequency_ghz: float, n_eff: float) -> complex:
    k0 = 2 * math.pi * frequency_ghz * 1e9 / C0
    return complex(n_eff * k0, -1e-4 * n_eff * k0)


def _samples(
    n_eff_of_mode,
    *,
    indices=(1, 2, 3, 4),
    excitations=(1,),
    frequencies=FREQUENCIES,
) -> list[ModeSample]:
    """Samples of the entries ``indices``; ``n_eff_of_mode(mode, face)`` gives n_eff."""
    return [
        ModeSample(
            port=index,
            mode=MODES[index],
            wave_number=_k(f, n_eff_of_mode(MODES[index], FACES[index])),
            frequency_ghz=float(f),
            excitation=excitation,
        )
        for excitation in excitations
        for f in frequencies
        for index in indices
    ]


def _by_name(checks):
    return {check.name: check for check in checks}


def distinct(mode: int, face: str) -> float:  # noqa: ARG001
    """A quasi-TEM-like pair: mode 1 at 2.3, mode 2 at 2.0, mode 3 at 1.7."""
    return {1: 2.3, 2: 2.0, 3: 1.7}[mode]


def test_distinct_modes_on_shared_faces_raise_no_flag_anywhere():
    """Entries of one face differ by 15 %, which no check may read as a disagreement."""
    samples = _samples(distinct)
    checks = _by_name(
        check_port_mode_continuity(samples, compare_ports=True, faces=FACES)
    )
    assert set(checks) == {"port_mode_continuity", "port_mode_ports", "port_mode_faces"}
    assert {name: c.status for name, c in checks.items()} == {
        "port_mode_continuity": "passed",
        "port_mode_ports": "passed",
        "port_mode_faces": "passed",
    }
    # Faces o1 and o2 each have the pair (1, 2): 12 frequencies x 2 faces.
    assert checks["port_mode_faces"].evidence[0].startswith("24 mode pairs")


def test_the_port_comparison_still_sees_a_mode_that_differs_between_faces():
    """Cross-face disagreement of the same mode is still ``port_mode_ports``' job."""

    def skewed(mode: int, face: str) -> float:
        return distinct(mode, face) * (1.05 if (face == "o2" and mode == 2) else 1.0)

    checks = _by_name(
        check_port_mode_continuity(_samples(skewed), compare_ports=True, faces=FACES)
    )
    assert checks["port_mode_ports"].status == "warning"
    evidence = " ".join(checks["port_mode_ports"].evidence)
    assert "ports 3 and 4 differ by" in evidence
    assert "ports 1 and 2 differ" not in evidence
    # ...while the modes on each face are still distinct and ordered.
    assert checks["port_mode_faces"].status == "passed"


def test_no_faces_means_no_face_check():
    names = [c.name for c in check_port_mode_continuity(_samples(distinct))]
    assert names == ["port_mode_continuity"]


def test_modes_that_coincide_on_a_face_are_flagged_with_face_and_frequency():
    def same(mode: int, face: str) -> float:  # noqa: ARG001
        return 2.0  # the second solve found the first mode again

    checks = _by_name(check_port_mode_continuity(_samples(same), faces=FACES))
    assert checks["port_mode_continuity"].status == "passed"
    faces = checks["port_mode_faces"]
    assert faces.status == "warning"
    assert faces.evidence[0] == "24 of 24 mode pairs flagged."
    assert len(faces.evidence) == 1 + MAX_EVIDENCE + 1
    assert "face o1, excitation 1: modes 1 and 2 coincide" in faces.evidence[1]
    assert "at 1 GHz" in faces.evidence[1]


def test_a_near_degenerate_pair_is_flagged_at_the_default_tolerance():
    def close(mode: int, face: str) -> float:  # noqa: ARG001
        return 2.0 * (1.0 if mode == 1 else 1.0 - FACE_MODE_RTOL / 2)

    assert (
        _by_name(check_port_mode_continuity(_samples(close), faces=FACES))[
            "port_mode_faces"
        ].status
        == "warning"
    )
    # ...and passes when the user accepts a tighter limit.
    tight = check_port_mode_continuity(
        _samples(close), faces=FACES, face_rtol=FACE_MODE_RTOL / 100
    )
    assert _by_name(tight)["port_mode_faces"].status == "passed"


def test_modes_out_of_order_on_a_face_are_flagged():
    def swapped(mode: int, face: str) -> float:  # noqa: ARG001
        return {1: 2.0, 2: 2.3, 3: 1.7}[mode]  # mode 2 above mode 1

    faces = _by_name(check_port_mode_continuity(_samples(swapped), faces=FACES))[
        "port_mode_faces"
    ]
    assert faces.status == "warning"
    assert "out of order" in faces.evidence[1]
    assert "modes 1 and 2" in faces.evidence[1]


def test_three_modes_per_face_are_compared_pairwise():
    samples = _samples(distinct, indices=(1, 2, 3, 4, 5, 6))
    faces = _by_name(check_port_mode_continuity(samples, faces=FACES))[
        "port_mode_faces"
    ]
    assert faces.status == "passed"
    # 3 pairs per face and frequency, two faces, 12 frequencies.
    assert faces.evidence[0].startswith(f"{3 * 2 * 12} mode pairs")


def test_nothing_to_compare_is_unknown_not_passed():
    one_mode_each = _samples(distinct, indices=(1, 2))
    faces = _by_name(check_port_mode_continuity(one_mode_each, faces=FACES))[
        "port_mode_faces"
    ]
    assert faces.status == "unknown"


def test_ports_missing_from_the_map_are_skipped_and_named():
    partial = {1: "o1", 3: "o1"}
    faces = _by_name(check_port_mode_continuity(_samples(distinct), faces=partial))[
        "port_mode_faces"
    ]
    assert faces.status == "passed"
    assert "[2, 4]" in faces.evidence[-1]


def test_excitations_are_not_mixed():
    """The same frequency under two excitations is two solves, never a pair."""

    def varying(mode: int, face: str) -> float:
        return distinct(mode, face)

    samples = _samples(varying, excitations=(1, 2))
    faces = _by_name(check_port_mode_continuity(samples, faces=FACES))[
        "port_mode_faces"
    ]
    assert faces.status == "passed"
    assert faces.evidence[0].startswith("48 mode pairs")


def test_a_nonfinite_wave_number_is_left_to_the_continuity_check():
    samples = _samples(distinct)
    samples.append(ModeSample(3, 2, complex(math.nan, 0), 4.0, 1))
    checks = _by_name(check_port_mode_continuity(samples, faces=FACES))
    assert checks["port_mode_continuity"].status == "warning"
    assert checks["port_mode_faces"].status == "passed"


def test_a_negative_face_tolerance_is_refused():
    with pytest.raises(ValueError, match="nonnegative"):
        check_port_mode_continuity(_samples(distinct), faces=FACES, face_rtol=-1.0)


def test_validate_solution_threads_the_faces_through():
    samples = _samples(distinct)
    plain = validate_solution("", mode_samples=samples)
    assert [c.name for c in plain.port_mode_checks] == ["port_mode_continuity"]
    report = validate_solution("", mode_samples=samples, port_faces=FACES)
    assert [c.name for c in report.port_mode_checks] == [
        "port_mode_continuity",
        "port_mode_faces",
    ]
    assert report.warnings == ()


def test_a_driven_log_with_shared_face_entries_is_read_entry_by_entry():
    """The log line carries the Palace index and mode, so entries stay separate."""
    log = (
        "Sweeping excitation index 1 (1/1):\n"
        "Calculating boundary modes at wave ports for "
        "\u03c9/2\u03c0 = 1.000e+00 GHz (4.8e-03)\n"
        " Port 1, mode 1: k\u2099 = 4.820e+01-3.0e-03i m\u207b\u00b9\n"
        " Port 2, mode 1: k\u2099 = 4.820e+01-3.0e-03i m\u207b\u00b9\n"
        " Port 3, mode 2: k\u2099 = 4.190e+01-3.0e-03i m\u207b\u00b9\n"
        " Port 4, mode 2: k\u2099 = 4.190e+01-3.0e-03i m\u207b\u00b9\n"
    )
    samples = port_mode_samples(log.splitlines())
    assert [(s.port, s.mode, s.excitation) for s in samples] == [
        (1, 1, 1),
        (2, 1, 1),
        (3, 2, 1),
        (4, 2, 1),
    ]
    assert [round(s.wave_number.real, 1) for s in samples] == [48.2, 48.2, 41.9, 41.9]


def test_the_sidecar_map_feeds_the_check():
    rows = [
        PortModeEntry(
            i, FACES[i], 1 if FACES[i] == "o1" else 2, MODES[i], i <= 2, i <= 2
        )
        for i in range(1, 7)
    ]
    faces = faces_by_index(rows)
    assert faces == FACES
    checks = _by_name(
        check_port_mode_continuity(
            _samples(distinct, indices=(1, 2, 3, 4, 5, 6)), faces=faces
        )
    )
    assert checks["port_mode_faces"].status == "passed"
