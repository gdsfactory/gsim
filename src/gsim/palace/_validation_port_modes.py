"""Continuity check of the wave-port mode that Palace selects at each frequency.

Palace picks the port mode only by its index, so a spurious or closed-box
solution (gsim#375, gsim#408) is silent in the S-parameters. A physical port
mode changes slowly with frequency, so an isolated jump of its effective index
``n_eff = Re(k_n) / k0`` is evidence that a different mode was selected at that
frequency. The check reads the samples from the driven log and never reruns a
simulation. A hit is a ``warning``: a mode that really changes quickly, for
example close to cutoff or on a very coarse sweep, is flagged in the same way.
"""

from __future__ import annotations

import itertools
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence

from gsim.palace._validation_models import ModeSample, SolutionCheck

SPEED_OF_LIGHT_M_S = 299_792_458.0
"""Vacuum speed of light in m/s, used for ``k0 = 2 pi f / c``."""

CONTINUITY_RTOL = 0.01
"""Largest relative distance of ``n_eff`` from the median of its nearest
neighbours (in frequency) before a point is flagged. The good points of the
real test logs scatter by at most 0.05 %, the spurious solves differ by 17 % or
more. A mode that really changes by more than this between neighbouring
frequencies (close to cutoff, very coarse sweep) is flagged as well.

When the neighbour median is exactly zero a relative distance is undefined. The
point is then flagged when ``abs(n_eff)`` exceeds this same number, read as an
absolute tolerance: ``n_eff`` is dimensionless and of order one, so ``0.01`` is
a small absolute step."""

PORT_AGREEMENT_RTOL = 0.005
"""Largest relative difference of ``n_eff`` between two ports solved at the same
frequency (only when the ports share a cross-section). The two identical ports
of the real test logs agree exactly in the printed digits; the spurious pairs
differ by 11 % and 49 %."""

FACE_MODE_RTOL = 0.005
"""Two modes of one face whose ``n_eff`` differ by less than this (relative)
are reported as coinciding (``port_mode_faces``). It is the size of the
port-to-port tolerance: the 2D solves of one face scatter by far less, and
two physical modes of a line are further apart than that unless they are
degenerate."""

NEIGHBOUR_COUNT = 5
"""Number of nearest points (by frequency, same port, mode and excitation) whose
median is the continuity reference. Up to two spurious neighbours are tolerated."""

MIN_DISTINCT_FREQUENCIES = 6
"""A series with fewer distinct frequencies is not judged: continuity cannot be
tested on so few points, so the result is ``unknown``, never ``passed``."""

MAX_EVIDENCE = 5
"""Number of flagged points listed in a check; the rest are only counted."""

CONTINUITY_CHECK = "port_mode_continuity"
PORT_AGREEMENT_CHECK = "port_mode_ports"
FACE_MODES_CHECK = "port_mode_faces"

_Key = tuple[int, int, int | None]
_Point = tuple[float, float]  # (frequency in GHz, effective index)


def _effective_index(sample: ModeSample, frequency_ghz: float) -> float:
    """Return ``Re(k_n) / k0``; NaN if the wave number is not finite."""
    wave_number = sample.wave_number
    if not (math.isfinite(wave_number.real) and math.isfinite(wave_number.imag)):
        return math.nan
    k0 = 2 * math.pi * frequency_ghz * 1e9 / SPEED_OF_LIGHT_M_S
    return wave_number.real / k0


def _series(samples: Sequence[ModeSample]) -> dict[_Key, list[_Point]]:
    """Group samples by (port, mode, excitation), sorted by frequency.

    Samples without a positive finite frequency are skipped: eigenmode and
    boundary-mode observations carry none, so they are ignored here.
    """
    groups: dict[_Key, list[_Point]] = defaultdict(list)
    for sample in samples:
        frequency = sample.frequency_ghz
        if frequency is None or not math.isfinite(frequency) or frequency <= 0:
            continue
        key = (sample.port, sample.mode, sample.excitation)
        groups[key].append((frequency, _effective_index(sample, frequency)))
    return {key: sorted(points) for key, points in groups.items()}


def _sort_key(key: _Key) -> tuple[int, int, int]:
    """Order series by port, mode and excitation (None sorts first)."""
    port, mode, excitation = key
    return (-1 if excitation is None else excitation, port, mode)


def _label(key: _Key) -> str:
    """Human-readable name of a (port, mode, excitation) series."""
    port, mode, excitation = key
    where = "" if excitation is None else f", excitation {excitation}"
    return f"port {port}, mode {mode}{where}"


def _neighbour_median(points: Sequence[_Point], index: int) -> float | None:
    """Median ``n_eff`` of the nearest other points in frequency.

    ``points`` must be sorted by frequency and finite, so the nearest points
    lie within ``NEIGHBOUR_COUNT`` positions on either side.
    """
    frequency = points[index][0]
    lower = max(0, index - NEIGHBOUR_COUNT)
    candidates = [
        (abs(points[other][0] - frequency), other)
        for other in range(lower, min(len(points), index + NEIGHBOUR_COUNT + 1))
        if other != index
    ]
    candidates.sort()
    nearest = [points[other][1] for _, other in candidates[:NEIGHBOUR_COUNT]]
    return statistics.median(nearest) if nearest else None


def _distinct_frequencies(points: Sequence[_Point]) -> int:
    """Number of different frequencies in a series."""
    return len({frequency for frequency, _ in points})


def _summary(flagged: Sequence[str], total: int, what: str) -> tuple[str, ...]:
    """Return evidence lines: a count, the first points, then the remainder."""
    lines = [f"{len(flagged)} of {total} {what} flagged."]
    lines.extend(flagged[:MAX_EVIDENCE])
    if len(flagged) > MAX_EVIDENCE:
        lines.append(f"... and {len(flagged) - MAX_EVIDENCE} more.")
    return tuple(lines)


def _departure(n_eff: float, reference: float, rtol: float) -> str | None:
    """Describe how ``n_eff`` departs from ``reference``; None if within ``rtol``.

    The distance is relative to ``abs(reference)``. A reference of exactly zero
    has no relative scale, so the absolute difference ``abs(n_eff)`` is compared
    with ``rtol`` instead; this never divides by zero.
    """
    difference = abs(n_eff - reference)
    if reference == 0:
        if difference <= rtol:
            return None
        return f"neighbour median 0 (absolute difference {difference:.4g} > {rtol:.2g})"
    if difference <= rtol * abs(reference):
        return None
    deviation = 100 * difference / abs(reference)
    return f"neighbour median {reference:.4f} ({deviation:.2f} % > {100 * rtol:.2g} %)"


def _continuity(series: dict[_Key, list[_Point]], rtol: float) -> SolutionCheck:
    """Flag points whose n_eff departs from the median of their frequency neighbours."""
    flagged: list[str] = []
    not_judged: list[str] = []
    judged = 0
    solves = sum(len(points) for points in series.values())
    for key in sorted(series, key=_sort_key):
        points = series[key]
        for frequency, n_eff in points:
            if not math.isfinite(n_eff):
                flagged.append(
                    f"{_label(key)}: nonfinite wave number at {frequency:.4g} GHz"
                )
        finite = [point for point in points if math.isfinite(point[1])]
        distinct = _distinct_frequencies(finite)
        if distinct < MIN_DISTINCT_FREQUENCIES:
            not_judged.append(
                f"{_label(key)}: only {distinct} distinct frequencies "
                f"(need {MIN_DISTINCT_FREQUENCIES}); not judged."
            )
            continue
        judged += 1
        for index, (frequency, n_eff) in enumerate(finite):
            reference = _neighbour_median(finite, index)
            if reference is None:
                continue
            departure = _departure(n_eff, reference, rtol)
            if departure is None:
                continue
            flagged.append(
                f"{_label(key)}: n_eff = {n_eff:.4f} at {frequency:.4g} GHz, "
                f"{departure}"
            )
    if flagged:
        return SolutionCheck(
            CONTINUITY_CHECK, "warning", _summary(flagged, solves, "port solves")
        )
    if not judged:
        return SolutionCheck(CONTINUITY_CHECK, "unknown", tuple(not_judged))
    return SolutionCheck(
        CONTINUITY_CHECK,
        "passed",
        (
            f"{judged} series, {solves} solves: no n_eff differs more than "
            f"{100 * rtol:.2g} % from its neighbour median.",
            *not_judged,
        ),
    )


def _port_agreement(series: dict[_Key, list[_Point]], rtol: float) -> SolutionCheck:
    """Flag frequencies where ports solved for the same mode disagree in n_eff.

    Every pair of ports that has the same mode and excitation is compared at the
    frequencies both were solved at. Pairs without a common frequency are noted
    and do not hide the pairs that can be compared.
    """
    groups: dict[tuple[int, int | None], dict[int, list[_Point]]] = defaultdict(dict)
    for (port, mode, excitation), points in series.items():
        groups[mode, excitation][port] = points
    flagged: list[str] = []
    notes: list[str] = []
    compared = 0
    port_pairs: set[tuple[int, int]] = set()  # distinct pairs that were compared
    worst: tuple[float, str] | None = None  # (difference in %, description)
    ordered = sorted(
        groups.items(),
        key=lambda item: (-1 if item[0][1] is None else item[0][1], item[0][0]),
    )
    for (mode, excitation), ports in ordered:
        where = "" if excitation is None else f", excitation {excitation}"
        for first, other in itertools.combinations(sorted(ports), 2):
            reference = defaultdict(list)
            for frequency, n_eff in ports[first]:
                reference[frequency].append(n_eff)
            shared = 0
            for frequency, n_eff in ports[other]:
                for n_first in reference.get(frequency, ()):
                    shared += 1
                    scale = max(abs(n_first), abs(n_eff))
                    if not (math.isfinite(n_first) and math.isfinite(n_eff)):
                        continue  # reported by the continuity check
                    difference = abs(n_first - n_eff)
                    if difference <= rtol * scale:
                        continue
                    percent = 100 * difference / scale
                    flagged.append(
                        f"mode {mode}{where}: ports {first} and {other} differ "
                        f"by {percent:.2f} % at {frequency:.4g} GHz "
                        f"(n_eff {n_first:.4f} vs {n_eff:.4f}, "
                        f"limit {100 * rtol:.2g} %)"
                    )
                    if worst is None or percent > worst[0]:
                        text = f"worst pair: mode {mode}{where}: "
                        text += f"ports {first} and {other} differ by {percent:.2f} %"
                        worst = (percent, f"{text} at {frequency:.4g} GHz.")
            if shared:
                compared += shared
                port_pairs.add((first, other))
            else:
                notes.append(
                    f"mode {mode}{where}: ports {first} and {other} share no "
                    "solve frequency; not compared."
                )
    if flagged:
        lines = list(_summary(flagged, compared, "port pairs"))
        if worst is not None and len(port_pairs) > 1:
            # With several port pairs the listed points may omit the worst one.
            lines.insert(1, worst[1])
        return SolutionCheck(PORT_AGREEMENT_CHECK, "warning", tuple(lines))
    if not compared:
        return SolutionCheck(
            PORT_AGREEMENT_CHECK,
            "unknown",
            (*notes, "No two ports were solved at a common frequency.")
            if notes
            else ("Only one port found; nothing to compare.",),
        )
    return SolutionCheck(
        PORT_AGREEMENT_CHECK,
        "passed",
        (
            f"{compared} port pairs agree within {100 * rtol:.2g} %.",
            *notes,
        ),
    )


def _face_modes(
    series: dict[_Key, list[_Point]], faces: Mapping[int, str], rtol: float
) -> SolutionCheck:
    """Check that the modes of one face are distinct and ranked by wave number.

    A multimode wave port has one Palace entry per mode on the same face, and
    every entry solves its own 2D eigenproblem. Two entries that report the
    same ``n_eff`` have most likely found the same mode twice (a degenerate
    pair, awslabs/palace#328, in an arbitrary basis on each entry), so their
    S-parameter columns cannot be told apart. Palace ranks the modes by
    decreasing wave number, so a later mode with a larger ``n_eff`` than an
    earlier one means the separate solves disagree about the order.

    Entries are grouped by face, excitation and frequency; entries on different
    faces are never compared with each other here (that is ``port_mode_ports``).
    """
    solves: dict[tuple[str, int | None, float], dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    unmapped: set[int] = set()
    for (port, mode, excitation), points in series.items():
        face = faces.get(port)
        if face is None:
            unmapped.add(port)
            continue
        for frequency, n_eff in points:
            solves[face, excitation, frequency][mode].append(n_eff)
    flagged: list[str] = []
    compared = 0
    for (face, excitation, frequency), by_mode in sorted(
        solves.items(),
        key=lambda item: (
            item[0][0],
            -1 if item[0][1] is None else item[0][1],
            item[0][2],
        ),
    ):
        where = "" if excitation is None else f", excitation {excitation}"
        for first, second in itertools.combinations(sorted(by_mode), 2):
            for n_first in by_mode[first]:
                for n_second in by_mode[second]:
                    if not (math.isfinite(n_first) and math.isfinite(n_second)):
                        continue  # reported by the continuity check
                    compared += 1
                    scale = max(abs(n_first), abs(n_second))
                    gap = n_first - n_second  # positive: modes in Palace's order
                    where_at = f"face {face}{where}: modes {first} and {second}"
                    detail = (
                        f"n_eff {n_first:.4f} vs {n_second:.4f} at {frequency:.4g} GHz"
                    )
                    if abs(gap) <= rtol * scale:
                        flagged.append(
                            f"{where_at} coincide within {100 * rtol:.2g} % ({detail})"
                        )
                    elif gap < 0:
                        flagged.append(
                            f"{where_at} are out of order: the later mode has the "
                            f"larger n_eff ({detail})"
                        )
    notes = (
        (f"Ports without a face in the map were skipped: {sorted(unmapped)}.",)
        if unmapped
        else ()
    )
    if flagged:
        return SolutionCheck(
            FACE_MODES_CHECK,
            "warning",
            (*_summary(flagged, compared, "mode pairs"), *notes),
        )
    if not compared:
        return SolutionCheck(
            FACE_MODES_CHECK,
            "unknown",
            (
                "No face has two modes solved at one frequency; nothing to compare.",
                *notes,
            ),
        )
    return SolutionCheck(
        FACE_MODES_CHECK,
        "passed",
        (
            (
                f"{compared} mode pairs on the same face are distinct (more than "
                f"{100 * rtol:.2g} % apart) and in order of decreasing n_eff."
            ),
            *notes,
        ),
    )


def check_port_mode_continuity(
    samples: Sequence[ModeSample],
    *,
    compare_ports: bool = False,
    rtol: float = CONTINUITY_RTOL,
    port_rtol: float = PORT_AGREEMENT_RTOL,
    faces: Mapping[int, str] | None = None,
    face_rtol: float = FACE_MODE_RTOL,
) -> tuple[SolutionCheck, ...]:
    """Check that the selected wave-port mode varies smoothly with frequency.

    For every (port, mode, excitation) series the effective index
    ``n_eff = Re(k_n) / k0`` is compared with the median of its
    ``NEIGHBOUR_COUNT`` nearest points in frequency. Points further than
    ``rtol`` (relative) from that median, and nonfinite wave numbers, give a
    ``"warning"`` whose evidence lists port, excitation, frequency, ``n_eff``
    and the neighbour median. A series with fewer than
    ``MIN_DISTINCT_FREQUENCIES`` frequencies is ``"unknown"``.

    A neighbour median of exactly zero has no relative scale. Such a point is
    flagged when ``abs(n_eff - median) = abs(n_eff)`` is larger than ``rtol``,
    taken as an absolute tolerance, and the evidence gives the absolute
    difference instead of a percentage. A series that is zero everywhere is
    therefore not flagged. No input makes this function divide by zero.

    Args:
        samples: Port-mode observations, normally ``SolutionReport.mode_samples``.
            Observations without a frequency (eigenmode, boundary mode) are
            ignored.
        compare_ports: Also require ports solved at the same frequency to agree
            within ``port_rtol``. Only set this when all ports share one
            cross-section: ports of different width or stack have different
            modes and would be flagged wrongly. Every pair of ports with the
            same mode and excitation is compared at the frequencies they have
            in common; pairs without a common frequency are noted and do not
            stop the other pairs from being compared. With more than one
            comparable pair the evidence names the worst pair.
        rtol: Relative continuity tolerance of ``n_eff`` (absolute when the
            neighbour median is zero).
        port_rtol: Relative port-to-port tolerance of ``n_eff``.
        faces: For multimode wave ports, the face each Palace port index
            sits on (``gsim.palace.ports.modes.faces_by_index`` of the
            ``port_modes.json`` sidecar). The entries of one face follow
            different modes, so they are not compared with each other by
            ``port_mode_ports``, which only compares entries that follow the
            same mode. With ``faces`` the ``port_mode_faces`` check is added:
            on every face the modes must be distinct in ``n_eff`` (within
            ``face_rtol``) and ranked by decreasing ``n_eff``.
        face_rtol: Relative ``n_eff`` distance below which two modes of one
            face count as coinciding.

    Returns:
        ``()`` when no sample has a frequency (no driven wave-port solve),
        otherwise the ``port_mode_continuity`` check followed by the
        ``port_mode_ports`` check when requested and the ``port_mode_faces``
        check when ``faces`` is given. Statuses are ``"passed"``,
        ``"warning"`` or ``"unknown"``; this check never reports ``"failed"``.

    Raises:
        ValueError: If a tolerance is negative or not finite.
    """
    for tolerance in (rtol, port_rtol, face_rtol):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Port-mode tolerances must be finite and nonnegative.")
    series = _series(samples)
    if not series:
        return ()
    checks = [_continuity(series, rtol)]
    if compare_ports:
        checks.append(_port_agreement(series, port_rtol))
    if faces is not None:
        checks.append(_face_modes(series, faces, face_rtol))
    return tuple(checks)
