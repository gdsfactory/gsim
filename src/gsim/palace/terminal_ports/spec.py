"""Terminal port layout and voltage paths.

:class:`TerminalPortSpec` names the terminals of a multi-pin port and its reference
conductor as rectangles on the port's cross-section (the
:class:`gsim.common.modes.Conductor` the line-impedance code already uses).
:func:`terminal_paths` draws one voltage path per terminal, from the terminal to
the reference, that never crosses a conductor, and :func:`declare_terminal_paths`
registers those paths on a :class:`~gsim.palace.boundarymode.BoundaryModeSim`.

Everything is stated in the cross-section's own ``(h, v)`` coordinates (um).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, pairwise
from typing import TYPE_CHECKING

import numpy as np

from gsim.common.modes import Conductor, Extent
from gsim.palace.line_impedance import PATH_CLEARANCE_FRACTION

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gsim.palace.boundarymode import BoundaryModeSim


Point = tuple[float, float]


def _overlap_or_touch(a: Extent, b: Extent) -> bool:
    """Whether two rectangles share an interior point or an edge."""
    (a_h0, a_h1), (a_v0, a_v1) = a
    (b_h0, b_h1), (b_v0, b_v1) = b
    return a_h0 <= b_h1 and b_h0 <= a_h1 and a_v0 <= b_v1 and b_v0 <= a_v1


def _check_extent(name: str, extent: Extent) -> None:
    """Refuse a rectangle that is empty, inverted or not finite."""
    (h0, h1), (v0, v1) = extent
    if not all(np.isfinite([h0, h1, v0, v1])) or not (h0 < h1 and v0 < v1):
        raise ValueError(
            f"Conductor {name!r} has the extent {extent}, which is not a "
            "rectangle ((h_min, h_max), (v_min, v_max)) with h_min < h_max and "
            "v_min < v_max."
        )


@dataclass(frozen=True)
class TerminalPortSpec:
    """The conductors of one multi-pin port, as rectangles on its cross-section.

    A port with ``N`` terminals needs ``N`` modes: the terminals are all the
    conductors of the port except the reference, and a conductor that floats
    is a terminal like any other (see the section on grounds in the terminal
    wave port guide).

    Attributes:
        name: Name of the physical port; it prefixes the names of the voltage
            paths, as ``"{port}:{terminal}"``.
        terminals: The conductors whose voltage and current are wanted, in
            the order of the terminal S-parameters.
        reference: The conductor every voltage is measured against.
    """

    name: str
    terminals: Sequence[Conductor]
    reference: Conductor

    def __post_init__(self) -> None:
        """Validate the layout: unique names, real rectangles, no overlaps.

        Raises:
            ValueError: When the port or a conductor has no name, there is no
                terminal, names repeat, the reference is also a terminal, a
                conductor is not a proper rectangle, or two conductors
                overlap or touch.
        """
        object.__setattr__(self, "terminals", tuple(self.terminals))
        if not self.name:
            raise ValueError("A terminal port needs a name.")
        if not self.terminals:
            raise ValueError(f"Port {self.name!r} has no terminals.")
        everyone = [*self.terminals, self.reference]
        names = [c.name for c in everyone]
        if any(not n for n in names):
            raise ValueError(f"Port {self.name!r} has an unnamed conductor.")
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"Port {self.name!r}: conductor names must be unique, and the "
                f"reference may not also be a terminal; repeated: {repeated}."
            )
        for conductor in everyone:
            _check_extent(conductor.name, conductor.extent)
        for first, second in combinations(everyone, 2):
            if _overlap_or_touch(first.extent, second.extent):
                raise ValueError(
                    f"Port {self.name!r}: conductors {first.name!r} and "
                    f"{second.name!r} overlap or touch, so there is no gap to "
                    "integrate a voltage across."
                )

    @property
    def terminal_names(self) -> tuple[str, ...]:
        """The terminal names, in port order."""
        return tuple(t.name for t in self.terminals)


def _segment_hits_box(p: Point, q: Point, box: Extent) -> bool:
    """Whether the segment ``p``-``q`` meets the closed rectangle ``box``."""
    (h0, h1), (v0, v1) = box
    t_lo, t_hi = 0.0, 1.0
    for start, delta, lo, hi in (
        (p[0], q[0] - p[0], h0, h1),
        (p[1], q[1] - p[1], v0, v1),
    ):
        if delta == 0.0:
            if start < lo or start > hi:
                return False
            continue
        t_a, t_b = (lo - start) / delta, (hi - start) / delta
        if t_a > t_b:
            t_a, t_b = t_b, t_a
        t_lo, t_hi = max(t_lo, t_a), min(t_hi, t_b)
        if t_lo > t_hi:
            return False
    return True


def _mid(lo: float, hi: float) -> float:
    """Midpoint of an interval."""
    return 0.5 * (lo + hi)


def _direct_path(term: Extent, ref: Extent, fraction: float) -> list[Point] | None:
    """Straight path across the gap between two conductors, if they have one.

    Across the horizontal gap from face to face at each conductor's
    mid-height when the two are side by side, else across the vertical gap
    at the middle of their common width. Each end keeps ``fraction`` of the
    smallest of the gap and the two conductors' thicknesses off its face, in
    the gap, as :func:`gsim.palace.line_impedance.line_impedance_paths` does.

    Args:
        term: The terminal's rectangle.
        ref: The reference conductor's rectangle.
        fraction: Inset fraction.

    Returns:
        Two points, or ``None`` when the rectangles are separated along
        neither axis.
    """
    (th0, th1), (tv0, tv1) = term
    (rh0, rh1), (rv0, rv1) = ref
    gap_h = max(rh0 - th1, th0 - rh1)
    gap_v = max(rv0 - tv1, tv0 - rv1)
    if gap_h > 0.0:
        a, b, sign = (th1, rh0, 1.0) if rh0 > th1 else (th0, rh1, -1.0)
        inset = fraction * min(gap_h, tv1 - tv0, rv1 - rv0)
        return [
            (a + sign * inset, _mid(tv0, tv1)),
            (b - sign * inset, _mid(rv0, rv1)),
        ]
    if gap_v > 0.0:
        lo, hi = max(th0, rh0), min(th1, rh1)
        if hi <= lo:
            return None
        a, b, sign = (tv1, rv0, 1.0) if rv0 > tv1 else (tv0, rv1, -1.0)
        inset = fraction * min(gap_v, th1 - th0, rh1 - rh0)
        h_mid = _mid(lo, hi)
        return [(h_mid, a + sign * inset), (h_mid, b - sign * inset)]
    return None


def _detour_path(
    index: int,
    boxes: Sequence[Extent],
    ref: Extent,
    domain: Extent,
    fraction: float,
) -> list[Point] | None:
    """Path out of a terminal's gap, over all the metal, down onto the reference.

    The path leaves the terminal at mid-height, into the gap on the side the
    reference lies on, rises there to a height above every conductor, runs
    across to the middle of the reference, and comes down onto its top face.
    The rise above the metal is the smaller of the gap it rose through and half
    the room the domain leaves above the metal, so the path stays close to the
    conductors rather than wandering out to the domain wall.

    Args:
        index: Position of the terminal in ``boxes``.
        boxes: Every conductor's rectangle (terminals, then the reference).
        ref: The reference conductor's rectangle.
        domain: The meshed domain.
        fraction: Inset fraction.

    Returns:
        The polyline (consecutive points distinct), or ``None`` when the
        domain leaves no room above the metal or the terminal has no gap on
        that side.
    """
    (th0, th1), (tv0, tv1) = boxes[index]
    (rh0, rh1), (rv0, rv1) = ref
    (d_h0, d_h1), (_, d_v1) = domain
    v_mid = _mid(tv0, tv1)
    direction = 1.0 if _mid(rh0, rh1) >= _mid(th0, th1) else -1.0
    face = th1 if direction > 0 else th0
    gaps = [(d_h1 - face) if direction > 0 else (face - d_h0)]
    for other, ((b_h0, b_h1), (b_v0, b_v1)) in enumerate(boxes):
        if other == index or not b_v0 <= v_mid <= b_v1:
            continue
        if direction > 0 and b_h0 >= face:
            gaps.append(b_h0 - face)
        elif direction < 0 and b_h1 <= face:
            gaps.append(face - b_h1)
    gap = min(gaps)
    top = max(v1 for _, (_, v1) in boxes)
    headroom = d_v1 - top
    if gap <= 0.0 or headroom <= 0.0:
        return None
    x_exit = face + direction * fraction * min(gap, tv1 - tv0)
    rise = min(0.5 * headroom, gap)
    y_top = top + rise
    x_down = _mid(rh0, rh1)
    y_end = rv1 + fraction * min(rise, rv1 - rv0)
    path = [(x_exit, v_mid), (x_exit, y_top), (x_down, y_top), (x_down, y_end)]
    return [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]


def _inside(point: Point, domain: Extent) -> bool:
    """Whether a point is strictly inside the domain rectangle."""
    (h0, h1), (v0, v1) = domain
    return h0 < point[0] < h1 and v0 < point[1] < v1


def terminal_paths(
    spec: TerminalPortSpec,
    domain: Extent,
    *,
    clearance_fraction: float = PATH_CLEARANCE_FRACTION,
) -> dict[str, tuple[Point, ...]]:
    """Draw one voltage path per terminal, from the terminal to the reference.

    Where the terminal and the reference face each other across a clear gap the
    path is the straight line across it, the same path
    :func:`gsim.palace.line_impedance.line_impedance_paths` uses. Where another
    conductor stands in the way, the path leaves the terminal at mid-height into
    the gap beside it, rises above all the metal, runs across and comes down
    onto the reference. Every path starts and ends a hair inside the gap
    (``clearance_fraction`` of the gap, as in
    :data:`gsim.palace.line_impedance.PATH_CLEARANCE_FRACTION`), on the meshed
    domain rather than on a conductor's face, and no segment of it meets any
    conductor's rectangle.

    The voltage of a full-wave mode depends a little on the path; for the
    quasi-TEM modes of an on-chip line the dependence is small, and the rule
    above is fixed so that it is the same in every run.

    Args:
        spec: The port's conductors.
        domain: ``((h_min, h_max), (v_min, v_max))`` of the meshed domain
            the paths are sampled on (um).
        clearance_fraction: Inset of the path ends from the conductor faces,
            as a fraction of the gap.

    Returns:
        ``{terminal name: ((h, v), ...)}`` in terminal order.

    Raises:
        ValueError: When the domain is not a rectangle, a conductor is not
            inside it, or no path from a terminal to the reference avoids
            the other conductors and stays in the domain.
    """
    _check_extent("domain", domain)
    conductors = [*spec.terminals, spec.reference]
    (d_h0, d_h1), (d_v0, d_v1) = domain
    for conductor in conductors:
        (h0, h1), (v0, v1) = conductor.extent
        if h0 < d_h0 or h1 > d_h1 or v0 < d_v0 or v1 > d_v1:
            raise ValueError(
                f"Conductor {conductor.name!r} at {conductor.extent} is not inside "
                f"the domain {domain}."
            )
    boxes = [c.extent for c in conductors]
    ref = spec.reference.extent
    paths: dict[str, tuple[Point, ...]] = {}
    for index, terminal in enumerate(spec.terminals):
        candidates = (
            _direct_path(terminal.extent, ref, clearance_fraction),
            _detour_path(index, boxes, ref, domain, clearance_fraction),
        )
        for path in candidates:
            if (
                path is not None
                and len(path) >= 2
                and all(_inside(p, domain) for p in path)
                and not any(
                    _segment_hits_box(p, q, box)
                    for p, q in pairwise(path)
                    for box in boxes
                )
            ):
                paths[terminal.name] = tuple(path)
                break
        else:
            raise ValueError(
                f"No voltage path from terminal {terminal.name!r} to the reference "
                f"{spec.reference.name!r} of port {spec.name!r} avoids the other "
                "conductors inside the domain: a conductor spans the way over the "
                "metal, or the domain leaves no room above it."
            )
    return paths


def declare_terminal_paths(
    sim: BoundaryModeSim,
    spec: TerminalPortSpec,
    domain: Extent,
    *,
    nsamples: int = 100,
    clearance_fraction: float = PATH_CLEARANCE_FRACTION,
) -> list[int]:
    """Register every terminal's voltage path on a boundary-mode solve.

    The paths are postprocessing only: they load nothing and leave the
    eigenproblem as it was. Palace then reports the complex voltage of each mode
    on each path in ``mode-V.csv``, under the index this returns.

    Args:
        sim: The boundary-mode simulation about to be solved.
        spec: The port's conductors.
        domain: The meshed domain (um), as for :func:`terminal_paths`.
        nsamples: Quadrature order of each line integral in Palace.
        clearance_fraction: Inset of the path ends, as for :func:`terminal_paths`.

    Returns:
        The 1-based Palace index of each terminal's path, in terminal order.
        A path named ``"{port}:{terminal}"`` that was declared before is
        replaced and keeps its index.
    """
    paths = terminal_paths(spec, domain, clearance_fraction=clearance_fraction)
    return [
        sim.add_impedance_path(
            f"{spec.name}:{name}",
            voltage=[[h, v] for h, v in points],
            nsamples=nsamples,
        )
        for name, points in paths.items()
    ]
