"""Choosing the physical line Mode out of a set of solved Modes.

An RF solve returns several Modes: the quasi-TEM line Mode a designer
wants, plus evanescent and spurious ones the discretization produces.
Picking the physical one is a library concern rather than a heuristic
every caller re-invents, so :func:`select_line_mode` owns the default
rule, the substitution point for unusual lines, and the ambiguity
warning.

The default rule keeps Modes that propagate — ``Re(n_eff)`` above the
light line, and losing no more per unit length than they advance in
phase — and picks the slowest-travelling of them: the right choice for a
line with a single signal conductor, where the loaded quasi-TEM Mode
carries the highest effective index. Lines that break that assumption
pass their own candidate rule instead.

How much loss still counts as propagating is the ``max_loss_ratio``
bound on ``|Im(n_eff)| / Re(n_eff)``. It defaults to one — the point
where a Mode decays as fast as it advances — because that is the widest
bound that is a bound at all; a caller who knows the answer is a
transmission line rather than a waveguide Mode should tighten it, since
a discretization's spurious Modes cluster just inside it.

The selection itself is on the effective index alone, so it serves any
guided Mode and not only a line's — an optical Stage selects its
waveguide Mode through :func:`select_line_mode` with nothing but the
guided-index floor. What is a transmission line's own rule is the loss
bound above and the candidate rule that carries it; the name follows
the caller that has them.

Modes are read duck-typed: anything with an ``n_eff`` attribute (femwell
``Mode``), a mapping with an ``"n_eff"`` key (the Palace result rows), or
a bare complex number works.

A shielded two-electrode line has a second propagating Mode the index
alone does not separate from the line Mode: the one on which both
electrodes sit at one potential and return their current through the
metallic wall around them. :func:`common_mode_fraction` tells the two
apart from the electrode currents, which either Backend can measure
once a Mode is in hand, and :func:`wall_mode_from_currents` turns that
into the diagnostic a Route reports.

What a Route reads off a selected RF Mode is one :class:`LineReading`
— its index, its characteristic impedance and whether it is the wall
Mode — given the two electrodes as :class:`Conductor` descriptors.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

__all__ = [
    "MAX_COMMON_MODE_FRACTION",
    "MAX_GAIN_RATIO",
    "Conductor",
    "Extent",
    "LineModeRule",
    "LineReading",
    "NoLineModeError",
    "common_mode_fraction",
    "mode_index",
    "propagating_modes",
    "select_line_mode",
    "wall_mode_from_currents",
    "z0_power_current",
]

#: ``((h_min, h_max), (v_min, v_max))`` of a rectangle on the Cross-section (um).
Extent = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class Conductor:
    """One electrode of a Traveling-wave line, as a current integral names it.

    A conductor reaches a mode solve one of two ways (ADR 0003), and its
    current is read one of two ways: meshed as a Region of metal
    (``"volume"``) it carries a conduction current over its elements,
    named by its Region; left out of the meshed domain as a perfect
    conductor (``"pec"``) it carries Ampere's contour integral around
    the hole its outline leaves, located by its extent.

    Which Backend reads which way is not symmetric: Palace's contour
    integral around the electrode's outline reads the same enclosed
    current for either model and so never branches on this field, while
    femwell's ``electrode_current`` does, taking the conduction current
    through the metal for a ``"volume"`` conductor and the contour
    integral for a ``"pec"`` one. The field stays as it is either way
    because it describes how the metal reached the mesh (ADR 0003),
    which is true whoever reads it.

    Attributes:
        name: Region name of the electrode.
        extent: ``((h_min, h_max), (v_min, v_max))`` the electrode
            occupies on the Cross-section (um).
        model: How the metal reached the mesh — ``"volume"`` or ``"pec"``.
    """

    name: str
    extent: Extent
    model: Literal["volume", "pec"] = "volume"


@dataclass(frozen=True)
class LineReading:
    """What a Route read off one selected RF Mode.

    Attributes:
        n_eff: The Mode's complex effective index.
        z0_ohm: Its characteristic impedance (ohm), by the power-current
            definition; NaN when the Route could not read it.
        wall_mode: Whether the Mode is the wall Mode rather than the line
            Mode (ADR 0005); ``None`` when the Route had nothing to tell
            them apart with.
        diagnostic: How the Route told them apart, or why it could not —
            the clause a warning quotes when ``wall_mode`` is true.
    """

    n_eff: complex
    z0_ohm: complex
    wall_mode: bool | None
    diagnostic: str = ""


#: A candidate rule: given every solved Mode, return the physical ones.
LineModeRule = Callable[[Sequence[Any]], Sequence[Any]]

#: How much ``Im(n_eff) / Re(n_eff)`` may run positive before a Mode counts
#: as growing rather than propagating. In the ``exp(+i omega t)``
#: convention loss is a negative imaginary part; a positive one is gain,
#: which a passive line cannot have, so it marks a spurious Mode of the
#: eigenvalue search. The bound leaves room for the numerical noise a
#: lossless Mode carries.
MAX_GAIN_RATIO: float = 1e-3


#: Above this :func:`common_mode_fraction` a Mode is not the line Mode
#: between the two electrodes. The line Mode's electrodes carry equal and
#: opposite currents (a fraction near zero, a few per mille on the
#: shipped demo); on the wall Mode both carry the same current and the
#: metallic wall returns it (a fraction of one). Halfway between the two
#: is a bound neither can drift across through the loading alone.
MAX_COMMON_MODE_FRACTION: float = 0.5


class NoLineModeError(ValueError):
    """No solved Mode qualifies as the physical line Mode."""


def common_mode_fraction(i_signal: complex, i_return: complex) -> float:
    """How much of a Mode's electrode current is common to both electrodes.

    ``|I_signal + I_return| / (|I_signal| + |I_return|)``, with both
    currents read the same way round — each as the current a contour
    around its own electrode encloses, or each as the conduction current
    through its own metal. The line Mode between the two electrodes has
    them equal and opposite, so the fraction is near zero; the Mode
    between both electrodes together and the shielding wall has them
    equal and alike, so it is near one.

    Args:
        i_signal: Longitudinal current on the signal electrode.
        i_return: Longitudinal current on the return electrode, in the
            same convention.

    Returns:
        The fraction in ``[0, 1]``, or NaN when neither electrode
        carries any current.
    """
    total = abs(i_signal) + abs(i_return)
    if total == 0.0:
        return math.nan
    return float(abs(i_signal + i_return) / total)


def wall_mode_from_currents(
    i_signal: complex, i_return: complex
) -> tuple[bool | None, str]:
    """Tell the line Mode from the wall Mode by the two electrode currents.

    The pairing logic behind the femwell Route's reading, on its own so
    it is testable without a solved Mode: both currents read the same
    way round, their :func:`common_mode_fraction` against
    :data:`MAX_COMMON_MODE_FRACTION`.

    Args:
        i_signal: Longitudinal current on the signal electrode.
        i_return: Longitudinal current on the return electrode, in the
            same convention.

    Returns:
        ``(wall_mode, diagnostic)``: whether the Mode is the wall Mode,
        or ``None`` when neither electrode carries any current, and the
        clause saying what was measured.
    """
    fraction = common_mode_fraction(i_signal, i_return)
    if math.isnan(fraction):
        return (
            None,
            "neither electrode carries any current, so nothing says which mode this is",
        )
    if fraction > MAX_COMMON_MODE_FRACTION:
        return True, (
            f"whose two electrodes carry currents {fraction:.0%} in common rather "
            "than equal and opposite: they sit at one potential, so this is the "
            "mode between them and the window wall rather than the line mode "
            "between them"
        )
    return False, (
        f"whose two electrodes carry currents {fraction:.0%} in common: equal "
        "and opposite, the line mode between them"
    )


def z0_power_current(power: complex, current: complex) -> complex:
    """The power-current impedance of a line Mode.

    ``Z_0 = 2 P / |I|^2`` (the definition Marks and Williams give):
    the one definition both Routes read a line's characteristic
    impedance by, given the two integrals each measures its own way —
    the complex Poynting flux over the whole Cross-section, and the
    longitudinal current on the signal conductor.
    Sharing the definition rather than the quadrature is the point: a
    Palace Mode's integrals are numpy over nodal arrays read back from a
    file, a femwell Mode's are skfem forms over the Basis the solver
    still holds, and neither belongs in the other's module.

    Both arguments arrive in whatever coordinate scale the Backend
    integrated in — the um meshes in use put the power at ``1e12`` times
    SI and the current at ``1e6`` — and the ratio ``2P / |I|^2`` cancels
    it, so the result is in ohms whichever Backend asked and no unit
    conversion happens here.

    Args:
        power: Complex Poynting flux of the Mode over the
            Cross-section.
        current: Longitudinal current on the signal conductor. Only its
            magnitude is read, so the two electrodes' opposite signs and
            either Route's phase convention make no difference.

    Returns:
        The complex characteristic impedance in ohms, its real part
        positive.

    Raises:
        ValueError: When the current is zero, which means the Mode
            carries no current on that conductor and so has no
            power-current impedance.
    """
    if current == 0:
        raise ValueError(
            "The mode carries no current on the signal conductor, so it has "
            "no power-current impedance."
        )
    z0 = complex(2.0 * power / (abs(current) ** 2))
    # An eigenmode's propagation direction is the solver's to choose, and
    # the Poynting flux changes sign with it while |I|^2 does not. The
    # line's impedance does not depend on which way the solver looked, so
    # a Mode saved travelling against the plane normal is flipped back.
    # Measured on the shipped cross-Route gate: suppressing the flip puts
    # Palace 203% from femwell instead of 2.98%, and the flipped reading
    # lands on femwell's phase, which is what shows it is the right
    # correction rather than a sign forced positive.
    return -z0 if z0.real < 0.0 else z0


def mode_index(mode: Any) -> complex:
    """Effective index of a solved Mode.

    Args:
        mode: A solver Mode (an object with ``n_eff``, a mapping with an
            ``"n_eff"`` key, or a bare complex effective index).

    Returns:
        The complex effective index.
    """
    if isinstance(mode, Mapping):
        return complex(mode["n_eff"])
    n_eff = getattr(mode, "n_eff", None)
    if n_eff is not None:
        return complex(n_eff)
    return complex(mode)


def propagating_modes[ModeT](
    modes: Sequence[ModeT],
    *,
    min_index: float = 1.0,
    max_loss_ratio: float = 1.0,
    max_gain_ratio: float = MAX_GAIN_RATIO,
) -> list[ModeT]:
    """Keep the Modes that propagate, dropping evanescent and spurious ones.

    A Mode is kept when it is guided (``Re(n_eff)`` above *min_index*),
    loses no more than *max_loss_ratio* per unit length than it advances
    in phase, and does not grow: ``Im(n_eff)`` positive beyond
    *max_gain_ratio* is gain, which a passive line cannot have.

    Args:
        modes: Every solved Mode.
        min_index: Lower bound on ``Re(n_eff)``; Modes at or below it are
            not guided by the line (default: the vacuum light line).
        max_loss_ratio: Upper bound on ``-Im(n_eff) / Re(n_eff)``; a
            Mode losing more than this per unit length than it advances
            in phase is not propagating (default: one, where the two are
            equal).
        max_gain_ratio: Upper bound on ``+Im(n_eff) / Re(n_eff)``, the
            room left for a lossless Mode's numerical noise (default:
            :data:`MAX_GAIN_RATIO`).

    Returns:
        The propagating Modes, in the order they were solved.
    """
    kept = []
    for mode in modes:
        n_eff = mode_index(mode)
        if (
            n_eff.real > min_index
            and -max_loss_ratio * n_eff.real < n_eff.imag <= max_gain_ratio * n_eff.real
        ):
            kept.append(mode)
    return kept


def _describe(modes: Sequence[Any]) -> str:
    """One-line summary of the effective indices that were solved."""
    if not modes:
        return "no modes were solved"
    indices = ", ".join(f"{mode_index(m):.4g}" for m in modes)
    return f"{len(modes)} mode(s) solved with n_eff = {indices}"


def select_line_mode[ModeT](
    modes: Sequence[ModeT],
    *,
    rule: LineModeRule | None = None,
    min_index: float = 1.0,
    max_loss_ratio: float = 1.0,
    degeneracy_rtol: float = 0.03,
) -> ModeT:
    """Select the physical line Mode from a set of solved Modes.

    Args:
        modes: Every Mode the solve returned.
        rule: Candidate rule replacing the default
            :func:`propagating_modes`. It receives every solved Mode and
            returns the physically admissible ones; the slowest of those
            (highest ``Re(n_eff)``) is selected.
        min_index: Lower bound on ``Re(n_eff)`` for the default rule.
        max_loss_ratio: Upper bound on ``|Im(n_eff)| / Re(n_eff)`` for
            the default rule.
        degeneracy_rtol: Relative spread in ``Re(n_eff)`` within which two
            candidates count as ambiguous and a warning is issued.

    Returns:
        The selected Mode, as handed in.

    Raises:
        NoLineModeError: When no candidate survives the rule, naming the
            effective indices that were solved.

    Warns:
        UserWarning: When two or more candidates sit within
            ``degeneracy_rtol`` of the selected effective index, so the
            choice between them is not physically meaningful.
    """
    candidates: Sequence[ModeT] = (
        propagating_modes(modes, min_index=min_index, max_loss_ratio=max_loss_ratio)
        if rule is None
        else list(rule(modes))
    )
    if not candidates:
        raise NoLineModeError(
            f"No propagating line mode among the solved modes: {_describe(modes)}. "
            "Solve more modes, move the eigenvalue guess (n_guess) toward the "
            "expected line index, or pass a rule= selecting the mode yourself."
        )

    selected = max(candidates, key=lambda mode: mode_index(mode).real)
    selected_index = mode_index(selected).real
    if selected_index != 0.0:
        degenerate = [
            mode
            for mode in candidates
            if mode is not selected
            and abs(mode_index(mode).real - selected_index) / abs(selected_index)
            <= degeneracy_rtol
        ]
        if degenerate:
            warnings.warn(
                f"Line mode selection is degenerate: n_eff = "
                f"{mode_index(selected):.6g} was selected but "
                f"{', '.join(f'{mode_index(m):.6g}' for m in degenerate)} "
                f"sit within {degeneracy_rtol:.1%} of it. Inspect the mode "
                "fields and pass rule= to select explicitly.",
                stacklevel=2,
            )
    return selected
