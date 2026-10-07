"""The characteristic impedance of a Palace boundary Mode, two ways.

A ``BoundaryMode`` solve can be asked, before it runs, to integrate the
voltage across a gap and the current around a conductor on every Mode it
finds; the impedance those integrals imply lands in its own tables. It
can also be asked to save the Modes themselves, and the same impedance
can then be integrated off the saved fields afterwards. Both are here,
with the policy that prefers the first, because they are one story:
size two paths from the conductor layout, declare them on the solve,
read the impedance off Palace's tables under the index the solve
assigned, and fall back to the fields when there is no table.

Splitting the table reader into :mod:`gsim.palace.results` and the field
reader into :mod:`gsim.palace.mode_fields` would push a line-mode
concept — signal against return conductor, the voltage the gap carries,
the Mode that runs to the wall instead — into two modules that know only
about tables and about fields, and would leave the tables-then-fields
policy with no home.

Everything here is stated in the cross-section's own ``(h, v)``
coordinates (um), and every electrode is the rectangle a
:class:`gsim.common.modes.Conductor` names. A Mode arrives as its
``mode_id`` and its ``n_eff`` rather than as a Mode object, because the
only solved-Mode type in gsim sits in a Route and this module must not
reach for it.

Nothing here is re-exported from :mod:`gsim.palace`, for the reason
:mod:`gsim.palace.mode_fields` gives about its integrals: the four
functions are one policy — size, declare, read, fall back — and are
meant to be read together rather than picked out of a package listing.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gsim.common.modes import Conductor, Extent, LineReading
from gsim.palace.mode_fields import check_field_is_the_mode
from gsim.palace.mode_fields import z0_power_current as palace_z0

if TYPE_CHECKING:
    from gsim.palace.boundarymode import BoundaryModeSim
    from gsim.palace.results import PalaceTextResults

__all__ = [
    "IMPEDANCE_PORT",
    "MIN_VOLTAGE_POWER_RATIO",
    "PATH_CLEARANCE_FRACTION",
    "ImpedancePaths",
    "declare_impedance_paths",
    "field_line_impedance",
    "line_impedance_paths",
    "native_line_impedance",
    "palace_line_impedance",
]

#: How far inside the gap a postprocessing path keeps from a conductor
#: face, as a fraction of the smallest clearance around it. The paths
#: are sampled on the meshed domain, and under the ``"pec"`` model a
#: conductor's interior is not in it (ADR 0003); a hair inside the gap
#: is on the domain, and a hair is what the voltage integral misses.
PATH_CLEARANCE_FRACTION: float = 1e-3

#: Name of the postprocessing port the line's paths are declared under.
IMPEDANCE_PORT: str = "line"

#: Below this ``Z_PV / Z_PI`` — which is ``(|V| |I| / 2P)^2`` — the voltage
#: across the gap carries none of the Mode's power, and the Mode is not
#: the line Mode between the two electrodes. A quasi-TEM line Mode has
#: ``|V| |I| ~ 2P``; a lossy one sits within a factor of a few of it; a
#: Mode running between both electrodes together and the Window wall
#: puts the electrodes at one potential and sits many decades under.
MIN_VOLTAGE_POWER_RATIO: float = 1e-2


@dataclass(frozen=True)
class ImpedancePaths:
    """Where Palace integrates the line's voltage and current.

    Palace's ``BoundaryMode`` postprocessing takes two paths per
    impedance entry: an open one along which it integrates ``E`` for
    the mode voltage, and a closed one around which it integrates ``H``
    for the current. Both are in the Cross-section's own ``(h, v)``
    coordinates (um).

    Attributes:
        voltage: Two points, from the signal conductor's face across the
            gap to the return conductor's face.
        current: The corners of a loop enclosing the signal conductor and
            nothing else. Palace joins the last corner back to the first,
            so the first is not repeated.
    """

    voltage: tuple[tuple[float, float], ...]
    current: tuple[tuple[float, float], ...]


def line_impedance_paths(
    *,
    signal: Extent,
    ground: Extent,
    domain: Extent,
    clearance_fraction: float = PATH_CLEARANCE_FRACTION,
) -> ImpedancePaths:
    """Size the impedance postprocessing paths from the electrode layout.

    The voltage path crosses the gap between the two electrodes, from
    the signal electrode's face at its mid-height to the return
    electrode's face at its own, so it is the same path whichever side
    the drive is on. The current loop is the signal electrode's outline
    pushed out by a clearance: tight enough that anything standing
    against the electrode contributes nothing to the enclosed current,
    which is what the Marks-Williams contour of the field-based fallback
    measures too.

    Args:
        signal: ``((h_min, h_max), (v_min, v_max))`` of the signal
            electrode (um).
        ground: The same for the return electrode.
        domain: The same for the meshed domain the paths are sampled on.
        clearance_fraction: Fraction of the smallest clearance around the
            signal electrode — the gap, its thickness, its distance to
            each domain wall — that the paths keep from every face.

    Returns:
        The two paths.

    Raises:
        ValueError: When the electrodes leave no gap to cross along the
            in-plane axis, or the signal electrode is not strictly inside
            the domain (an electrode the domain clips has no outline to
            loop around).
    """
    (s_lo, s_hi), (t_lo, t_hi) = signal
    (g_lo, g_hi), (u_lo, u_hi) = ground
    (d_lo, d_hi), (e_lo, e_hi) = domain

    if not (d_lo < s_lo < s_hi < d_hi and e_lo < t_lo < t_hi < e_hi):
        raise ValueError(
            f"The signal electrode spans {signal} but is not inside the meshed "
            f"domain {domain}, so no path around it can be sampled."
        )
    if g_lo > s_hi:
        gap, faces = g_lo - s_hi, (s_hi, g_lo)
    elif g_hi < s_lo:
        gap, faces = s_lo - g_hi, (s_lo, g_hi)
    else:
        raise ValueError(
            f"The electrodes at {signal[0]} and {ground[0]} leave no gap along "
            "the in-plane axis to integrate the line voltage across."
        )

    clearance = clearance_fraction * min(
        gap, t_hi - t_lo, s_lo - d_lo, d_hi - s_hi, t_lo - e_lo, e_hi - t_hi
    )
    # A hair into the gap from each face, on the domain rather than on
    # (or, under the "pec" model, inside) the conductor.
    inset = clearance if faces[0] < faces[1] else -clearance
    voltage = (
        (faces[0] + inset, 0.5 * (t_lo + t_hi)),
        (faces[1] - inset, 0.5 * (u_lo + u_hi)),
    )
    current = (
        (s_lo - clearance, t_lo - clearance),
        (s_hi + clearance, t_lo - clearance),
        (s_hi + clearance, t_hi + clearance),
        (s_lo - clearance, t_hi + clearance),
    )
    return ImpedancePaths(voltage=voltage, current=current)


def declare_impedance_paths(
    sim: BoundaryModeSim, paths: ImpedancePaths, *, nsamples: int = 100
) -> int:
    """Register the paths as a postprocessing path of the solve.

    A ``BoundaryMode`` path is postprocessing only — it loads nothing
    and leaves the eigenproblem as it was. The simulation says which
    index Palace will report it under, and that index is what
    :func:`native_line_impedance` reads.

    Args:
        sim: The simulation about to be solved; meshed or not, since the
            paths take no part in meshing.
        paths: What :func:`line_impedance_paths` sized.
        nsamples: Quadrature order of each line integral.

    Returns:
        The index Palace reports the line's impedance under.
    """
    return sim.add_impedance_path(
        IMPEDANCE_PORT,
        voltage=[list(point) for point in paths.voltage],
        current=[list(point) for point in paths.current],
        nsamples=nsamples,
    )


def native_line_impedance(
    results: PalaceTextResults, *, index: int, mode_id: int, n_eff: complex
) -> LineReading | None:
    """One Mode's reading off Palace's own ``mode-Z.csv``.

    Palace reports two magnitudes per entry: ``Z_PV = |V|^2 / 2P`` from
    the voltage path alone, and ``Z_VI = |V| / |I|`` once a current path
    is declared. Their ratio ``Z_VI^2 / Z_PV = 2P / |I|^2`` is the
    power-current impedance the field integral computes too, so that is
    what is read; a table carrying ``Z_PV`` alone is a different
    definition, not a nearer answer, and is declined the same as no
    table.

    The voltage cancels out of that ratio, so it is read on its own for
    the wall-Mode diagnostic: a Mode whose gap voltage carries almost
    none of its power is not running between the two electrodes.

    The reading is real-only, because the tables carry magnitudes. It
    sits 0.15% from what :func:`field_line_impedance` reads on the same
    solve (2.83% against femwell, versus 2.98%), and only the field path
    carries the reactance — so a caller wanting a complex ``Z_0`` reads
    the fields, not this.

    Palace also tabulates ``L_PV`` and ``C_PV``, and they are no second
    opinion: they are ``Z_PV`` divided and multiplied by the phase
    velocity, so ``sqrt(L_PV / C_PV)`` returns ``Z_PV`` exactly, and
    they carry no ``R`` and no ``G``. The ``Z_PV``/``Z_PI`` ratio this
    function already computes for the wall-Mode diagnostic is therefore
    the cross-check the tables afford; there is no other one to add.

    Args:
        results: The run's text results.
        index: The postprocessing index the path was declared under.
        mode_id: Palace's own mode number of the Mode being read.
        n_eff: That Mode's effective index, carried into the reading.

    Returns:
        The reading — its impedance real — or ``None`` when the tables
        are absent, carry no current path, no current (a loop enclosing
        none), or not this Mode.
    """
    z_pv = results.characteristic_impedance(index=index, mode=mode_id, quantity="Z_PV")
    z_vi = results.characteristic_impedance(index=index, mode=mode_id, quantity="Z_VI")
    if z_pv is None or z_vi is None or z_pv <= 0.0 or z_vi <= 0.0:
        return None
    z_pi = z_vi * z_vi / z_pv
    wall_mode = z_pv < MIN_VOLTAGE_POWER_RATIO * z_pi
    if wall_mode:
        diagnostic = (
            f"whose voltage across the gap carries {z_pv / z_pi:.2g} of the "
            "power its current implies: the two electrodes sit at one "
            "potential, so this is a mode between them and the window wall "
            "rather than the line mode between them"
        )
    else:
        diagnostic = (
            f"whose voltage across the gap carries {z_pv / z_pi:.2g} of the "
            "power its current implies: the line mode between the electrodes"
        )
    return LineReading(
        n_eff=complex(n_eff),
        z0_ohm=complex(z_pi),
        wall_mode=wall_mode,
        diagnostic=diagnostic,
    )


def field_line_impedance(
    sim: BoundaryModeSim,
    *,
    mode_id: int,
    n_eff: complex,
    h_span: tuple[float, float],
    v_span: tuple[float, float],
    context: str,
) -> complex:
    """Characteristic impedance of a Palace Mode, off its saved fields.

    The Marks-Williams power-current integral, run on the fields Palace
    wrote for one Mode: the complex Poynting flux over the whole
    Cross-section, over the current Ampere's law reads around the signal
    conductor.

    Reading fields back is the one part of a Palace solve that depends
    on a file the solver may not have written, so a missing or
    unreadable one is reported as NaN with the reason rather than
    raising in the middle of a frequency sweep.

    Args:
        sim: The simulation that was run, which reads its saved fields
            back.
        mode_id: Palace's own mode number, which names the saved fields.
        n_eff: The index that Mode's table reports, which the saved
            fields are checked against.
        h_span: ``(min, max)`` of the signal conductor along the
            Cross-section's in-plane axis (um).
        v_span: ``(min, max)`` along its vertical axis (um).
        context: Who is asking, opening the warnings.

    Returns:
        The complex characteristic impedance in ohms, or NaN when the
        fields could not be read.
    """
    try:
        field = sim.read_mode_field(mode_id)
        z0 = palace_z0(field, h_span=h_span, v_span=v_span)
    except Exception as err:
        warnings.warn(
            f"{context} could not read mode {mode_id}'s saved fields, so its "
            f"characteristic impedance comes back NaN: {err} Palace's output "
            f"is in {sim.output_dir}.",
            stacklevel=2,
        )
        return complex(math.nan, math.nan)
    # Outside the guard above: the check warns rather than raises, and a
    # caller running with warnings as errors must see the wrong-Mode
    # diagnostic rather than have it caught here and reported as an
    # unreadable file.
    check_field_is_the_mode(field, mode_id=mode_id, n_eff=n_eff, context=context)
    return z0


def palace_line_impedance(
    sim: BoundaryModeSim,
    results: PalaceTextResults,
    *,
    index: int | None,
    mode_id: int,
    n_eff: complex,
    signal: Conductor,
    context: str,
) -> LineReading:
    """One Mode's reading: tables first, saved fields as fallback.

    Palace's own answer first: the power-current impedance its
    postprocessing paths measured (:func:`native_line_impedance`), read
    off this run's tables under the index the path was declared at, with
    the wall-Mode diagnostic the gap voltage gives. A run without the
    tables — a solve that declared no path, or an older Palace — falls
    back to the saved fields (:func:`field_line_impedance`), whose NaN
    contract stands and which say nothing about which Mode this is.

    Args:
        sim: The simulation that was run.
        results: That run's text results.
        index: The postprocessing index the impedance path was declared
            under, or ``None`` when none could be declared.
        mode_id: Palace's own mode number of the Mode being read.
        n_eff: That Mode's effective index.
        signal: The signal electrode, whose outline the fallback
            integrates around.
        context: Who is asking, opening the fallback's warning.

    Returns:
        The reading: impedance real off the tables, complex off the
        fields, NaN when neither could be read.
    """
    if index is not None:
        native = native_line_impedance(
            results, index=index, mode_id=mode_id, n_eff=n_eff
        )
        if native is not None:
            return native
    h_span, v_span = signal.extent
    z0 = field_line_impedance(
        sim,
        mode_id=mode_id,
        n_eff=n_eff,
        h_span=h_span,
        v_span=v_span,
        context=context,
    )
    return LineReading(
        n_eff=complex(n_eff),
        z0_ohm=z0,
        wall_mode=None,
        diagnostic="the impedance was read off the saved fields, which carry "
        "no gap voltage to compare against the power",
    )
