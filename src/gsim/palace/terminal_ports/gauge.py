"""Gauge alignment of the 2D modes with the 3D port modes.

:func:`align_gauge` pairs the 2D modes with the 3D port modes by ``n_eff`` and puts
``T_V`` and ``T_I`` into the gauge of the 3D modes, by the pilot-terminal rule or by
a fit to the voltages of Palace's own saved port modes, which
:func:`port_mode_reference_voltages` reads off a driven run. The aligned matrices
are interpolated onto the frequency grid of the S-parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from gsim.common.modes import Extent
from gsim.palace._validation_port_modes import FACE_MODE_RTOL, PORT_AGREEMENT_RTOL
from gsim.palace.mode_fields import (
    VOLTAGE_NSAMPLES,
    load_port_mode_fields,
    voltage_line_integral,
)
from gsim.palace.ports.modes import read_port_modes
from gsim.palace.terminal import degenerate_mode_groups
from gsim.palace.terminal_ports.modal import _k0
from gsim.palace.terminal_ports.modes2d import _UM, PortModes2D, _pilot_rows
from gsim.palace.terminal_ports.spec import Point, TerminalPortSpec, terminal_paths

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from gsim.palace.ports.modes import PortModeEntry


#: How far a path sample may sit outside the mesh of a saved 3D port face (um)
#: and still be evaluated: Palace writes single-precision coordinates, and a
#: terminal path starts a hair inside a gap.
PORT_FIELD_TOL_UM: float = 1e-3


#: Largest relative difference of ``n_eff`` between a 2D mode and a 3D port mode
#: for the two to be paired. Modes of one face closer than this are reported
#: as degenerate, because the pairing cannot tell them apart.
PAIRING_RTOL: float = PORT_AGREEMENT_RTOL


def port_mode_reference_voltages(
    source: str | Path | dict,
    spec: TerminalPortSpec,
    entries: Sequence[PortModeEntry] | str | Path,
    *,
    cycles: Sequence[int],
    domain: Extent | None = None,
    paths: dict[str, Sequence[Point]] | None = None,
    excitation: int = 1,
    nsamples: int = VOLTAGE_NSAMPLES,
    tol_um: float = PORT_FIELD_TOL_UM,
) -> list[NDArray[np.complex128]]:
    """Terminal voltages of Palace's own port modes, as ``reference_voltages``.

    A driven run that saves fields (``set_driven(save_fields_at=...)``) writes,
    for every ``WavePort`` entry, the unit-power field of its port mode
    (``E0_<index>``, see :func:`gsim.palace.mode_fields.load_port_mode_fields`).
    This integrates those fields along the terminal paths, one ``(N, N)``
    matrix per saved cycle, column ``k`` for mode ``k`` of the port in
    ascending mode number: what :func:`align_gauge` (and
    :func:`terminal_s`) take as ``reference_voltages`` to fit the 2D modes to the
    3D ones instead of assuming the pilot-terminal gauge.

    Palace writes one cycle per saved frequency, in ascending order, so
    ``cycles`` is the position (1-based) of each 2D solve's frequency among the
    saved ones, in the order of the 2D solves handed to :func:`align_gauge`.
    Use the same paths as for the 2D matrices.

    Args:
        source: The run that saved the fields (results dict or directory); any
            run with the port's entries will do, the port modes do not depend on
            the excitation.
        spec: The port's conductors; its name is the name of the port in the
            sidecar.
        entries: The sidecar ``port_modes.json`` (file or folder), or its
            entries as read by
            :func:`gsim.palace.ports.modes.read_port_modes`.
        cycles: ParaView cycle of each reference, 1-based.
        domain: The meshed domain (um), to draw the paths. Not needed when
            ``paths`` is given.
        paths: Voltage paths per terminal name (um), overriding ``domain``.
        excitation: Excitation whose output is read.
        nsamples: Sample points per path segment.
        tol_um: How far outside the face mesh a path sample may sit (um).

    Returns:
        One ``(N, N)`` complex matrix of volts per cycle.

    Raises:
        ValueError: When the sidecar has not one entry per terminal for the
            port, the face has no attribute, no cycle is given, neither
            ``domain`` nor ``paths`` is given or a terminal has no path, or the
            saved fields cannot be read (see
            :func:`gsim.palace.mode_fields.port_mode_fields_from_grid`).
    """
    rows = (
        read_port_modes(entries) if isinstance(entries, str | Path) else tuple(entries)
    )
    face = sorted(
        (row for row in rows if row.port == spec.name), key=lambda row: row.mode
    )
    count = len(spec.terminals)
    if len(face) != count:
        raise ValueError(
            f"Port {spec.name!r} has {len(face)} entries in the sidecar but "
            f"{count} terminals; the reference needs one saved mode per terminal."
        )
    attributes = sorted({a for row in face for a in row.attributes})
    if not attributes:
        raise ValueError(
            f"The entries of port {spec.name!r} list no boundary attribute."
        )
    if not cycles:
        raise ValueError("cycles is empty: give the ParaView cycle of each 2D solve.")
    if paths is None:
        if domain is None:
            raise ValueError("Pass domain= (to draw the paths) or paths=.")
        paths = dict(terminal_paths(spec, domain))
    absent = [n for n in spec.terminal_names if n not in paths]
    if absent:
        raise ValueError(f"No voltage path was given for the terminals {absent}.")

    references: list[NDArray[np.complex128]] = []
    for cycle in cycles:
        fields = load_port_mode_fields(
            source,
            indices=[row.index for row in face],
            attributes=attributes,
            cycle=int(cycle),
            excitation=excitation,
        )
        volts = np.empty((count, count), dtype=np.complex128)
        for k, field in enumerate(fields):
            for i, name in enumerate(spec.terminal_names):
                volts[i, k] = voltage_line_integral(
                    field, paths[name], nsamples=nsamples, tol_um=tol_um
                )
        references.append(volts * _UM)
    return references


@dataclass(frozen=True, eq=False)
class AlignedPortModes:
    """``T_V`` and ``T_I`` of one port in the gauge of its 3D port modes.

    Attributes:
        freq_ghz: Frequencies of the target grid in GHz.
        t_v: ``(nf, N, N)`` terminal voltages, column ``k`` for the 3D mode
            ``k`` of the port (ascending Palace mode number).
        t_i: ``(nf, N, N)`` terminal currents, same layout.
        pilot: The terminal row that fixes the phase of each mode (empty when
            the gauge came from ``reference_voltages``).
        pairing: For each 2D solve (in ascending frequency), the 2D mode
            index paired with each 3D mode.
        mismatch: Largest relative difference of ``n_eff`` over all pairs.
        degenerate: Groups of 3D modes (0-based positions) that the pairing
            could not tell apart at some 2D frequency.
        gauge_residual: ``(n_2d, N)`` relative residual of the fit to
            ``reference_voltages`` per mode, ``None`` without them.
        gauge_scale: ``(n_2d, N)`` magnitude of the fitted gauge factor per mode
            (1 when the 2D and 3D modes are normalised alike), ``None`` without
            ``reference_voltages``.
    """

    freq_ghz: NDArray[np.float64]
    t_v: NDArray[np.complex128]
    t_i: NDArray[np.complex128]
    pilot: tuple[int, ...]
    pairing: tuple[tuple[int, ...], ...]
    mismatch: float
    degenerate: tuple[tuple[int, ...], ...]
    gauge_residual: NDArray[np.float64] | None = None
    gauge_scale: NDArray[np.float64] | None = None


def _check_grid(freq_ghz: ArrayLike) -> NDArray[np.float64]:
    """A strictly increasing, positive, finite 1-D frequency grid."""
    freq = np.asarray(freq_ghz, dtype=float)
    if (
        freq.ndim != 1
        or freq.size == 0
        or not np.all(np.isfinite(freq))
        or np.any(freq <= 0.0)
        or np.any(np.diff(freq) <= 0.0)
    ):
        raise ValueError(
            "freq_ghz must be a strictly increasing 1-D array of positive "
            "finite frequencies."
        )
    return freq


def _fit_gauge(
    t_v: NDArray[np.complex128],
    t_i: NDArray[np.complex128],
    reference: NDArray[np.complex128],
    groups: Sequence[tuple[int, ...]],
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Fit ``T_V`` to the voltages of the 3D port modes, mode by mode.

    For every mode (or group of indistinguishable modes) the least-squares
    factor (matrix) ``M`` with ``reference ~ T_V M``. ``T_I`` gets the same
    ``M``, so voltage and current keep their shape and take the 3D gauge.

    Returns:
        ``T_V M``, ``T_I M``, the relative residual per mode and the column
        norm of ``M`` per mode.

    Raises:
        ValueError: When a reference mode has no voltage at all.
    """
    count = t_v.shape[0]
    fit = np.zeros((count, count), dtype=np.complex128)
    residual = np.zeros(count)
    for group in groups:
        cols = list(group)
        v2 = t_v[:, cols]
        v3 = reference[:, cols]
        norms = np.linalg.norm(v3, axis=0)
        if np.any(norms == 0.0):
            raise ValueError("A reference mode has zero terminal voltage.")
        sol, *_ = np.linalg.lstsq(v2, v3, rcond=None)
        fit[np.ix_(cols, cols)] = sol
        residual[cols] = np.linalg.norm(v3 - v2 @ sol, axis=0) / norms
    return (
        np.asarray(t_v @ fit, dtype=np.complex128),
        np.asarray(t_i @ fit, dtype=np.complex128),
        residual,
        np.asarray(np.linalg.norm(fit, axis=0), dtype=float),
    )


def align_gauge(
    modes_2d: Sequence[PortModes2D],
    *,
    freq_ghz: ArrayLike,
    wave_numbers: ArrayLike,
    terminal_names: Sequence[str] = (),
    pilot: Sequence[int | str] | None = None,
    reference_voltages: Sequence[ArrayLike] | None = None,
    rtol: float = PAIRING_RTOL,
    degenerate_rtol: float = FACE_MODE_RTOL,
    allow_degenerate: bool = False,
    extrapolate: bool = False,
    interpolation: str = "inverse",
) -> AlignedPortModes:
    """Bring the 2D ``T_V`` and ``T_I`` of a port into the gauge of its 3D modes.

    The modal S-parameters from Palace and the terminal matrices must describe
    the same modes with the same scale and phase at every port; a scaling that
    differs from port to port changes the terminal S. This does three things:

    1. **Pairing.** At every 2D frequency the 3D modes (the wave numbers
       Palace printed, interpolated to that frequency) are paired with the 2D
       modes by ``n_eff`` (an optimal assignment), so a swap of the mode order
       with frequency does no harm. The pair must agree within ``rtol``.
    2. **Magnitude.** The 2D columns already have ``|2 P| = 1``
       (:func:`port_terminal_matrices`), Palace's unit-power convention.
    3. **Phase and sign.** Every column is rotated so that the voltage on its
       *pilot* terminal is real and positive. Palace's mode phase is set by a
       fixed rule and its sign so that the voltage along the port's
       ``VoltagePath`` is positive; for a low-loss quasi-TEM mode the two
       agree when the 3D pilot path is the 2D pilot terminal's path. That is
       an assumption, checked only by a comparison with a run (see the
       guide); where it is doubtful pass ``reference_voltages`` instead.

    With ``reference_voltages`` (the terminal voltages of Palace's own saved
    port modes, one ``(N, N)`` matrix per 2D solve, column ``k`` for the 3D mode
    ``k``) the gauge comes from the 3D fields: every mode (a degenerate group as
    a block) is fitted ``V_3D ~ V_2D M`` and ``T_V``, ``T_I`` are multiplied by
    ``M``. The fit residual measures the agreement of the 2D and 3D modes.

    The aligned matrices are then interpolated (real and imaginary parts, linearly
    in the coordinate ``interpolation`` names) from the 2D frequencies to
    ``freq_ghz``. The default is ``1 / f``: the modes of a lossy line depart
    from those of the lossless one by terms like ``R / (w L)`` and
    ``G / (w C)``, so ``T_V`` varies most at low frequency and is nearly flat
    above. On a synthetic lossy line (1-100 GHz, 5 solves) the largest ``T_V``
    error was 1e-4 with ``1 / f``, 1.7e-2 with ``"log"`` and 4.8e-2 with
    ``"linear"``. A real line also has dispersion and skin effect, so check the
    choice with a 2D solve at a frequency not used for the interpolation.

    Args:
        modes_2d: The 2D solves of the port, at one or more frequencies.
        freq_ghz: Target frequencies in GHz, strictly increasing.
        wave_numbers: ``(nf, N)`` complex 3D wave numbers (1/m) of the port's
            modes at ``freq_ghz``, column ``k`` for mode ``k``
            (:attr:`ModalS.wave_numbers` for the port's columns).
        terminal_names: Names of the terminals, to give ``pilot`` as names.
        pilot: Per 3D mode, the terminal (index or name) that sets the phase.
            Default: the terminal with the largest voltage in the lowest 2D
            frequency's paired mode, kept for all frequencies. Use the same
            terminal for the 3D ``VoltagePath`` of that mode.
        reference_voltages: See above; same length and order as ``modes_2d``.
        rtol: Largest relative ``n_eff`` difference of a 2D mode and the 3D mode
            it is paired with. It may be opened (for example to 0.15) to pair
            2D modes that differ from the 3D ones by more than the default,
            which says nothing about whether two 3D modes can be told apart.
        degenerate_rtol: Distance under which two 3D modes count as
            indistinguishable, so that their 2D basis need not be Palace's
            (default: the face-mode tolerance of the port-mode checks, 0.5 %).
            Separate from ``rtol`` on purpose: opening ``rtol`` does not
            declare modes a few percent apart degenerate.
        allow_degenerate: Accept indistinguishable 3D modes without
            ``reference_voltages``. Their 2D basis then need not be the 3D
            one and the terminal S can be wrong; a ``ValueError`` is the default.
        extrapolate: Hold the end values for target frequencies outside the
            range of the 2D solves (otherwise that is an error).
        interpolation: ``"inverse"`` (``1 / f``, default), ``"linear"`` or ``"log"``.

    Returns:
        The aligned matrices on the target grid, and the audit numbers.

    Raises:
        ValueError: For an unknown ``interpolation``, malformed input,
            missing wave numbers, a 2D solve
            outside the 3D wave-number range or target frequencies outside
            the 2D range without ``extrapolate``, a pair of modes that do not
            agree within ``rtol``, a negative or non-finite ``degenerate_rtol``,
            indistinguishable modes (within ``degenerate_rtol``) without a way
            to fix their basis, or a zero pilot voltage.
    """
    from scipy.optimize import linear_sum_assignment

    _axis(np.ones(1), interpolation)  # refuse an unknown name before any work
    if not (np.isfinite(degenerate_rtol) and degenerate_rtol >= 0.0):
        raise ValueError(
            f"degenerate_rtol must be a finite number >= 0, got {degenerate_rtol}."
        )
    freq = _check_grid(freq_ghz)
    kn = np.asarray(wave_numbers, dtype=np.complex128)
    if not modes_2d:
        raise ValueError("align_gauge needs at least one 2D solve.")
    count = modes_2d[0].n_eff.size
    if kn.shape != (freq.size, count):
        raise ValueError(
            f"wave_numbers has shape {kn.shape}, expected {(freq.size, count)}: one "
            "column per mode of the port."
        )
    if not np.all(np.isfinite(kn)):
        missing = sorted({float(f) for f in freq[~np.all(np.isfinite(kn), axis=1)]})
        raise ValueError(
            f"The 3D wave numbers are missing at {missing} GHz: read the run's log "
            "(load_modal_run(..., log=...)) for every frequency."
        )
    n_3d = kn.real / _k0(freq)[:, None]
    if np.any(n_3d <= 0.0):
        raise ValueError("A 3D mode has a non-positive effective index.")
    if reference_voltages is not None and len(reference_voltages) != len(modes_2d):
        raise ValueError(
            f"reference_voltages has {len(reference_voltages)} entries for "
            f"{len(modes_2d)} 2D solves."
        )
    refs = list(reference_voltages) if reference_voltages is not None else None
    ranked = sorted(range(len(modes_2d)), key=lambda p: modes_2d[p].freq_ghz)
    solves = [modes_2d[p] for p in ranked]
    f_2d = np.array([m.freq_ghz for m in solves])
    if np.any(np.diff(f_2d) <= 0.0):
        raise ValueError("Two 2D solves have the same frequency.")
    if any(m.n_eff.size != count for m in solves):
        raise ValueError("The 2D solves do not all have the same number of modes.")
    outside = freq[0] < f_2d[0] * (1 - 1e-9) or freq[-1] > f_2d[-1] * (1 + 1e-9)
    if outside and not extrapolate:
        raise ValueError(
            f"The 2D solves cover {f_2d[0]:g} to {f_2d[-1]:g} GHz but the S-parameters "
            f"go from {freq[0]:g} to {freq[-1]:g} GHz; add 2D solves, or pass "
            "extrapolate=True to hold the end values."
        )
    beyond = [
        float(f) for f in f_2d if f < freq[0] * (1 - 1e-9) or f > freq[-1] * (1 + 1e-9)
    ]
    if beyond:
        raise ValueError(
            f"The 2D solves at {beyond} GHz lie outside the 3D wave-number range "
            f"{freq[0]:g} to {freq[-1]:g} GHz, where the 3D n_eff they are paired "
            "with is not known (an interpolation would silently hold the end "
            "value): drop those solves, or extend freq_ghz and wave_numbers."
        )

    pairing: list[tuple[int, ...]] = []
    degenerate: set[tuple[int, ...]] = set()
    mismatch = 0.0
    aligned_v: list[NDArray[np.complex128]] = []
    aligned_i: list[NDArray[np.complex128]] = []
    residuals: list[NDArray[np.float64]] = []
    scales: list[NDArray[np.float64]] = []
    rows: tuple[int, ...] | None = None
    for position, solve in enumerate(solves):
        n_at = np.array(
            [np.interp(solve.freq_ghz, freq, n_3d[:, k]) for k in range(count)]
        )
        cost = np.abs(n_at[:, None] - solve.n_eff[None, :]) / n_at[:, None]
        three_d, two_d = linear_sum_assignment(cost)
        worst = float(cost[three_d, two_d].max())
        if worst > rtol:
            bad = [int(k) + 1 for k in three_d[cost[three_d, two_d] > rtol]]
            raise ValueError(
                f"At {solve.freq_ghz:g} GHz the 2D modes cannot be paired with the 3D "
                f"modes {bad}: the closest n_eff differs by {worst:.2%} "
                f"(limit {rtol:.2%}). 3D n_eff {np.round(n_at, 4).tolist()}, "
                f"2D n_eff {np.round(solve.n_eff, 4).tolist()}."
            )
        mismatch = max(mismatch, worst)
        perm = tuple(int(c) for c in two_d)
        pairing.append(perm)
        groups = degenerate_mode_groups(n_at, rtol=degenerate_rtol)
        indistinct = [g for g in groups if len(g) > 1]
        degenerate.update(indistinct)
        t_v = solve.t_v[:, perm]
        t_i = solve.t_i[:, perm]
        if refs is not None:
            reference = np.asarray(refs[ranked[position]], dtype=np.complex128)
            if reference.shape != (count, count):
                raise ValueError(
                    f"reference_voltages[{ranked[position]}] has shape "
                    f"{reference.shape}, expected {(count, count)}."
                )
            t_v, t_i, residual, scale = _fit_gauge(t_v, t_i, reference, groups)
            residuals.append(residual)
            scales.append(scale)
        else:
            if indistinct and not allow_degenerate:
                raise ValueError(
                    f"At {solve.freq_ghz:g} GHz the 3D modes "
                    f"{[[k + 1 for k in g] for g in indistinct]} have the same n_eff "
                    f"within {degenerate_rtol:.2%}: their 2D basis need not be "
                    "Palace's, which "
                    "changes the terminal S. Pass reference_voltages= (from the "
                    "saved port-mode fields) or allow_degenerate=True."
                )
            if rows is None:
                rows = _pilot_rows(t_v, terminal_names, pilot)
            t_v = t_v.copy()
            t_i = t_i.copy()
            for k in range(count):
                voltage = t_v[rows[k], k]
                if abs(voltage) == 0.0:
                    raise ValueError(
                        f"Mode {k + 1} has zero voltage on its pilot terminal at "
                        f"{solve.freq_ghz:g} GHz."
                    )
                phase = np.conj(voltage) / abs(voltage)
                t_v[:, k] *= phase
                t_i[:, k] *= phase
        aligned_v.append(t_v)
        aligned_i.append(t_i)

    return AlignedPortModes(
        freq_ghz=freq,
        t_v=_interpolate_matrices(f_2d, aligned_v, freq, interpolation),
        t_i=_interpolate_matrices(f_2d, aligned_i, freq, interpolation),
        pilot=() if rows is None else rows,
        pairing=tuple(pairing),
        mismatch=mismatch,
        degenerate=tuple(sorted(degenerate)),
        gauge_residual=np.array(residuals) if residuals else None,
        gauge_scale=np.array(scales) if scales else None,
    )


#: The frequency axes :func:`align_gauge` can interpolate the 2D matrices on.
INTERPOLATIONS: tuple[str, ...] = ("inverse", "linear", "log")


def _axis(freq: NDArray[np.float64], interpolation: str) -> NDArray[np.float64]:
    """The increasing coordinate that the matrices are interpolated along."""
    if interpolation == "inverse":
        return np.asarray(-1.0 / freq, dtype=float)
    if interpolation == "linear":
        return freq
    if interpolation == "log":
        return np.asarray(np.log(freq), dtype=float)
    raise ValueError(
        f"interpolation must be one of {list(INTERPOLATIONS)}, got {interpolation!r}."
    )


def _interpolate_matrices(
    f_2d: NDArray[np.float64],
    stack: Sequence[NDArray[np.complex128]],
    freq: NDArray[np.float64],
    interpolation: str,
) -> NDArray[np.complex128]:
    """Interpolate ``(N, N)`` matrices given at ``f_2d`` onto ``freq``.

    Real and imaginary parts are interpolated linearly in the chosen
    coordinate, entry by entry; outside ``f_2d`` the end values are held.
    """
    data = np.asarray(stack, dtype=np.complex128)
    count = data.shape[1]
    x_2d = _axis(f_2d, interpolation)
    x = _axis(freq, interpolation)
    out = np.empty((freq.size, count, count), dtype=np.complex128)
    for i in range(count):
        for j in range(count):
            out[:, i, j] = np.interp(x, x_2d, data[:, i, j].real) + 1j * np.interp(
                x, x_2d, data[:, i, j].imag
            )
    return out
