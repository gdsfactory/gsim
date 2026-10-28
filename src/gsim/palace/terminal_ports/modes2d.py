"""``T_V`` and ``T_I`` of a port's modes from their saved 2D fields.

:func:`port_terminal_matrices` integrates the saved mode fields along the voltage
paths (``T_V``) and around the terminals or through the reaction matrix (``T_I``);
:func:`port_modes_2d` adds the effective indices of the modes. The conventions
(sign, scale and phase of the columns) are listed in the package docstring of
:mod:`gsim.palace.terminal_ports`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from gsim.common.modes import Extent
from gsim.palace.mode_fields import (
    DEFAULT_PERIMETER_TOL_UM,
    VOLTAGE_NSAMPLES,
    contour_current,
    power_flux,
    reaction_matrix,
    voltage_line_integral,
)
from gsim.palace.terminal import currents_from_reaction
from gsim.palace.terminal_ports.spec import Point, TerminalPortSpec, terminal_paths

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import ArrayLike, NDArray

    from gsim.palace.mode_fields import BoundaryModeField


#: ``cond(T_V)`` above which the modes are taken not to span the terminals;
#: the same bound :func:`gsim.palace.terminal.modal_to_terminal_s` applies.
MAX_COND: float = 1e12


#: Palace's mode fields are in um coordinates against SI field values, so a
#: voltage or current integrated over um comes back a factor 1e6 too large and
#: a power integrated over um^2 a factor 1e12 too large.
_UM: float = 1e-6


@dataclass(frozen=True, eq=False)
class PortTerminalMatrices:
    """Terminal voltages and currents of one port's modes.

    Attributes:
        t_v: ``(N, N)`` terminal voltages in volts, column ``k`` for mode ``k``
            (row ``i`` is terminal ``i``), normalised to ``|2 P| = 1`` with the
            pilot voltage real and positive.
        t_i: ``(N, N)`` terminal currents into the network in amperes, same
            layout, same scale and phase as ``t_v``.
        pilot: Terminal row used as the phase reference of each mode.
        power: ``|2 P|`` of each saved mode before normalisation, in watts for
            the amplitude the field was saved with.
        cond: Condition number of ``t_v``.
    """

    t_v: NDArray[np.complex128]
    t_i: NDArray[np.complex128]
    pilot: tuple[int, ...]
    power: NDArray[np.float64]
    cond: float


def _pilot_rows(
    t_v: NDArray[np.complex128],
    names: Sequence[str],
    pilot: Sequence[int | str] | None,
) -> tuple[int, ...]:
    """The terminal each mode's phase is referred to.

    Default: the terminal with the largest voltage magnitude, the lowest index
    among those within 1e-6 of it so that a symmetric mode picks the same
    terminal on every run.
    """
    count = t_v.shape[0]
    if pilot is None:
        rows = []
        for k in range(count):
            magnitude = np.abs(t_v[:, k])
            near_max = np.flatnonzero(magnitude >= (1.0 - 1e-6) * magnitude.max())
            rows.append(int(near_max[0]))
        return tuple(rows)
    if len(pilot) != count:
        raise ValueError(f"pilot names {len(pilot)} terminals for {count} modes.")
    rows = []
    for entry in pilot:
        if isinstance(entry, str):
            if entry not in names:
                known = list(names)
                raise ValueError(f"pilot terminal {entry!r} is not one of {known}.")
            rows.append(names.index(entry))
        elif 0 <= int(entry) < count:
            rows.append(int(entry))
        else:
            raise ValueError(f"pilot terminal index {entry} is outside 0..{count - 1}.")
    return tuple(rows)


def port_terminal_matrices(
    fields: Sequence[BoundaryModeField],
    spec: TerminalPortSpec,
    *,
    domain: Extent | None = None,
    paths: dict[str, Sequence[Point]] | None = None,
    current: Literal["contour", "reaction"] = "contour",
    pilot: Sequence[int | str] | None = None,
    nsamples: int = VOLTAGE_NSAMPLES,
    tol_um: float = DEFAULT_PERIMETER_TOL_UM,
) -> PortTerminalMatrices:
    """``T_V`` and ``T_I`` of a port's modes from their saved 2D fields.

    ``T_V[i, k]`` is the line integral of ``E_t`` of mode ``k`` along terminal
    ``i``'s path. ``T_I`` comes one of two ways:

    * ``"contour"``: Ampere's contour integral of ``H`` around each terminal's
      rectangle, :func:`gsim.palace.mode_fields.contour_current`;
    * ``"reaction"``: from the unconjugated reaction of the modes,
      ``T_I = T_V^-T R`` with ``R_jk = integral (E_j x H_k) . z dA``
      (:func:`gsim.palace.terminal.currents_from_reaction`), which needs no
      conductor outline on the mesh and covers a degenerate group in any basis.

    Each column is then scaled to ``|2 P| = 1`` (``P`` the complex power flux of
    the mode, in SI), oriented so that the incident wave carries power into the
    network (``Re(T_V^H T_I) > 0`` per column, by flipping the sign of ``T_I``),
    and rotated so that its pilot terminal's voltage is real and positive.

    Args:
        fields: The saved fields of the modes, one per terminal, all on the same
            mesh. Mode ``k`` becomes column ``k``.
        spec: The port's conductors.
        domain: The meshed domain (um), used to draw the paths. Not needed when
            ``paths`` is given.
        paths: Voltage paths per terminal name in the cross-section's
            coordinates (um), for example the ones declared on the boundary-mode
            solve. Overrides ``domain``.
        current: How to obtain ``T_I``.
        pilot: Per mode, the terminal (index or name) whose voltage sets the
            phase. ``None`` takes the terminal with the largest voltage.
        nsamples: Sample points per path segment for the voltage integrals
            (default 800: at 200 the integral along a path that passes close to
            a metal corner is off by about 1 %).
        tol_um: How far outside the mesh a path sample may sit (um).

    Returns:
        The matrices and the numbers needed to audit them.

    Raises:
        ValueError: When the number of modes differs from the number of
            terminals, ``current`` is not one of the two routes, neither
            ``domain`` nor ``paths`` is given or a terminal has no path, a mode
            carries no power or no real power into the port (so it cannot be
            oriented), the pilot voltage is zero, or the modes do not span the
            terminals (``cond(T_V) > MAX_COND``).
    """
    modes = list(fields)
    count = len(spec.terminals)
    if len(modes) != count:
        raise ValueError(
            f"Port {spec.name!r} has {count} terminals and needs one saved mode "
            f"per terminal, but {len(modes)} were given."
        )
    if current not in ("contour", "reaction"):
        raise ValueError(f"current must be 'contour' or 'reaction', got {current!r}.")
    if paths is None:
        if domain is None:
            raise ValueError("Pass domain= (to draw the paths) or paths=.")
        paths = dict(terminal_paths(spec, domain))
    absent = [n for n in spec.terminal_names if n not in paths]
    if absent:
        raise ValueError(f"No voltage path was given for the terminals {absent}.")

    t_v = np.empty((count, count), dtype=np.complex128)
    for k, mode in enumerate(modes):
        for i, name in enumerate(spec.terminal_names):
            t_v[i, k] = voltage_line_integral(
                mode, paths[name], nsamples=nsamples, tol_um=tol_um
            )
    t_v *= _UM

    power = np.array([abs(2.0 * power_flux(mode)) for mode in modes]) * _UM**2
    if not np.all(np.isfinite(power)) or np.any(power <= 0.0):
        bad = [k + 1 for k in range(count) if not power[k] > 0.0]
        raise ValueError(f"Mode(s) {bad} of port {spec.name!r} carry no power.")
    scale = 1.0 / np.sqrt(power)
    t_v = t_v * scale[None, :]
    cond = float(np.linalg.cond(t_v))
    if not np.isfinite(cond) or cond > MAX_COND:
        raise ValueError(
            f"T_V of port {spec.name!r} is ill-conditioned (cond = {cond:.1e}): the "
            "modes do not span the terminals."
        )

    if current == "contour":
        t_i = np.empty((count, count), dtype=np.complex128)
        for k, mode in enumerate(modes):
            for i, terminal in enumerate(spec.terminals):
                t_i[i, k] = contour_current(
                    mode,
                    h_span=terminal.extent[0],
                    v_span=terminal.extent[1],
                    tol_um=tol_um,
                )
        t_i = t_i * _UM * scale[None, :]
    else:
        reaction = reaction_matrix(modes) * _UM**2 * np.outer(scale, scale)
        t_i = np.asarray(currents_from_reaction(t_v, reaction), dtype=np.complex128)

    rows = _pilot_rows(t_v, spec.terminal_names, pilot)
    for k in range(count):
        real_power = float(np.real(np.vdot(t_v[:, k], t_i[:, k])))
        apparent = float(np.linalg.norm(t_v[:, k]) * np.linalg.norm(t_i[:, k]))
        if abs(real_power) <= 1e-9 * apparent:
            raise ValueError(
                f"Mode {k + 1} of port {spec.name!r} carries no real power into the "
                "port, so the direction of its current cannot be fixed."
            )
        if real_power < 0.0:
            t_i[:, k] = -t_i[:, k]
        pilot_voltage = t_v[rows[k], k]
        if abs(pilot_voltage) == 0.0:
            raise ValueError(
                f"Mode {k + 1} of port {spec.name!r} has zero voltage on its pilot "
                f"terminal {spec.terminal_names[rows[k]]!r}."
            )
        phase = np.conj(pilot_voltage) / abs(pilot_voltage)
        t_v[:, k] = t_v[:, k] * phase
        t_i[:, k] = t_i[:, k] * phase
    return PortTerminalMatrices(
        t_v=t_v, t_i=t_i, pilot=rows, power=np.asarray(power), cond=cond
    )


@dataclass(frozen=True, eq=False)
class PortModes2D:
    """``T_V`` and ``T_I`` of one port from one 2D boundary-mode solve.

    Attributes:
        freq_ghz: Frequency of the 2D solve in GHz.
        n_eff: ``(N,)`` effective indices of the 2D modes, in the 2D solve's
            mode order (the real part is used).
        t_v: ``(N, N)`` terminal voltages, column ``l`` for 2D mode ``l``
            (see :func:`port_terminal_matrices`).
        t_i: ``(N, N)`` terminal currents, same layout.
    """

    freq_ghz: float
    n_eff: NDArray[np.float64]
    t_v: NDArray[np.complex128]
    t_i: NDArray[np.complex128]

    def __post_init__(self) -> None:
        """Coerce and validate the arrays.

        Raises:
            ValueError: For a frequency that is not positive and finite, or
                arrays that are not finite, ``(N,)`` and ``(N, N)``.
        """
        if not (np.isfinite(self.freq_ghz) and self.freq_ghz > 0.0):
            raise ValueError(
                f"freq_ghz must be positive and finite, got {self.freq_ghz}."
            )
        n_eff = np.real(np.asarray(self.n_eff, dtype=np.complex128))
        t_v = np.asarray(self.t_v, dtype=np.complex128)
        t_i = np.asarray(self.t_i, dtype=np.complex128)
        count = n_eff.size
        if (
            n_eff.ndim != 1
            or t_v.shape != (count, count)
            or t_i.shape != (count, count)
        ):
            raise ValueError(
                f"Expected n_eff (N,), t_v and t_i (N, N); got {n_eff.shape}, "
                f"{t_v.shape} and {t_i.shape}."
            )
        if not (
            np.all(np.isfinite(n_eff))
            and np.all(np.isfinite(t_v))
            and np.all(np.isfinite(t_i))
        ):
            raise ValueError("n_eff, t_v and t_i must be finite.")
        object.__setattr__(self, "n_eff", n_eff)
        object.__setattr__(self, "t_v", t_v)
        object.__setattr__(self, "t_i", t_i)


def port_modes_2d(
    fields: Sequence[BoundaryModeField],
    spec: TerminalPortSpec,
    *,
    freq_ghz: float,
    n_eff: ArrayLike,
    domain: Extent | None = None,
    paths: dict[str, Sequence[Point]] | None = None,
    current: Literal["contour", "reaction"] = "contour",
    nsamples: int = VOLTAGE_NSAMPLES,
    tol_um: float = DEFAULT_PERIMETER_TOL_UM,
) -> PortModes2D:
    """``T_V`` and ``T_I`` of one 2D solve, ready for :func:`align_gauge`.

    A thin wrapper of :func:`port_terminal_matrices` that keeps the frequency
    and the effective indices with the matrices; the phase reference is chosen
    later, per 3D mode, by :func:`align_gauge`.

    Args:
        fields: The saved fields of the modes of the solve, one per terminal.
        spec: The port's conductors.
        freq_ghz: Frequency of the solve.
        n_eff: ``(N,)`` effective indices of the modes, in the order of
            ``fields`` (from ``mode-kn.csv``).
        domain: The meshed domain, to draw the paths (see
            :func:`port_terminal_matrices`).
        paths: The paths declared on the solve instead of ``domain``.
        current: How ``T_I`` is obtained.
        nsamples: Sample points per path segment (default 800, see
            :func:`port_terminal_matrices`).
        tol_um: Distance a path sample may sit outside the mesh (um).

    Returns:
        The matrices with their frequency and effective indices.
    """
    matrices = port_terminal_matrices(
        fields,
        spec,
        domain=domain,
        paths=paths,
        current=current,
        nsamples=nsamples,
        tol_um=tol_um,
    )
    return PortModes2D(
        freq_ghz=float(freq_ghz),
        n_eff=np.asarray(n_eff),
        t_v=matrices.t_v,
        t_i=matrices.t_i,
    )
