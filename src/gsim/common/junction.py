"""The series-RC junction branch and its fit to a small-signal admittance.

A reverse-biased junction seen from its terminals is, to first order, its
depletion capacitance behind the series resistance of the doped slab.
:func:`series_rc_from_admittance` reads that pair out of a complex
admittance, the one a charge-transport small-signal solve returns
(:meth:`gsim.tcad.BiasPoint.junction_branch`), and :class:`JunctionBranch`
carries the two numbers together.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["JunctionBranch", "series_rc_from_admittance"]


class JunctionBranch(NamedTuple):
    """The series-RC shunt branch per meter of junction.

    The lumped junction model the standard loaded-line workflow inserts
    per unit length: the junction capacitance behind the series
    resistance of the doped slab. A named pair, so the two numbers that
    always travel together do so under their own names; both are floats
    from a single Bias point's fit and same-shape arrays from a sweep's.

    Attributes:
        r_s_ohm_m: Series resistance (ohm*m).
        c_j_f_per_m: Junction capacitance (F/m).
    """

    r_s_ohm_m: float | NDArray[np.float64]
    c_j_f_per_m: float | NDArray[np.float64]


def series_rc_from_admittance(
    y_s_per_m: ArrayLike,
    *,
    freq_hz: float,
) -> JunctionBranch:
    """Fit a series-RC shunt branch to a small-signal admittance.

    Inverts ``Y = 1 / (R_s + 1/(j omega C_j))``: the branch impedance is
    ``Z = 1/Y = R_s - j/(omega C_j)``, so ``R_s = Re(Z)`` and
    ``C_j = -1/(omega Im(Z))``.

    Args:
        y_s_per_m: Complex shunt admittance per meter of junction (S/m);
            scalar or array, fit element by element.
        freq_hz: Frequency the admittance was measured at (Hz, > 0).

    Returns:
        The fitted :class:`JunctionBranch`, its fields the same shape as
        ``y_s_per_m``.

    Raises:
        ValueError: When the frequency is not positive, or the admittance
            is not one a series RC can represent (negative conductance,
            or a non-capacitive susceptance).
    """
    if freq_hz <= 0:
        raise ValueError("The fit frequency must be positive.")
    y = np.asarray(y_s_per_m, dtype=np.complex128)
    if np.any(y.real < 0):
        raise ValueError(
            "The admittance has negative conductance, which no series RC "
            "branch can represent."
        )
    if np.any(y.imag <= 0):
        raise ValueError(
            "The admittance is not capacitive (Im(Y) <= 0), so a series-RC "
            "junction branch cannot represent it."
        )
    omega = 2.0 * np.pi * freq_hz
    z = 1.0 / y
    return JunctionBranch(
        r_s_ohm_m=np.asarray(z.real, dtype=np.float64),
        c_j_f_per_m=np.asarray(-1.0 / (omega * z.imag), dtype=np.float64),
    )
