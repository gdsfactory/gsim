"""Cross-checks between the TCAD charge solve and the analytic Sze model.

The retained depletion-approximation path
(:class:`gsim.common.stack.pn_junction.PNJunctionConfig`) doubles as a
validation reference for the numeric solve: on an abrupt junction in the
fully depleted regime the TCAD small-signal C(V) must track the analytic
``eps_s / W(V)`` capacitance, and the carrier profile edges must match the
analytic depletion extents. These helpers expose that comparison to user
workflows as well as to the gated test-suite checks.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict

from gsim.common.stack.pn_junction import PNJunctionConfig
from gsim.tcad.results import BiasSweepResult

__all__ = [
    "CapacitanceComparison",
    "analytic_capacitance_f_per_cm",
    "compare_capacitance",
    "estimate_depletion_width_um",
]


def analytic_capacitance_f_per_cm(
    junction: PNJunctionConfig,
    v_reverse: ArrayLike,
    *,
    height_um: float,
) -> NDArray[np.float64]:
    """Sze depletion capacitance per cm of device depth versus reverse bias.

    Matches the units of the TCAD result: a 2D DEVSIM device is one cm
    deep, so the junction face area per cm of depth is ``height_um x 1 cm``
    and ``C = (eps_s / W) * height_um * 1e-6 [m] * 1e-2 [m]`` in F/cm.

    Args:
        junction: Depletion-model parameters (its ``v_reverse`` field is
            overridden per evaluation point).
        v_reverse: Reverse-bias values in volts (positive = reverse).
        height_um: Junction z-extent (um), e.g. the rib height.

    Returns:
        Analytic C(V) in F per cm of depth, one value per bias.
    """
    if height_um <= 0:
        raise ValueError("height_um must be positive.")
    biases = np.atleast_1d(np.asarray(v_reverse, dtype=np.float64))
    area_m2_per_cm = height_um * 1e-6 * 1e-2
    values = [
        junction.model_copy(update={"v_reverse": float(v)}).c_per_area * area_m2_per_cm
        for v in biases
    ]
    return np.asarray(values, dtype=np.float64)


class CapacitanceComparison(BaseModel):
    """Analytic (Sze) versus TCAD small-signal C(V) on shared bias points.

    Attributes:
        v_reverse: Reverse-bias values in volts (positive = reverse).
        c_tcad_f_per_cm: TCAD small-signal capacitance (F per cm of depth).
        c_analytic_f_per_cm: Sze depletion capacitance (F per cm of depth).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    v_reverse: NDArray[np.float64]
    c_tcad_f_per_cm: NDArray[np.float64]
    c_analytic_f_per_cm: NDArray[np.float64]

    @property
    def relative_deviation(self) -> NDArray[np.float64]:
        """Per-point ``|C_tcad - C_analytic| / C_analytic``."""
        return np.asarray(
            np.abs(self.c_tcad_f_per_cm - self.c_analytic_f_per_cm)
            / self.c_analytic_f_per_cm,
            dtype=np.float64,
        )

    @property
    def max_relative_deviation(self) -> float:
        """Worst-case relative deviation across the sweep."""
        return float(np.max(self.relative_deviation))

    def within(self, tolerance: float) -> bool:
        """True when every point agrees within the relative tolerance."""
        return bool(self.max_relative_deviation <= tolerance)


def compare_capacitance(
    junction: PNJunctionConfig,
    sweep: BiasSweepResult,
    *,
    height_um: float,
    reverse_bias_sign: float = 1.0,
) -> CapacitanceComparison:
    """Compare a TCAD bias sweep's C(V) against the analytic Sze model.

    Args:
        junction: Depletion-model parameters of the same junction.
        sweep: TCAD bias sweep result.
        height_um: Junction z-extent (um) for the analytic capacitance.
        reverse_bias_sign: Sign mapping the swept-contact bias to reverse
            bias, ``v_reverse = sign * bias_v``. The default ``+1`` fits a
            sweep on the cathode (n-side) contact, where positive bias
            reverse-biases the junction; use ``-1`` for an anode sweep.

    Returns:
        The per-point comparison of both C(V) curves.
    """
    v_reverse = reverse_bias_sign * sweep.voltages
    c_analytic = analytic_capacitance_f_per_cm(junction, v_reverse, height_um=height_um)
    return CapacitanceComparison(
        v_reverse=np.asarray(v_reverse, dtype=np.float64),
        c_tcad_f_per_cm=sweep.capacitance_f_per_cm,
        c_analytic_f_per_cm=c_analytic,
    )


def estimate_depletion_width_um(
    position_um: ArrayLike,
    electrons_cm3: ArrayLike,
    holes_cm3: ArrayLike,
    *,
    na_cm3: float,
    nd_cm3: float,
    fraction: float = 0.5,
) -> float:
    """Estimate the depletion width from a 1D carrier profile.

    The depleted region is where both carrier densities fall below
    ``fraction`` of the respective majority doping level; its extent along
    the profile is the depletion width. This mirrors how depletion edges
    are read off a drift-diffusion solution to compare against the
    analytic extents (``PNJunctionConfig.w_um``).

    Args:
        position_um: 1D positions along the junction axis (um).
        electrons_cm3: Electron concentrations at those positions.
        holes_cm3: Hole concentrations at those positions.
        na_cm3: Acceptor doping level on the P side (cm^-3).
        nd_cm3: Donor doping level on the N side (cm^-3).
        fraction: Majority-carrier threshold fraction defining depletion.

    Returns:
        Depletion width in um (0.0 when no point is depleted).
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1).")
    pos = np.asarray(position_um, dtype=np.float64)
    n = np.asarray(electrons_cm3, dtype=np.float64)
    p = np.asarray(holes_cm3, dtype=np.float64)
    if pos.ndim != 1 or n.shape != pos.shape or p.shape != pos.shape:
        raise ValueError("position, electrons and holes must be equal-length 1D.")
    depleted = (n < fraction * nd_cm3) & (p < fraction * na_cm3)
    if not np.any(depleted):
        return 0.0
    span = pos[depleted]
    return float(span.max() - span.min())
