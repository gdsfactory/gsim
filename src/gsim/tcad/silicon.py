"""Silicon constants of the charge-transport solve.

DEVSIM's ``simple_physics`` sets the static relative permittivity and the
intrinsic carrier density of silicon as two constants, whatever the
temperature. The device keeps its permittivity and replaces its intrinsic
density with :func:`intrinsic_density_cm3`, which follows the band gap
with temperature and equals DEVSIM's value at 300 K. The depletion-model
comparison (:func:`gsim.tcad.compare_capacitance`) reads the same values,
so the two sides of it describe the same silicon.

References:
    Y. P. Varshni, "Temperature dependence of the energy gap in
    semiconductors," Physica 34, 149-154 (1967).

    S. M. Sze and K. K. Ng, *Physics of Semiconductor Devices*, 3rd ed.,
    Wiley (2007), section 1.4.
"""

from __future__ import annotations

import math

from scipy.constants import Boltzmann, elementary_charge

#: Static relative permittivity of silicon in DEVSIM's ``simple_physics``.
#: The RF and optical stacks use 11.9; the charge solve keeps DEVSIM's own.
SILICON_RELATIVE_PERMITTIVITY: float = 11.1

#: Intrinsic carrier density of silicon at 300 K in DEVSIM's
#: ``simple_physics`` (cm^-3).
INTRINSIC_DENSITY_300K_CM3: float = 1e10

#: Varshni fit of the silicon band gap, ``E_g(T) = E_g0 - a T^2 / (T + b)``:
#: ``E_g0`` (eV), ``a`` (eV/K) and ``b`` (K).
VARSHNI_SILICON: tuple[float, float, float] = (1.170, 4.73e-4, 636.0)

_REFERENCE_K: float = 300.0
_BOLTZMANN_EV_PER_K: float = Boltzmann / elementary_charge


def _require_temperature(temperature_k: float) -> None:
    """Refuse a non-positive or non-finite temperature."""
    if not (math.isfinite(temperature_k) and temperature_k > 0):
        raise ValueError(f"temperature_k must be positive; got {temperature_k}")


def band_gap_ev(temperature_k: float) -> float:
    """Band gap of silicon (eV) at a lattice temperature (K), Varshni's fit."""
    _require_temperature(temperature_k)
    e_g0, alpha, beta = VARSHNI_SILICON
    return e_g0 - alpha * temperature_k**2 / (temperature_k + beta)


def intrinsic_density_cm3(temperature_k: float) -> float:
    """Intrinsic carrier density of silicon (cm^-3) at a temperature (K).

    ``n_i(T) = n_i(300 K) (T / 300)^1.5 exp(E_g(300)/2kT_300 - E_g(T)/2kT)``,
    anchored at DEVSIM's 300 K value, so a 300 K solve is unchanged.
    """
    _require_temperature(temperature_k)

    def exponent(t: float) -> float:
        return band_gap_ev(t) / (2.0 * _BOLTZMANN_EV_PER_K * t)

    return (
        INTRINSIC_DENSITY_300K_CM3
        * (temperature_k / _REFERENCE_K) ** 1.5
        * math.exp(exponent(_REFERENCE_K) - exponent(temperature_k))
    )


__all__ = [
    "INTRINSIC_DENSITY_300K_CM3",
    "SILICON_RELATIVE_PERMITTIVITY",
    "VARSHNI_SILICON",
    "band_gap_ev",
    "intrinsic_density_cm3",
]
