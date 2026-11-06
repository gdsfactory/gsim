"""Scalar evaluation of schema oscillator models in SI angular frequency."""

from __future__ import annotations

from cmath import sqrt
from math import pi

from pdk_schema import Drude, Lorentz

_SPEED_OF_LIGHT_M_S = 299_792_458.0


def oscillator_index(
    model: Drude | Lorentz, wavelength_um: float
) -> tuple[float, float]:
    """Return n+ik for the schema's exp(-i*omega*t) convention."""
    angular_frequency = 2 * pi * _SPEED_OF_LIGHT_M_S / (wavelength_um * 1e-6)
    permittivity = complex(model.eps_inf)
    if isinstance(model, Drude):
        for term in model.terms:
            permittivity -= term.omega_p**2 / complex(
                angular_frequency**2, term.gamma * angular_frequency
            )
    else:
        for term in model.terms:
            permittivity += (
                term.delta_eps
                * term.omega_0**2
                / complex(
                    term.omega_0**2 - angular_frequency**2,
                    -term.gamma * angular_frequency,
                )
            )
    # Complex sqrt retains very small extinction coefficients without subtracting
    # two almost equal real numbers.
    refractive_index = sqrt(permittivity)
    return refractive_index.real, refractive_index.imag
