"""Carrier-map to material-response coupling (solver-agnostic, pure functions).

This module converts charge-transport results — free-carrier concentrations
n(x, y), p(x, y) in cm^-3 — into the material responses the EM solvers
consume:

- RF: Drude conductivity ``sigma = q (mu_n n + mu_p p)`` in S/m
  (:func:`carrier_conductivity`), with the low-field mobilities falling with
  the ionized-impurity concentration (:class:`MobilityModel`) — the same
  model the charge-transport solve runs on, so the series resistance the two
  report is one quantity.
- Optics: refractive-index shift ``Delta n`` and free-carrier absorption
  ``Delta alpha`` through the plasma-dispersion power-law fits of
  Soref-Bennett (1987) and Nedeljkovic-Soref-Mashanovich (2011)
  (:class:`PlasmaDispersionModel`, :func:`carrier_index_shift`,
  :func:`carrier_absorption_cm`), plus the complex permittivity they imply
  (:func:`permittivity_perturbation`).
All coefficients are explicit and overridable so foundry-calibrated values
can be substituted for the published fits.

References:
    R. Soref and B. Bennett, "Electrooptical effects in silicon,"
    IEEE J. Quantum Electron. 23, 123-129 (1987).

    M. Nedeljkovic, R. Soref, and G. Z. Mashanovich, "Free-Carrier
    Electrorefraction and Electroabsorption Modulation Predictions for
    Silicon Over the 1-14 um Infrared Wavelength Range,"
    IEEE Photonics J. 3, 1171-1180 (2011).

    G. Masetti, M. Severi, and S. Solmi, "Modeling of carrier mobility
    against carrier concentration in arsenic-, phosphorus-, and boron-doped
    silicon," IEEE Trans. Electron Devices 30, 764-769 (1983).
"""

from __future__ import annotations

from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.constants import elementary_charge as Q  # noqa: N812

__all__ = [
    "DEFAULT_MU_N_CM2",
    "DEFAULT_MU_P_CM2",
    "MobilityModel",
    "PlasmaDispersionModel",
    "carrier_absorption_cm",
    "carrier_conductivity",
    "carrier_index_shift",
    "permittivity_perturbation",
]

#: Low-field electron mobility of lightly doped silicon at 300 K (cm^2/Vs).
DEFAULT_MU_N_CM2: float = 1417.0

#: Low-field hole mobility of lightly doped silicon at 300 K (cm^2/Vs).
DEFAULT_MU_P_CM2: float = 470.5


class MobilityModel(BaseModel):
    """Low-field carrier mobility against ionized-impurity concentration.

    The Masetti form, per carrier::

        mu(N) = mu_min1 exp(-P_c / N)
              + (mu_max - mu_min2) / (1 + (N / C_r)^alpha)
              - mu_1 / (1 + (C_s / N)^beta)

    with N the total ionized-impurity concentration in cm^-3. A lightly
    doped guide core keeps most of the lattice mobility; the contact
    Regions, doped a thousand times harder, keep about a twentieth of it —
    which a single constant cannot say, and which sets the series
    resistance of the slab. The presets carry the published silicon fit;
    :meth:`constant` recovers doping-independent mobilities, and the fields
    take foundry-calibrated values.

    Each ``*_n`` field is the electron parameter and each ``*_p`` field the
    hole one.
    """

    model_config = ConfigDict(validate_assignment=True)

    mu_max_n: float = Field(ge=0, description="Lattice mobility (cm^2/Vs)")
    mu_min1_n: float = Field(default=0.0, ge=0)
    mu_min2_n: float = Field(default=0.0, ge=0)
    mu_1_n: float = Field(default=0.0, ge=0)
    p_c_n: float = Field(default=0.0, ge=0, description="cm^-3")
    c_r_n: float = Field(default=1e17, gt=0, description="cm^-3")
    c_s_n: float = Field(default=1e20, gt=0, description="cm^-3")
    alpha_n: float = Field(default=1.0, gt=0)
    beta_n: float = Field(default=2.0, gt=0)

    mu_max_p: float = Field(ge=0, description="Lattice mobility (cm^2/Vs)")
    mu_min1_p: float = Field(default=0.0, ge=0)
    mu_min2_p: float = Field(default=0.0, ge=0)
    mu_1_p: float = Field(default=0.0, ge=0)
    p_c_p: float = Field(default=0.0, ge=0, description="cm^-3")
    c_r_p: float = Field(default=1e17, gt=0, description="cm^-3")
    c_s_p: float = Field(default=1e20, gt=0, description="cm^-3")
    alpha_p: float = Field(default=1.0, gt=0)
    beta_p: float = Field(default=2.0, gt=0)

    @classmethod
    def masetti_silicon(cls) -> MobilityModel:
        """Masetti et al. (1983) silicon at 300 K: arsenic and boron fits."""
        return cls(
            mu_max_n=DEFAULT_MU_N_CM2,
            mu_min1_n=52.2,
            mu_min2_n=52.2,
            mu_1_n=43.4,
            p_c_n=0.0,
            c_r_n=9.68e16,
            c_s_n=3.43e20,
            alpha_n=0.680,
            beta_n=2.0,
            mu_max_p=DEFAULT_MU_P_CM2,
            mu_min1_p=44.9,
            mu_min2_p=0.0,
            mu_1_p=29.0,
            p_c_p=9.23e16,
            c_r_p=2.23e17,
            c_s_p=6.10e20,
            alpha_p=0.719,
            beta_p=2.0,
        )

    @classmethod
    def constant(
        cls,
        *,
        mu_n_cm2: float = DEFAULT_MU_N_CM2,
        mu_p_cm2: float = DEFAULT_MU_P_CM2,
    ) -> MobilityModel:
        """Doping-independent mobilities.

        Args:
            mu_n_cm2: Electron mobility (cm^2/Vs).
            mu_p_cm2: Hole mobility (cm^2/Vs).
        """
        # mu_min1 = mu_min2 = mu_max cancels the doping-dependent term.
        return cls(
            mu_max_n=mu_n_cm2,
            mu_min1_n=mu_n_cm2,
            mu_min2_n=mu_n_cm2,
            mu_max_p=mu_p_cm2,
            mu_min1_p=mu_p_cm2,
            mu_min2_p=mu_p_cm2,
        )

    def _mobility(
        self, impurity_cm3: ArrayLike, carrier: Literal["n", "p"]
    ) -> NDArray[np.float64]:
        """Evaluate the Masetti form for one carrier."""
        impurity = np.asarray(impurity_cm3, dtype=np.float64)
        if np.any(impurity < 0):
            raise ValueError("Impurity concentrations must be non-negative (cm^-3).")
        mu_max, mu_min1, mu_min2, mu_1, p_c, c_r, c_s, alpha, beta = (
            getattr(self, f"{name}_{carrier}")
            for name in (
                "mu_max",
                "mu_min1",
                "mu_min2",
                "mu_1",
                "p_c",
                "c_r",
                "c_s",
                "alpha",
                "beta",
            )
        )
        # Undoped silicon is the N -> 0 limit of every term, which the
        # floor reaches without dividing by zero.
        doped = np.maximum(impurity, 1.0)
        return np.asarray(
            mu_min1 * np.exp(-p_c / doped)
            + (mu_max - mu_min2) / (1.0 + (doped / c_r) ** alpha)
            - mu_1 / (1.0 + (c_s / doped) ** beta),
            dtype=np.float64,
        )

    def electrons_cm2(self, impurity_cm3: ArrayLike) -> NDArray[np.float64]:
        """Electron mobility (cm^2/Vs) at an impurity concentration (cm^-3)."""
        return self._mobility(impurity_cm3, "n")

    def holes_cm2(self, impurity_cm3: ArrayLike) -> NDArray[np.float64]:
        """Hole mobility (cm^2/Vs) at an impurity concentration (cm^-3)."""
        return self._mobility(impurity_cm3, "p")


class PlasmaDispersionModel(BaseModel):
    """Power-law plasma-dispersion coefficients at one wavelength.

    The model evaluates::

        Delta n     = -(a_n N^b_n + a_p P^b_p)
        Delta alpha =   c_n N^d_n + c_p P^d_p      [cm^-1]

    with N, P the electron/hole concentrations in cm^-3. The presets carry
    the published silicon fits; construct the model directly (or
    ``model_copy(update=...)`` a preset) to substitute foundry-calibrated
    coefficients.

    Attributes:
        wavelength_um: Wavelength the coefficients are valid at (um).
        dn_electron_coeff: ``a_n`` in the Delta-n electron term.
        dn_electron_exp: ``b_n`` exponent of the Delta-n electron term.
        dn_hole_coeff: ``a_p`` in the Delta-n hole term.
        dn_hole_exp: ``b_p`` exponent of the Delta-n hole term.
        dalpha_electron_coeff: ``c_n`` in the Delta-alpha electron term.
        dalpha_electron_exp: ``d_n`` exponent of the Delta-alpha electron term.
        dalpha_hole_coeff: ``c_p`` in the Delta-alpha hole term.
        dalpha_hole_exp: ``d_p`` exponent of the Delta-alpha hole term.
    """

    model_config = ConfigDict(validate_assignment=True)

    wavelength_um: float = Field(gt=0, description="Validity wavelength (um)")
    dn_electron_coeff: float = Field(ge=0)
    dn_electron_exp: float = Field(gt=0)
    dn_hole_coeff: float = Field(ge=0)
    dn_hole_exp: float = Field(gt=0)
    dalpha_electron_coeff: float = Field(ge=0)
    dalpha_electron_exp: float = Field(gt=0)
    dalpha_hole_coeff: float = Field(ge=0)
    dalpha_hole_exp: float = Field(gt=0)

    @classmethod
    def nedeljkovic_1550(cls) -> PlasmaDispersionModel:
        """Nedeljkovic et al. (2011) power-law fit at 1.55 um."""
        return cls(
            wavelength_um=1.55,
            dn_electron_coeff=5.4e-22,
            dn_electron_exp=1.011,
            dn_hole_coeff=1.53e-18,
            dn_hole_exp=0.838,
            dalpha_electron_coeff=8.88e-21,
            dalpha_electron_exp=1.167,
            dalpha_hole_coeff=5.84e-20,
            dalpha_hole_exp=1.109,
        )

    @classmethod
    def nedeljkovic_1310(cls) -> PlasmaDispersionModel:
        """Nedeljkovic et al. (2011) power-law fit at 1.31 um."""
        return cls(
            wavelength_um=1.31,
            dn_electron_coeff=2.98e-22,
            dn_electron_exp=1.016,
            dn_hole_coeff=1.25e-18,
            dn_hole_exp=0.835,
            dalpha_electron_coeff=3.48e-22,
            dalpha_electron_exp=1.229,
            dalpha_hole_coeff=1.02e-19,
            dalpha_hole_exp=1.089,
        )

    @classmethod
    def soref_1550(cls) -> PlasmaDispersionModel:
        """Soref-Bennett (1987) linearized fit at 1.55 um."""
        return cls(
            wavelength_um=1.55,
            dn_electron_coeff=8.8e-22,
            dn_electron_exp=1.0,
            dn_hole_coeff=8.5e-18,
            dn_hole_exp=0.8,
            dalpha_electron_coeff=8.5e-18,
            dalpha_electron_exp=1.0,
            dalpha_hole_coeff=6.0e-18,
            dalpha_hole_exp=1.0,
        )


def _validated_carriers(
    n_cm3: ArrayLike, p_cm3: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert carrier inputs to arrays, rejecting negative concentrations."""
    n = np.asarray(n_cm3, dtype=np.float64)
    p = np.asarray(p_cm3, dtype=np.float64)
    if np.any(n < 0) or np.any(p < 0):
        raise ValueError("Carrier concentrations must be non-negative (cm^-3).")
    return n, p


def _power_term(
    x: NDArray[np.float64], coeff: float, exp: float
) -> NDArray[np.float64]:
    """Evaluate ``coeff * x**exp`` with 0**exp = 0 (no 0-division warnings)."""
    out = np.zeros_like(x)
    mask = x > 0
    out[mask] = coeff * x[mask] ** exp
    return out


@overload
def carrier_index_shift(
    n_cm3: NDArray[np.floating],
    p_cm3: ArrayLike,
    *,
    model: PlasmaDispersionModel,
) -> NDArray[np.float64]: ...
@overload
def carrier_index_shift(
    n_cm3: float,
    p_cm3: float,
    *,
    model: PlasmaDispersionModel,
) -> float: ...
def carrier_index_shift(
    n_cm3: ArrayLike,
    p_cm3: ArrayLike,
    *,
    model: PlasmaDispersionModel,
) -> NDArray[np.float64] | float:
    """Refractive-index shift ``Delta n`` from free carriers.

    Args:
        n_cm3: Electron concentration(s) in cm^-3 (scalar or array).
        p_cm3: Hole concentration(s) in cm^-3 (scalar or array).
        model: Plasma-dispersion coefficients at the target wavelength.

    Returns:
        ``Delta n`` (negative for positive carrier densities); scalar in,
        scalar out.
    """
    n, p = _validated_carriers(n_cm3, p_cm3)
    dn = np.asarray(
        -(
            _power_term(n, model.dn_electron_coeff, model.dn_electron_exp)
            + _power_term(p, model.dn_hole_coeff, model.dn_hole_exp)
        ),
        dtype=np.float64,
    )
    return dn if dn.ndim else float(dn)


@overload
def carrier_absorption_cm(
    n_cm3: NDArray[np.floating],
    p_cm3: ArrayLike,
    *,
    model: PlasmaDispersionModel,
) -> NDArray[np.float64]: ...
@overload
def carrier_absorption_cm(
    n_cm3: float,
    p_cm3: float,
    *,
    model: PlasmaDispersionModel,
) -> float: ...
def carrier_absorption_cm(
    n_cm3: ArrayLike,
    p_cm3: ArrayLike,
    *,
    model: PlasmaDispersionModel,
) -> NDArray[np.float64] | float:
    """Free-carrier absorption ``Delta alpha`` in cm^-1.

    Args:
        n_cm3: Electron concentration(s) in cm^-3 (scalar or array).
        p_cm3: Hole concentration(s) in cm^-3 (scalar or array).
        model: Plasma-dispersion coefficients at the target wavelength.

    Returns:
        ``Delta alpha`` in cm^-1 (non-negative); scalar in, scalar out.
    """
    n, p = _validated_carriers(n_cm3, p_cm3)
    dalpha = np.asarray(
        _power_term(n, model.dalpha_electron_coeff, model.dalpha_electron_exp)
        + _power_term(p, model.dalpha_hole_coeff, model.dalpha_hole_exp),
        dtype=np.float64,
    )
    return dalpha if dalpha.ndim else float(dalpha)


@overload
def carrier_conductivity(
    n_cm3: NDArray[np.floating],
    p_cm3: ArrayLike,
    *,
    mu_n_cm2: float = ...,
    mu_p_cm2: float = ...,
    mobility: MobilityModel | None = ...,
) -> NDArray[np.float64]: ...
@overload
def carrier_conductivity(
    n_cm3: float,
    p_cm3: float,
    *,
    mu_n_cm2: float = ...,
    mu_p_cm2: float = ...,
    mobility: MobilityModel | None = ...,
) -> float: ...
def carrier_conductivity(
    n_cm3: ArrayLike,
    p_cm3: ArrayLike,
    *,
    mu_n_cm2: float = DEFAULT_MU_N_CM2,
    mu_p_cm2: float = DEFAULT_MU_P_CM2,
    mobility: MobilityModel | None = None,
) -> NDArray[np.float64] | float:
    """Drude conductivity ``sigma = q (mu_n n + mu_p p)`` in S/m.

    With a ``mobility`` model the mobilities follow the impurity
    concentration, which the carriers themselves stand in for: ``n + p``.
    Wherever silicon conducts it is neutral, and its majority carriers
    number its ionized impurities; where it is depleted the stand-in reads
    low and the mobility high, on a conductivity that is vanishing anyway.
    That keeps the coupling a function of the Carrier map alone, so it
    applies unchanged to carriers averaged over a Strip.

    Args:
        n_cm3: Electron concentration(s) in cm^-3 (scalar or array).
        p_cm3: Hole concentration(s) in cm^-3 (scalar or array).
        mu_n_cm2: Electron mobility in cm^2/Vs, when no model is given.
        mu_p_cm2: Hole mobility in cm^2/Vs, when no model is given.
        mobility: Doping-dependent mobility model replacing the two
            constants.

    Returns:
        Conductivity in S/m; scalar in, scalar out.
    """
    if mu_n_cm2 < 0 or mu_p_cm2 < 0:
        raise ValueError("Mobilities must be non-negative (cm^2/Vs).")
    n, p = _validated_carriers(n_cm3, p_cm3)
    mu_n: NDArray[np.float64] | float = mu_n_cm2
    mu_p: NDArray[np.float64] | float = mu_p_cm2
    if mobility is not None:
        mu_n = mobility.electrons_cm2(n + p)
        mu_p = mobility.holes_cm2(n + p)
    # q [C] * mu [cm^2/Vs] * n [cm^-3] = sigma [S/cm]; * 100 -> S/m.
    sigma = np.asarray(Q * (mu_n * n + mu_p * p) * 100.0, dtype=np.float64)
    return sigma if sigma.ndim else float(sigma)


def permittivity_perturbation(
    *,
    n0: float,
    dn: float,
    dalpha_cm: float,
    wavelength_um: float,
) -> complex:
    """Complex relative permittivity of a carrier-perturbed dielectric.

    Builds ``eps = (n0 + dn - i kappa)^2`` with the extinction coefficient
    ``kappa = alpha * lambda / (4 pi)`` from the absorption change, using the
    ``exp(+i omega t)`` convention (lossy medium: ``Im(eps) < 0``).

    Args:
        n0: Unperturbed refractive index.
        dn: Carrier-induced index shift (from :func:`carrier_index_shift`).
        dalpha_cm: Carrier-induced absorption in cm^-1 (non-negative).
        wavelength_um: Vacuum wavelength in um.

    Returns:
        Complex relative permittivity.
    """
    if n0 <= 0:
        raise ValueError("n0 must be positive.")
    if dalpha_cm < 0:
        raise ValueError("dalpha_cm must be non-negative.")
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")
    alpha_m = dalpha_cm * 100.0
    kappa = alpha_m * wavelength_um * 1e-6 / (4.0 * np.pi)
    n_complex = (n0 + dn) - 1j * kappa
    return complex(n_complex * n_complex)
