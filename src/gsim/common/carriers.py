"""Low-field carrier mobility of doped silicon.

:class:`MobilityModel` gives the electron and hole mobilities against the
ionized-impurity concentration, the Masetti fit by default. The
charge-transport solve evaluates it node by node, so a contact Region
doped a thousand times harder than the guide core keeps only a fraction
of the lattice mobility, which sets the series resistance of the slab.

References:
    G. Masetti, M. Severi, and S. Solmi, "Modeling of carrier mobility
    against carrier concentration in arsenic-, phosphorus-, and boron-doped
    silicon," IEEE Trans. Electron Devices 30, 764-769 (1983).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DEFAULT_MU_N_CM2", "DEFAULT_MU_P_CM2", "MobilityModel"]

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
