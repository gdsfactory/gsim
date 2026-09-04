"""PN-junction depletion model (Sze, *Physics of Semiconductor Devices*).

This module implements the textbook depletion approximation for an abrupt or
linearly graded PN junction:

- S. M. Sze and K. K. Ng, *Physics of Semiconductor Devices*, 3rd ed.,
  Wiley (2007), chapter 2 ("p-n Junction Diodes").

Provided quantities (all concentrations in ``cm^-3``, lengths in ``um``):

1. Built-in potential::

       V_bi = (k_B T / q) ln(Na Nd / ni^2)                    (Sze eq. 2.60)

2. Depletion width under reverse bias VR (abrupt junction)::

       W  = sqrt( 2 eps_s (V_bi + VR) / q * (Na + Nd)/(Na Nd) )  (eq. 2.66)
       x_p = W Nd / (Na + Nd)   (spilled into the P side)
       x_n = W Na / (Na + Nd)   (spilled into the N side)

3. Depletion width for a linearly graded junction with grade constant
   ``a = |dN/dx|`` near the metallurgical junction::

       W = [ 12 eps_s (V_bi + VR) / (q a) ]^(1/3)              (eq. 2.72)

4. Junction capacitance per unit area (parallel-plate form of the depletion
   charge, valid for W much smaller than the device lateral dimensions)::

       C_j = eps_s / W

The same module also provides :func:`select_junction_mode`, which decides
whether the depletion strip can be resolved on the simulation mesh
(``"high_res"``) or should be collapsed into a lumped capacitance boundary
(``"capacitance"``).

Example:
-------
    >>> from gsim.common.stack.junction import PNJunctionConfig
    >>> junc = PNJunctionConfig(na_cm3=1e19, nd_cm3=1e19, v_reverse=0.0)
    >>> junc.v_bi  # built-in potential [V]
    >>> junc.w_um  # total depletion width [um]
    >>> junc.xp_um  # depletion extent into the P side [um]
    >>> junc.xn_um  # depletion extent into the N side [um]
    >>> junc.capacitance(length_um=10.0, height_um=0.22)  # absolute C [F]
"""

from __future__ import annotations

import math
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.constants import Boltzmann as KB  # noqa: N814
from scipy.constants import elementary_charge as Q  # noqa: N812
from scipy.constants import epsilon_0 as EPS0  # noqa: N812

__all__ = [
    "DEFAULT_SI_PERMITTIVITY",
    "JUNCTION_MODE_FRACTION",
    "NI_SI_300K_CM3",
    "PNJunctionConfig",
    "built_in_voltage",
    "depletion_extents",
    "depletion_width",
    "junction_capacitance_per_area",
    "select_junction_mode",
]

#: Intrinsic carrier concentration of silicon at 300 K in cm^-3.
#: Classic textbook value used by Sze; override for other materials/T.
NI_SI_300K_CM3: float = 1.5e10

#: Default relative permittivity of depleted (intrinsic) silicon.
DEFAULT_SI_PERMITTIVITY: float = 11.9

#: A depletion width is considered mesh-resolvable when it reaches this
#: fraction of the smallest doped section flanking the junction.
JUNCTION_MODE_FRACTION: float = 0.2

JunctionMode = Literal["capacitance", "high_res"]


def built_in_voltage(
    na_cm3: float,
    nd_cm3: float,
    *,
    temperature_k: float = 300.0,
    ni_cm3: float = NI_SI_300K_CM3,
) -> float:
    """Compute the built-in potential ``V_bi`` of a PN junction in volts.

    Implements ``V_bi = (k_B T / q) ln(Na Nd / ni^2)`` (Sze ch. 2).

    Args:
        na_cm3: Acceptor concentration on the P side in cm^-3 (> 0).
        nd_cm3: Donor concentration on the N side in cm^-3 (> 0).
        temperature_k: Lattice temperature in kelvin (> 0).
        ni_cm3: Intrinsic carrier concentration in cm^-3 (> 0).

    Returns:
        Built-in potential in volts.

    Raises:
        ValueError: If any input is non-positive or ``Na*Nd <= ni**2``.
    """
    if na_cm3 <= 0 or nd_cm3 <= 0:
        raise ValueError("Doping concentrations must be positive (cm^-3).")
    if temperature_k <= 0:
        raise ValueError("temperature_k must be positive.")
    if ni_cm3 <= 0:
        raise ValueError("ni_cm3 must be positive.")
    product = na_cm3 * nd_cm3
    if product <= ni_cm3**2:
        raise ValueError(
            f"Na*Nd ({product:.3g} cm^-6) must exceed ni^2 "
            f"({ni_cm3**2:.3g} cm^-6); degenerate case has no junction."
        )
    vt = KB * temperature_k / Q
    return float(vt * math.log(product / ni_cm3**2))


def _validate_bias(v_reverse: float, v_bi: float) -> None:
    """Reject bias points beyond flat-band (no physical solution)."""
    if v_bi + v_reverse <= 0:
        raise ValueError(
            f"V_bi + v_reverse = {v_bi + v_reverse:.4g} V must be > 0 "
            "(applied forward bias beyond flat-band has no solution)."
        )


def _eps_si(permittivity: float) -> float:
    """Return absolute permittivity in F/m from a relative value."""
    if permittivity < 1.0:
        raise ValueError("permittivity must be >= 1.")
    return permittivity * EPS0


def depletion_width(
    na_cm3: float,
    nd_cm3: float,
    *,
    v_reverse: float = 0.0,
    temperature_k: float = 300.0,
    ni_cm3: float = NI_SI_300K_CM3,
    permittivity: float = DEFAULT_SI_PERMITTIVITY,
    grading: Literal["abrupt", "linear"] = "abrupt",
    grade_const_cm4: float | None = None,
) -> float:
    """Compute the total depletion width ``W`` in micrometers.

    Args:
        na_cm3: Acceptor concentration in cm^-3 (> 0).
        nd_cm3: Donor concentration in cm^-3 (> 0).
        v_reverse: Applied reverse-bias voltage in volts (positive = reverse).
            Negative values model forward bias down to (but excluding)
            flat-band.
        temperature_k: Lattice temperature in kelvin.
        ni_cm3: Intrinsic carrier concentration in cm^-3.
        permittivity: Relative permittivity of the semiconductor.
        grading: ``"abrupt"`` (step junction) or ``"linear"`` (linearly
            graded).
        grade_const_cm4: Grade constant ``a = |dN/dx|`` in cm^-4 for
            ``grading="linear"``.

    Returns:
        Total depletion width in micrometers.

    Raises:
        ValueError: On non-positive inputs, missing grade constant, or bias
            beyond flat-band.
    """
    v_bi = built_in_voltage(na_cm3, nd_cm3, temperature_k=temperature_k, ni_cm3=ni_cm3)
    _validate_bias(v_reverse, v_bi)
    eps_s = _eps_si(permittivity)

    if grading == "linear":
        if grade_const_cm4 is None or grade_const_cm4 <= 0:
            raise ValueError("grading='linear' requires grade_const_cm4 > 0.")
        # a in m^-4 (1 cm^-4 = 1e6 m^-4); W comes out in meters.
        a_m4 = grade_const_cm4 * 1e6
        w_m = (12.0 * eps_s * (v_bi + v_reverse) / (Q * a_m4)) ** (1.0 / 3.0)
        return float(w_m * 1e6)

    if grading != "abrupt":
        raise ValueError(f"Unknown grading type: {grading!r}")

    # Abrupt junction: W = sqrt(2 eps_s (V_bi+VR)/q * (Na+Nd)/(NaNd)).
    na_m3 = na_cm3 * 1e6
    nd_m3 = nd_cm3 * 1e6
    w_m = math.sqrt(
        2.0 * eps_s * (v_bi + v_reverse) / Q * (na_m3 + nd_m3) / (na_m3 * nd_m3)
    )
    return float(w_m * 1e6)


def depletion_extents(
    na_cm3: float,
    nd_cm3: float,
    *,
    w_um: float,
    grading: Literal["abrupt", "linear"] = "abrupt",
) -> tuple[float, float]:
    """Split a total depletion width into P-side/N-side extents in micrometers.

    For an abrupt junction the depletion spills asymmetrically::

        x_p = W Nd / (Na + Nd),    x_n = W Na / (Na + Nd)

    A linearly graded junction is symmetric around the metallurgical
    junction, so ``x_p = x_n = W/2``.

    Args:
        na_cm3: Acceptor concentration in cm^-3 (> 0).
        nd_cm3: Donor concentration in cm^-3 (> 0).
        w_um: Total depletion width in micrometers (from
            :func:`depletion_width`).
        grading: Junction grading type.

    Returns:
        ``(xp_um, xn_um)`` — extents spilled into the P and N sides.
    """
    if na_cm3 <= 0 or nd_cm3 <= 0:
        raise ValueError("Doping concentrations must be positive (cm^-3).")
    if w_um < 0:
        raise ValueError("w_um must be non-negative.")
    if grading == "linear":
        return w_um / 2.0, w_um / 2.0
    total = na_cm3 + nd_cm3
    return w_um * nd_cm3 / total, w_um * na_cm3 / total


def junction_capacitance_per_area(
    permittivity: float,
    w_um: float,
) -> float:
    """Depletion capacitance per unit area ``C_j = eps_s / W`` in F/m^2.

    Args:
        permittivity: Relative permittivity of the semiconductor.
        w_um: Total depletion width in micrometers (> 0).

    Returns:
        Capacitance per unit area in F/m^2.
    """
    if w_um <= 0:
        raise ValueError("w_um must be positive.")
    return _eps_si(permittivity) / (w_um * 1e-6)


def select_junction_mode(
    w_um: float,
    p_extent_um: float,
    n_extent_um: float,
    *,
    fraction: float = JUNCTION_MODE_FRACTION,
) -> JunctionMode:
    """Choose how to represent the depletion region in a simulation.

    The depletion strip is meshed explicitly (``"high_res"``) when its width
    is comparable to the doped sections flanking it — specifically when
    ``w_um >= fraction * min(p_extent, n_extent)``. Otherwise the region is
    far thinner than its neighbours and meshing it would only bloat the
    model, so a lumped capacitance boundary is used instead
    (``"capacitance"``).

    Args:
        w_um: Total depletion width in micrometers (> 0).
        p_extent_um: Size of the doped section flanking the junction on the
            P side (micrometers, > 0).
        n_extent_um: Size of the doped section flanking the junction on the
            N side (micrometers, > 0).
        fraction: Resolvability threshold as a fraction of the smaller flank
            (default ~1/5).

    Returns:
        ``"high_res"`` when the geometry should carry the depletion strip,
        ``"capacitance"`` otherwise.
    """
    if w_um <= 0:
        raise ValueError("w_um must be positive.")
    if p_extent_um <= 0 or n_extent_um <= 0:
        raise ValueError("Flank extents must be positive.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1].")
    threshold_um = fraction * min(p_extent_um, n_extent_um)
    return "high_res" if w_um >= threshold_um else "capacitance"


class PNJunctionConfig(BaseModel):
    """Parameters of a PN-junction depletion model (depletion approximation).

    Concentrations use the semiconductor-industry convention (cm^-3);
    derived lengths are exposed in micrometers and capacitances in farads.
    See module docstring for the underlying formulas (Sze ch. 2).

    Attributes:
        na_cm3: Acceptor concentration on the P side (cm^-3).
        nd_cm3: Donor concentration on the N side (cm^-3).
        v_reverse: Applied reverse bias in volts (positive = reverse;
            negative values model forward bias below flat-band).
        temperature_k: Lattice temperature in kelvin.
        ni_cm3: Intrinsic carrier concentration (cm^-3).
        permittivity: Relative permittivity of the depleted semiconductor.
        grading: ``"abrupt"`` or ``"linear"`` junction profile.
        grade_const_cm4: Grade constant ``a = |dN/dx|`` in cm^-4, required
            when ``grading="linear"``.
    """

    model_config = ConfigDict(validate_assignment=True)

    na_cm3: float = Field(gt=0, description="Acceptor concentration (cm^-3)")
    nd_cm3: float = Field(gt=0, description="Donor concentration (cm^-3)")
    v_reverse: float = Field(
        default=0.0, description="Applied reverse bias [V] (positive = reverse)"
    )
    temperature_k: float = Field(default=300.0, gt=0, description="Temperature [K]")
    ni_cm3: float = Field(
        default=NI_SI_300K_CM3, gt=0, description="Intrinsic carriers (cm^-3)"
    )
    permittivity: float = Field(
        default=DEFAULT_SI_PERMITTIVITY,
        ge=1.0,
        description="Relative permittivity of the semiconductor",
    )
    grading: Literal["abrupt", "linear"] = Field(default="abrupt")
    grade_const_cm4: float | None = Field(
        default=None, gt=0, description="Grade constant a = |dN/dx| (cm^-4)"
    )

    @model_validator(mode="after")
    def _validate_physics(self) -> Self:
        """Check grading configuration and bias range."""
        if self.grading == "linear" and self.grade_const_cm4 is None:
            raise ValueError("grading='linear' requires grade_const_cm4.")
        _validate_bias(self.v_reverse, self.v_bi)
        return self

    @property
    def v_bi(self) -> float:
        """Built-in potential in volts."""
        return built_in_voltage(
            self.na_cm3,
            self.nd_cm3,
            temperature_k=self.temperature_k,
            ni_cm3=self.ni_cm3,
        )

    @property
    def w_um(self) -> float:
        """Total depletion width in micrometers at the configured bias."""
        return depletion_width(
            self.na_cm3,
            self.nd_cm3,
            v_reverse=self.v_reverse,
            temperature_k=self.temperature_k,
            ni_cm3=self.ni_cm3,
            permittivity=self.permittivity,
            grading=self.grading,
            grade_const_cm4=self.grade_const_cm4,
        )

    @property
    def xp_um(self) -> float:
        """Depletion extent spilled into the P side (micrometers)."""
        xp, _xn = depletion_extents(
            self.na_cm3, self.nd_cm3, w_um=self.w_um, grading=self.grading
        )
        return xp

    @property
    def xn_um(self) -> float:
        """Depletion extent spilled into the N side (micrometers)."""
        _xp, xn = depletion_extents(
            self.na_cm3, self.nd_cm3, w_um=self.w_um, grading=self.grading
        )
        return xn

    @property
    def c_per_area(self) -> float:
        """Junction capacitance per unit area in F/m^2 (``eps_s / W``)."""
        return junction_capacitance_per_area(self.permittivity, self.w_um)

    def capacitance(self, length_um: float, height_um: float) -> float:
        """Absolute junction capacitance for a rectangular junction face.

        Treats the depletion strip as a parallel-plate capacitor of area
        ``length x height`` filled with the depleted semiconductor:
        ``C = eps_s * A / W``.

        Args:
            length_um: Device length along the propagation direction (um).
            height_um: Junction z-extent (um), e.g. the rib height.

        Returns:
            Absolute capacitance in farads.
        """
        if length_um <= 0 or height_um <= 0:
            raise ValueError("length_um and height_um must be positive.")
        area_m2 = length_um * height_um * 1e-12
        return float(self.c_per_area * area_m2)

    def select_mode(
        self,
        p_extent_um: float,
        n_extent_um: float,
        *,
        fraction: float = JUNCTION_MODE_FRACTION,
    ) -> JunctionMode:
        """Auto-select the representation mode for this junction.

        Thin wrapper around :func:`select_junction_mode` using this config's
        computed depletion width.

        Args:
            p_extent_um: Size of the doped flank on the P side (um).
            n_extent_um: Size of the doped flank on the N side (um).
            fraction: Resolvability threshold fraction (~1/5 default).

        Returns:
            ``"high_res"`` or ``"capacitance"``.
        """
        return select_junction_mode(
            self.w_um, p_extent_um, n_extent_um, fraction=fraction
        )

    def to_metadata(self) -> dict[str, Any]:
        """Return a plain-dict summary of the computed junction quantities."""
        return {
            "na_cm3": self.na_cm3,
            "nd_cm3": self.nd_cm3,
            "v_reverse": self.v_reverse,
            "temperature_k": self.temperature_k,
            "v_bi": self.v_bi,
            "w_um": self.w_um,
            "xp_um": self.xp_um,
            "xn_um": self.xn_um,
            "c_per_area_f_m2": self.c_per_area,
            "grading": self.grading,
        }
