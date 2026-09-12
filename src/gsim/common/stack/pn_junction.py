"""PN-junction model (Sze, *Physics of Semiconductor Devices*).

This module consolidates the PN-junction helpers: the textbook depletion
approximation for abrupt or linearly graded junctions, the 1D free-carrier
plasma-dispersion model for the complex optical permittivity, and the
solver-agnostic geometry builders for contiguous doping regions.

- S. M. Sze and K. K. Ng, *Physics of Semiconductor Devices*, 3rd ed.,
  Wiley (2007), chapter 2 ("p-n Junction Diodes").

Depletion quantities (all concentrations in ``cm^-3``, lengths in ``um``):

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
(``"capacitance"``), and :func:`junction_epsilon_profile`, which evaluates
the 1D complex permittivity across the junction at optical frequencies
from the Drude plasma-dispersion of the free carriers.

Example:
-------
    >>> from gsim.common.stack.pn_junction import PNJunctionConfig
    >>> junc = PNJunctionConfig(na_cm3=1e19, nd_cm3=1e19, v_reverse=0.0)
    >>> junc.v_bi  # built-in potential [V]
    >>> junc.w_um  # total depletion width [um]
    >>> junc.xp_um  # depletion extent into the P side [um]
    >>> junc.xn_um  # depletion extent into the N side [um]
    >>> junc.capacitance(length_um=10.0, height_um=0.22)  # absolute C [F]
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Literal, Self, cast

import gdsfactory as gf
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.constants import Boltzmann as KB  # noqa: N814
from scipy.constants import c as C_LIGHT  # noqa: N812
from scipy.constants import electron_mass as M0  # noqa: N812
from scipy.constants import elementary_charge as Q  # noqa: N812
from scipy.constants import epsilon_0 as EPS0  # noqa: N812

from gsim.common.stack.materials import MaterialProperties, make_doped_materials

if TYPE_CHECKING:
    from gsim.common.stack.extractor import Layer

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SI_PERMITTIVITY",
    "JUNCTION_MODE_FRACTION",
    "MU_N_CM2_VS",
    "MU_P_CM2_VS",
    "M_CE_STAR",
    "M_CH_STAR",
    "NI_SI_300K_CM3",
    "SIGMA_NEGLIGIBLE_SM",
    "PNJunctionConfig",
    "built_in_voltage",
    "carrier_profile_1d",
    "default_eps_bg_rel",
    "depletion_extents",
    "depletion_width",
    "drude_relaxation_times",
    "epsilon_eff_relative",
    "junction_capacitance_per_area",
    "junction_epsilon_profile",
    "make_doping_profile",
    "make_pn_junction_profile",
    "make_segmented_junction_profile",
    "optical_params",
    "refractive_index",
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


# ---------------------------------------------------------------------------
# Doping-profile construction (merged from the former ``doping`` module).
# ---------------------------------------------------------------------------


_SideConfig = dict[str, dict[str, Any]]


def make_doping_profile(
    comp: gf.Component,
    *,
    length: float,
    rib_center_y: float,
    rib_width: float,
    profile: dict[str, list[tuple[float, float]]],
    sides: _SideConfig,
    zmin: float,
    zmax: float,
    permittivity: float = 11.9,
    fmax: float = 200e9,
    mesh_resolution: str | float = "fine",
) -> dict[str, dict[str, Any]]:
    """Add contiguous doping regions beside a rib and build layer/material specs.

    For each side (e.g. ``"upper"`` / ``"lower"``) the regions listed in
    *profile* are placed as adjacent rectangles starting at the rib edge and
    extending outward, so the doping is contiguous with no gaps.  Each region
    ``i`` on a side gets:

    - a gdsfactory rectangle of size ``(length, width)`` on the GDS layer
      ``(base_layer[0], base_layer[1] + i)``,
    - a ``Layer`` spec named ``"{name_prefix}{i}"``,
    - a ``MaterialProperties`` entry with the region's Drude conductivity.

    Args:
        comp: gdsfactory component the rectangles are added to.
        length: Rectangle length along the propagation direction (um).
        rib_center_y: Y coordinate of the rib centre (um).
        rib_width: Rib width (um); regions start at the rib edges.
        profile: Per-side region list ``{side: [(width_um, sigma_S_per_m), ...]}``.
        sides: Per-side configuration: each value is a dict with keys
            ``base_layer`` (``(layer, datatype)`` tuple for the first region),
            ``name_prefix`` (region-name prefix) and ``sign`` (+1 extends in
            +y, -1 in -y).
        zmin: Bottom z of the doping regions (um).
        zmax: Top z of the doping regions (um).
        permittivity: Relative permittivity shared by all regions (e.g. 11.9).
        fmax: Upper frequency of the dispersion-model validity range (Hz).
        mesh_resolution: Mesh resolution assigned to the generated ``Layer``.

    Returns:
        Dict with keys ``layer_specs`` (``{name: Layer}``), ``materials``
        (``{name: MaterialProperties}``) and ``centres``
        (``{side: [y_centre, ...]}``).
    """
    from gsim.common.stack.extractor import Layer

    result: dict[str, dict[str, Any]] = {
        "layer_specs": {},
        "materials": {},
        "centres": {},
    }
    layer_specs = cast("dict[str, Layer]", result["layer_specs"])
    materials: dict[str, Any] = result["materials"]
    centres: dict[str, list[float]] = result["centres"]

    for side, cfg in sides.items():
        regions = profile.get(side, [])
        sign = cfg["sign"]
        base_layer = tuple(cfg["base_layer"])
        prefix = cfg["name_prefix"]

        pos = rib_center_y + sign * rib_width / 2  # start at rib edge
        side_centres: list[float] = []
        side_specs: dict[str, tuple[Any, float]] = {}

        for i, (width, sigma) in enumerate(regions):
            name = f"{prefix}{i}"
            gds_layer = (base_layer[0], base_layer[1] + i)
            centre = pos + sign * width / 2

            rect = comp << gf.c.rectangle((length, width), layer=gds_layer)
            rect.y = centre
            side_centres.append(centre)
            side_specs[name] = (gds_layer, sigma)
            pos += sign * width

        centres[side] = side_centres
        if not side_specs:
            continue

        layer_specs.update(
            {
                name: Layer(
                    name=name,
                    gds_layer=gds_layer,
                    zmin=zmin,
                    zmax=zmax,
                    thickness=zmax - zmin,
                    material=name,
                    layer_type="dielectric",
                    mesh_resolution=mesh_resolution,
                )
                for name, (gds_layer, _sigma) in side_specs.items()
            }
        )
        materials.update(
            make_doped_materials(
                [(name, sigma) for name, (_gds, sigma) in side_specs.items()],
                permittivity=permittivity,
                fmax=fmax,
                source_prefix="doped Si",
            )
        )

    return result


def _as_junction_config(
    junction: PNJunctionConfig | dict[str, Any],
) -> PNJunctionConfig:
    """Accept a config object or plain dict for the junction parameters."""
    if isinstance(junction, PNJunctionConfig):
        return junction
    return PNJunctionConfig.model_validate(junction)


def _add_rect(
    comp: gf.Component,
    *,
    length: float,
    y0: float,
    y1: float,
    gds_layer: tuple[int, int],
) -> float:
    """Draw a rectangle spanning ``[y0, y1]`` and return its y-centre."""
    rect = comp << gf.c.rectangle((length, y1 - y0), layer=gds_layer)
    rect.y = (y0 + y1) / 2
    return (y0 + y1) / 2


def make_pn_junction_profile(
    comp: gf.Component,
    *,
    length: float,
    center_y: float,
    rib_width: float,
    junction: PNJunctionConfig | dict[str, Any],
    p_region: tuple[str, tuple[int, int], float],
    n_region: tuple[str, tuple[int, int], float],
    junction_region: tuple[str, tuple[int, int]] | None = None,
    zmin: float = 0.0,
    zmax: float | None = None,
    fmax: float = 200e9,
    mode: Literal["auto", "capacitance", "high_res"] = "auto",
    mode_fraction: float = JUNCTION_MODE_FRACTION,
    mesh_resolution: str | float = "fine",
) -> dict[str, dict[str, Any]]:
    """Build P / depletion-junction / N rib regions around ``center_y``.

    The depletion width ``W`` (and its asymmetric split ``xp``/``xn`` into
    the P and N halves) comes from :class:`PNJunctionConfig` (the depletion
    model in this module).

    Two representation modes are supported:

    - ``"high_res"``: three contiguous rectangles are drawn — N
      ``[cy - rib_width/2, cy - xn]``, depleted-junction dielectric strip
      ``[cy - xn, cy + xp]``, P ``[cy + xp, cy + rib_width/2]``. The
      junction strip is registered as a patterned dielectric with a real
      GDS layer so it appears on the simulation mesh.
    - ``"capacitance"``: geometry is unchanged from a plain P/N split
      (adjacent half-rectangles); no junction polygon is drawn and callers
      apply the computed capacitance as a lumped impedance boundary instead
      (see ``PalaceSimMixin.set_pn_junction``).

    With ``mode="auto"`` the choice falls out of
    :func:`select_junction_mode`: the strip is
    meshed only when ``W >= mode_fraction * min(P flank, N flank)``, where
    each flank is ``rib_width / 2``.

    Args:
        comp: gdsfactory component the rectangles are added to.
        length: Rectangle length along the propagation direction (um).
        center_y: Y coordinate of the metallurgical junction / rib centre.
        rib_width: Full rib width (um); P occupies the upper half, N the
            lower half.
        junction: Depletion-model parameters
            (:class:`PNJunctionConfig` or its dict form).
        p_region: ``(name, gds_layer, sigma_S_per_m)`` for the P region.
        n_region: ``(name, gds_layer, sigma_S_per_m)`` for the N region.
        junction_region: ``(name, gds_layer)`` used to register the
            depletion strip in high-res mode. Required when the selected
            mode is ``"high_res"``; ignored in capacitance mode.
        zmin: Bottom z of the regions (um).
        zmax: Top z of the regions (um); defaults to ``zmin + 0.22``.
        fmax: Upper frequency of the Drude-model validity range (Hz).
        mode: ``"auto"``, ``"capacitance"`` or ``"high_res"``.
        mode_fraction: Auto-mode threshold fraction (~1/5 default).
        mesh_resolution: Mesh resolution assigned to the generated layers.

    Returns:
        Dict with keys:

        - ``layer_specs``: ``{name: Layer}`` for every drawn region.
        - ``materials``: ``{name: MaterialProperties}`` (Drude models for
          P/N, plain dielectric for the junction strip).
        - ``centres``: ``{role: y_centre}`` for drawn regions.
        - ``junction``: computed quantities (widths, capacitance, chosen
          mode and selection reason).
    """
    from gsim.common.stack.extractor import Layer

    cfg = _as_junction_config(junction)
    p_name, p_layer, p_sigma = p_region
    n_name, n_layer, n_sigma = n_region

    ztop = 0.22 if zmax is None else zmax
    if ztop <= zmin:
        raise ValueError("zmax must exceed zmin.")
    if length <= 0:
        raise ValueError("length must be positive.")
    if cfg.xp_um + cfg.xn_um > rib_width:
        raise ValueError(
            f"Depletion width W={cfg.w_um:.4g} um does not fit in the "
            f"{rib_width:.4g} um rib."
        )

    flank_um = rib_width / 2
    if mode == "auto":
        mode = select_junction_mode(
            cfg.w_um, flank_um, flank_um, fraction=mode_fraction
        )
        reason = (
            f"W={cfg.w_um:.4g} um vs threshold "
            f"{mode_fraction * flank_um:.4g} um (= {mode_fraction} * flank)"
        )
    else:
        reason = f"forced by caller (mode={mode!r})"
    logger.info("PN junction mode: %s (%s)", mode, reason)

    result: dict[str, dict[str, Any]] = {
        "layer_specs": {},
        "materials": {},
        "centres": {},
    }
    layer_specs = cast("dict[str, Layer]", result["layer_specs"])
    materials: dict[str, Any] = result["materials"]
    centres: dict[str, float] = result["centres"]

    def _doped_spec(name: str, gds_layer: tuple[int, int], _sigma: float) -> Layer:
        return Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=zmin,
            zmax=ztop,
            thickness=ztop - zmin,
            material=name,
            layer_type="dielectric",
            mesh_resolution=mesh_resolution,
        )

    xp, xn = cfg.xp_um, cfg.xn_um

    # N region: lower half, trimmed by xn when the strip is meshed.
    n_y0 = center_y - flank_um
    n_y1 = center_y if mode == "capacitance" else center_y - xn
    centres["n"] = _add_rect(
        comp, length=length, y0=n_y0, y1=n_y1, gds_layer=tuple(n_layer)
    )
    layer_specs[n_name] = _doped_spec(n_name, tuple(n_layer), n_sigma)

    # P region: upper half, trimmed by xp when the strip is meshed.
    p_y0 = center_y if mode == "capacitance" else center_y + xp
    p_y1 = center_y + flank_um
    centres["p"] = _add_rect(
        comp, length=length, y0=p_y0, y1=p_y1, gds_layer=tuple(p_layer)
    )
    layer_specs[p_name] = _doped_spec(p_name, tuple(p_layer), p_sigma)

    materials.update(
        make_doped_materials(
            [(p_name, p_sigma), (n_name, n_sigma)],
            permittivity=cfg.permittivity,
            fmax=fmax,
            source_prefix="doped Si",
        )
    )

    if mode == "high_res":
        if junction_region is None:
            raise ValueError(
                "mode='high_res' requires junction_region=(name, gds_layer)."
            )
        j_name, j_layer = junction_region
        centres["junction"] = _add_rect(
            comp,
            length=length,
            y0=center_y - xn,
            y1=center_y + xp,
            gds_layer=tuple(j_layer),
        )
        layer_specs[j_name] = Layer(
            name=j_name,
            gds_layer=tuple(j_layer),
            zmin=zmin,
            zmax=ztop,
            thickness=ztop - zmin,
            material=j_name,
            layer_type="dielectric",
            mesh_resolution=mesh_resolution,
        )
        # Depleted silicon has no free carriers: pure real permittivity.
        materials[j_name] = MaterialProperties(
            permittivity=cfg.permittivity,
            dispersion_models=[],
        )

    result["junction"] = {
        **cfg.to_metadata(),
        "c_f": cfg.capacitance(length, ztop - zmin),
        "mode": mode,
        "selection_reason": reason,
    }
    return result


# ---------------------------------------------------------------------------
# Free-carrier plasma dispersion (1D complex permittivity).
#
# Minimal extraction of the optical-properties model from the
# ``semiconductor.ipynb`` notebook: the Sze depletion widths locate the
# quasi-neutral and depleted slices of a 1D cut, and the Drude-Sommerfeld
# conductivity of each carrier population gives the local complex
# permittivity at optical frequencies.
# ---------------------------------------------------------------------------

#: Conduction-band effective mass in units of the free-electron mass
#: (Ioffe NSM silicon band structure).
M_CE_STAR: float = 0.26

#: Valence-band (heavy-hole) effective mass in units of ``m0``.
M_CH_STAR: float = 0.38

#: Default electron mobility in cm^2/(V s) (Sze, ``N_I = 2e16`` cm^-3).
MU_N_CM2_VS: float = 1000.0

#: Default hole mobility in cm^2/(V s) (Sze, ``N_I = 2e16`` cm^-3).
MU_P_CM2_VS: float = 450.0

#: Optical conductivities below this (S/m) are treated as negligible and
#: mapped to ``None`` so depleted strips stay pure dielectrics.
SIGMA_NEGLIGIBLE_SM: float = 1e-6


def carrier_profile_1d(
    y_um: np.ndarray | list[float] | float,
    *,
    center_um: float,
    xp_um: float,
    xn_um: float,
    na_cm3: float,
    nd_cm3: float,
    ni_cm3: float = NI_SI_300K_CM3,
) -> tuple[np.ndarray, np.ndarray]:
    """Free-carrier densities along a 1D cut through an abrupt PN junction.

    Depletion approximation: outside ``[center - xn, center + xp]`` each
    side is quasi-neutral with the local-equilibrium densities for the net
    doping ``C`` (``n0 = (C + sqrt(C^2 + 4 ni^2))/2``,
    ``p0 = (-C + sqrt(C^2 + 4 ni^2))/2`` with ``C = +Nd`` / ``-Na``,
    evaluated in the ``n*p = ni^2`` form for float64 safety);
    inside the strip the carriers are swept out (``C = 0`` gives ``ni``).
    This is the 1D counterpart of the notebook's ``C_Sze`` charge profile.

    Args:
        y_um: Sample positions in micrometers (scalar, list or array).
        center_um: Metallurgical-junction position in micrometers.
        xp_um: Depletion extent into the P side (``>= 0``).
        xn_um: Depletion extent into the N side (``>= 0``).
        na_cm3: Acceptor concentration on the P side in cm^-3 (> 0).
        nd_cm3: Donor concentration on the N side in cm^-3 (> 0).
        ni_cm3: Intrinsic carrier concentration in cm^-3 (> 0).

    Returns:
        ``(n_cm3, p_cm3)`` electron/hole density arrays in cm^-3 with the
        broadcast shape of ``y_um``.

    Raises:
        ValueError: On non-positive concentrations or negative extents.
    """
    if na_cm3 <= 0 or nd_cm3 <= 0 or ni_cm3 <= 0:
        raise ValueError("Doping and intrinsic concentrations must be positive.")
    if xp_um < 0 or xn_um < 0:
        raise ValueError("Depletion extents must be non-negative.")
    y = np.asarray(y_um, dtype=float)
    net = np.zeros_like(y)
    net[y >= center_um + xp_um] = -na_cm3
    net[y <= center_um - xn_um] = nd_cm3
    # Majority density first, minority via n*p = ni^2: the direct
    # (-C + sqrt(C^2 + 4 ni^2))/2 form cancels catastrophically in float64
    # when |C| >> ni (ulp(1e18) ~ 128 vs minority values ~1e2).
    disc = np.sqrt(net**2 + 4.0 * ni_cm3**2)
    majority = (np.abs(net) + disc) / 2.0
    minority = ni_cm3**2 / majority
    n = np.where(net >= 0.0, majority, minority)
    p = np.where(net >= 0.0, minority, majority)
    return n, p


def drude_relaxation_times(
    mu_n_cm2_vs: float = MU_N_CM2_VS,
    mu_p_cm2_vs: float = MU_P_CM2_VS,
) -> tuple[float, float]:
    """Drude momentum-relaxation times from mobilities (``tau = m* mu / q``).

    Args:
        mu_n_cm2_vs: Electron mobility in cm^2/(V s) (> 0).
        mu_p_cm2_vs: Hole mobility in cm^2/(V s) (> 0).

    Returns:
        ``(tau_e_s, tau_h_s)`` relaxation times in seconds.

    Raises:
        ValueError: On non-positive mobilities.
    """
    if mu_n_cm2_vs <= 0 or mu_p_cm2_vs <= 0:
        raise ValueError("Mobilities must be positive.")
    tau_e = M_CE_STAR * M0 * (mu_n_cm2_vs * 1e-4) / Q
    tau_h = M_CH_STAR * M0 * (mu_p_cm2_vs * 1e-4) / Q
    return float(tau_e), float(tau_h)


def _optical_omega(wavelength_um: float) -> float:
    """Angular frequency in rad/s for a vacuum wavelength in micrometers."""
    if wavelength_um <= 0:
        raise ValueError("wavelength_um must be positive.")
    return 2.0 * math.pi * C_LIGHT / (wavelength_um * 1e-6)


def epsilon_eff_relative(
    n_cm3: np.ndarray | list[float] | float,
    p_cm3: np.ndarray | list[float] | float,
    *,
    wavelength_um: float,
    eps_bg_rel: float,
    mu_n_cm2_vs: float = MU_N_CM2_VS,
    mu_p_cm2_vs: float = MU_P_CM2_VS,
    tau_e_s: float | None = None,
    tau_h_s: float | None = None,
) -> np.ndarray:
    """Complex relative permittivity from free-carrier plasma dispersion.

    Full Drude-Sommerfeld form (notebook ``effective_eps``, SI-corrected)::

        eps_r = eps_bg - [n q mu_n / (tau_e eps0) (1 - j/(w tau_e))
                        + p q mu_p / (tau_h eps0) (1 - j/(w tau_h))] / w^2

    with densities converted from cm^-3 to m^-3. At 1550 nm
    ``w tau >> 1`` (relaxation regime), so the real part carries the
    plasma shift and the imaginary part the free-carrier absorption.

    Args:
        n_cm3: Electron density in cm^-3 (scalar or array).
        p_cm3: Hole density in cm^-3 (scalar or array, broadcastable).
        wavelength_um: Optical wavelength in micrometers (> 0).
        eps_bg_rel: Relative permittivity of the undoped lattice at the
            target wavelength (e.g. Si Sellmeier, see
            :func:`default_eps_bg_rel`).
        mu_n_cm2_vs: Electron mobility in cm^2/(V s).
        mu_p_cm2_vs: Hole mobility in cm^2/(V s).
        tau_e_s: Electron relaxation time in s (derived from ``mu_n``
            when omitted).
        tau_h_s: Hole relaxation time in s (derived from ``mu_p``
            when omitted).

    Returns:
        Complex relative-permittivity array.

    Raises:
        ValueError: On non-positive wavelength or background permittivity.
    """
    if eps_bg_rel < 1.0:
        raise ValueError("eps_bg_rel must be >= 1.")
    if tau_e_s is None or tau_h_s is None:
        tau_e_d, tau_h_d = drude_relaxation_times(mu_n_cm2_vs, mu_p_cm2_vs)
        tau_e_s = tau_e_d if tau_e_s is None else tau_e_s
        tau_h_s = tau_h_d if tau_h_s is None else tau_h_s
    if tau_e_s <= 0 or tau_h_s <= 0:
        raise ValueError("Relaxation times must be positive.")
    omega = _optical_omega(wavelength_um)
    n_m3 = np.asarray(n_cm3, dtype=float) * 1e6
    p_m3 = np.asarray(p_cm3, dtype=float) * 1e6
    mu_n_si = mu_n_cm2_vs * 1e-4
    mu_p_si = mu_p_cm2_vs * 1e-4
    shift = (
        n_m3 * Q * mu_n_si / tau_e_s * (1.0 - 1j / (omega * tau_e_s))
        + p_m3 * Q * mu_p_si / tau_h_s * (1.0 - 1j / (omega * tau_h_s))
    ) / (EPS0 * omega**2)
    return eps_bg_rel - shift


def optical_params(
    eps_rel: np.ndarray | list[float] | complex,
    wavelength_um: float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Split a complex relative permittivity for Palace material entry.

    Palace carries a real ``Permittivity`` plus a ``Conductivity``, so
    ``eps''`` is mapped through ``sigma = omega eps0 eps''``.

    Args:
        eps_rel: Complex relative permittivity (scalar or array).
        wavelength_um: Optical wavelength in micrometers (> 0).

    Returns:
        ``(eps_prime, sigma_Sm)`` real part and conductivity in S/m
        (Python floats for scalar input, arrays otherwise).
    """
    omega = _optical_omega(wavelength_um)
    eps = np.asarray(eps_rel, dtype=complex)
    prime = np.real(eps)
    sigma = omega * EPS0 * np.imag(eps)
    if eps.ndim == 0:
        return float(prime), float(sigma)
    return prime, sigma


def refractive_index(
    eps_rel: np.ndarray | list[float] | complex,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Refractive index and extinction coefficient from ``n + jk = sqrt(eps)``.

    Args:
        eps_rel: Complex relative permittivity (scalar or array).

    Returns:
        ``(n, k)`` index and extinction (floats for scalar input).
    """
    m = np.sqrt(np.asarray(eps_rel, dtype=complex))
    if m.ndim == 0:
        return float(np.real(m)), float(np.imag(m))
    return np.real(m), np.imag(m)


def default_eps_bg_rel(
    wavelength_um: float,
    material: str = "silicon",
) -> float:
    """Lattice background permittivity for the Drude model at ``wavelength``.

    Resolves the undoped ``material`` dispersion model (e.g. Si Sellmeier)
    at the target wavelength; falls back to
    :data:`DEFAULT_SI_PERMITTIVITY` when the database has no value.

    Args:
        wavelength_um: Optical wavelength in micrometers (> 0).
        material: Undoped material name in the materials database.

    Returns:
        Relative background permittivity as a float.
    """
    from gsim.common.stack.materials import resolve_material_at_wavelength

    resolved = resolve_material_at_wavelength(material, wavelength_um)
    if resolved is not None and resolved.permittivity_scalar is not None:
        return float(resolved.permittivity_scalar)
    return DEFAULT_SI_PERMITTIVITY


def junction_epsilon_profile(
    y_um: np.ndarray | list[float],
    junction: PNJunctionConfig | dict[str, Any],
    *,
    center_um: float = 0.0,
    wavelength_um: float = 1.55,
    eps_bg_rel: float | None = None,
    mu_n_cm2_vs: float = MU_N_CM2_VS,
    mu_p_cm2_vs: float = MU_P_CM2_VS,
) -> dict[str, Any]:
    """1D Sze-based complex permittivity across a PN junction.

    Combines the depletion model (:class:`PNJunctionConfig` gives
    ``xp``/``xn``) with the 1D carrier profile and Drude dispersion, so a
    single call maps positions to the complex permittivity that each
    slice of a segmented waveguide should carry.

    Args:
        y_um: Sample positions in micrometers (list or array).
        junction: Depletion-model parameters (config or its dict form).
        center_um: Metallurgical-junction position in micrometers.
        wavelength_um: Optical wavelength in micrometers (> 0).
        eps_bg_rel: Lattice background (resolved from the Si Sellmeier
            model at ``wavelength_um`` when omitted).
        mu_n_cm2_vs: Electron mobility in cm^2/(V s).
        mu_p_cm2_vs: Hole mobility in cm^2/(V s).

    Returns:
        Dict with ``y_um``, ``n_cm3``, ``p_cm3``, ``eps_rel``,
        ``eps_prime``, ``sigma_Sm``, ``n_index``, ``k_index``,
        ``eps_bg_rel`` and the ``junction`` metadata.
    """
    cfg = _as_junction_config(junction)
    y = np.asarray(y_um, dtype=float)
    n, p = carrier_profile_1d(
        y,
        center_um=center_um,
        xp_um=cfg.xp_um,
        xn_um=cfg.xn_um,
        na_cm3=cfg.na_cm3,
        nd_cm3=cfg.nd_cm3,
        ni_cm3=cfg.ni_cm3,
    )
    bg = default_eps_bg_rel(wavelength_um) if eps_bg_rel is None else eps_bg_rel
    eps = epsilon_eff_relative(
        n,
        p,
        wavelength_um=wavelength_um,
        eps_bg_rel=bg,
        mu_n_cm2_vs=mu_n_cm2_vs,
        mu_p_cm2_vs=mu_p_cm2_vs,
    )
    prime, sigma = optical_params(eps, wavelength_um)
    n_index, k_index = refractive_index(eps)
    return {
        "y_um": y,
        "n_cm3": n,
        "p_cm3": p,
        "eps_rel": eps,
        "eps_prime": prime,
        "sigma_Sm": sigma,
        "n_index": n_index,
        "k_index": k_index,
        "eps_bg_rel": bg,
        "junction": cfg.to_metadata(),
    }


# ---------------------------------------------------------------------------
# Segmented-junction geometry (fine bins carrying the 1D permittivity).
# ---------------------------------------------------------------------------


def make_segmented_junction_profile(
    comp: gf.Component,
    *,
    length: float,
    center_y: float,
    rib_width: float,
    junction: PNJunctionConfig | dict[str, Any],
    n_p: int,
    n_n: int,
    wavelength_um: float = 1.55,
    eps_bg_rel: float | None = None,
    mu_n_cm2_vs: float = MU_N_CM2_VS,
    mu_p_cm2_vs: float = MU_P_CM2_VS,
    p_prefix: str = "p_",
    n_prefix: str = "n_",
    p_gds_start: tuple[int, int] = (21, 1),
    n_gds_start: tuple[int, int] = (20, 1),
    zmin: float = 0.0,
    zmax: float | None = None,
    mesh_resolution: str | float = "fine",
) -> dict[str, dict[str, Any]]:
    """Bin the rib into fine strips sampling the 1D Sze permittivity.

    The P half ``[center_y, center_y + rib_width/2]`` is split into ``n_p``
    uniform strips (``p_1`` at the metallurgical junction, ``p_{n_p}`` at
    the rib edge) and the N half mirrored into ``n_n`` strips (``n_1`` at
    the junction). Each strip gets its own GDS layer and material whose
    ``(eps', sigma)`` comes from :func:`junction_epsilon_profile` sampled
    at the strip centre, so an optical eigenmode sees the laterally varying
    free-carrier permittivity instead of a homogeneous body. Strips whose
    centres fall inside the depletion slice sample ``~ni`` and stay pure
    dielectrics (``conductivity=None``).

    Unlike :func:`make_pn_junction_profile` there is no capacitance mode:
    the bins always resolve whatever depletion width the bias point gives.

    Args:
        comp: gdsfactory component the rectangles are added to.
        length: Rectangle length along the propagation direction (um).
        center_y: Y coordinate of the metallurgical junction / rib centre.
        rib_width: Full rib width (um).
        junction: Depletion-model parameters (config or its dict form).
        n_p: Strip count on the P side (>= 1).
        n_n: Strip count on the N side (>= 1).
        wavelength_um: Optical wavelength in micrometers (> 0).
        eps_bg_rel: Lattice background (resolved from the Si Sellmeier
            model at ``wavelength_um`` when omitted).
        mu_n_cm2_vs: Electron mobility in cm^2/(V s).
        mu_p_cm2_vs: Hole mobility in cm^2/(V s).
        p_prefix: Name prefix for P-side strips.
        n_prefix: Name prefix for N-side strips.
        p_gds_start: ``(layer, datatype)`` of ``p_1``; datatype increments
            per strip. Must not collide with other drawn layers.
        n_gds_start: ``(layer, datatype)`` of ``n_1``.
        zmin: Bottom z of the strips (um).
        zmax: Top z of the strips (um); defaults to ``zmin + 0.22``.
        mesh_resolution: Mesh resolution assigned to the layers.

    Returns:
        Dict with keys ``layer_specs`` (``{name: Layer}``),
        ``materials`` (``{name: MaterialProperties}`` with per-strip
        ``permittivity``/``conductivity``), ``centres``
        (``{name: y_centre}``), ``segments`` (per-strip geometry plus
        sampled ``n``/``p``/``eps_prime``/``sigma_Sm``) and ``junction``
        (depletion metadata plus binning parameters).

    Raises:
        ValueError: On invalid counts, geometry, or a depletion width
            wider than the rib.
    """
    from gsim.common.stack.extractor import Layer

    cfg = _as_junction_config(junction)
    if not isinstance(n_p, int) or not isinstance(n_n, int) or n_p < 1 or n_n < 1:
        raise ValueError("n_p and n_n must be positive integers.")
    ztop = 0.22 if zmax is None else zmax
    if ztop <= zmin:
        raise ValueError("zmax must exceed zmin.")
    if length <= 0:
        raise ValueError("length must be positive.")
    if cfg.xp_um + cfg.xn_um > rib_width:
        raise ValueError(
            f"Depletion width W={cfg.w_um:.4g} um does not fit in the "
            f"{rib_width:.4g} um rib."
        )

    half = rib_width / 2.0
    # (name, y0, y1, gds, side) ordered junction-outward on each side.
    p_edges = np.linspace(center_y, center_y + half, n_p + 1)
    n_edges = np.linspace(center_y - half, center_y, n_n + 1)
    strips: list[tuple[str, float, float, tuple[int, int], str]] = [
        (
            f"{p_prefix}{i + 1}",
            float(p_edges[i]),
            float(p_edges[i + 1]),
            (p_gds_start[0], p_gds_start[1] + i),
            "p",
        )
        for i in range(n_p)
    ]
    strips += [
        (
            f"{n_prefix}{i + 1}",
            float(n_edges[n_n - i - 1]),
            float(n_edges[n_n - i]),
            (n_gds_start[0], n_gds_start[1] + i),
            "n",
        )
        for i in range(n_n)
    ]

    centres_y = np.array([(y0 + y1) / 2.0 for _, y0, y1, _, _ in strips])
    prof = junction_epsilon_profile(
        centres_y,
        cfg,
        center_um=center_y,
        wavelength_um=wavelength_um,
        eps_bg_rel=eps_bg_rel,
        mu_n_cm2_vs=mu_n_cm2_vs,
        mu_p_cm2_vs=mu_p_cm2_vs,
    )

    result: dict[str, dict[str, Any]] = {
        "layer_specs": {},
        "materials": {},
        "centres": {},
        "segments": {},
    }
    layer_specs = cast("dict[str, Layer]", result["layer_specs"])
    materials: dict[str, Any] = result["materials"]
    centres: dict[str, float] = result["centres"]
    segments: dict[str, Any] = result["segments"]

    for k, (name, y0, y1, gds_layer, side) in enumerate(strips):
        yc = centres[name] = _add_rect(
            comp, length=length, y0=y0, y1=y1, gds_layer=gds_layer
        )
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=zmin,
            zmax=ztop,
            thickness=ztop - zmin,
            material=name,
            layer_type="dielectric",
            mesh_resolution=mesh_resolution,
        )
        eps_prime = float(prof["eps_prime"][k])
        sigma = float(prof["sigma_Sm"][k])
        materials[name] = MaterialProperties(
            permittivity=eps_prime,
            conductivity=sigma if sigma >= SIGMA_NEGLIGIBLE_SM else None,
        )
        segments[name] = {
            "y0_um": y0,
            "y1_um": y1,
            "yc_um": yc,
            "gds_layer": gds_layer,
            "side": side,
            "n_cm3": float(prof["n_cm3"][k]),
            "p_cm3": float(prof["p_cm3"][k]),
            "eps_prime": eps_prime,
            "sigma_Sm": sigma,
        }

    result["junction"] = {
        **cfg.to_metadata(),
        "mode": "segmented",
        "wavelength_um": wavelength_um,
        "eps_bg_rel": prof["eps_bg_rel"],
        "n_p": n_p,
        "n_n": n_n,
    }
    return result
