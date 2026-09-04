"""Doping-profile construction for semiconductor cross-sections.

This module provides solver-agnostic helpers to build contiguous (gapless)
doping regions on both sides of a rib/waveguide and to generate the
corresponding ``Layer`` specs (``gsim.common.stack.extractor``) and
``MaterialProperties`` (``gsim.common.stack.materials``).

All geometry-specific values (layer tuples, naming prefixes, doping widths,
conductivities, z-extents) are caller-supplied — nothing is hardcoded here so
the helpers are reusable across PDKs and processes.

Example:
-------
    >>> import gdsfactory as gf
    >>> from gsim.common.stack.doping import make_doping_profile
    >>> comp = gf.Component()
    >>> result = make_doping_profile(
    ...     comp,
    ...     length=10.0,
    ...     rib_center_y=-20.0,
    ...     rib_width=0.4,
    ...     profile={
    ...         "upper": [(2.0, 2e4), (2.0, 8e4)],
    ...         "lower": [(2.0, 2e4), (2.0, 8e4)],
    ...     },
    ...     sides={
    ...         "upper": {"base_layer": (23, 0), "name_prefix": "pp_slab_", "sign": 1},
    ...         "lower": {
    ...             "base_layer": (24, 0),
    ...             "name_prefix": "npp_slab_",
    ...             "sign": -1,
    ...         },
    ...     },
    ...     zmin=0.0,
    ...     zmax=0.09,
    ... )
    >>> result["layer_specs"]
    >>> result["materials"]
    >>> result["centres"]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

import gdsfactory as gf

from gsim.common.stack.junction import (
    JUNCTION_MODE_FRACTION,
    PNJunctionConfig,
)
from gsim.common.stack.materials import MaterialProperties, make_doped_materials

if TYPE_CHECKING:
    from gsim.common.stack.extractor import Layer

logger = logging.getLogger(__name__)

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
    the P and N halves) comes from :class:`PNJunctionConfig`, which
    implements the textbook abrupt/linearly-graded junction formulas
    (Sze, *Physics of Semiconductor Devices*, ch. 2).

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
    :func:`gsim.common.stack.junction.select_junction_mode`: the strip is
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
    from gsim.common.stack.junction import select_junction_mode

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


__all__ = ["make_doping_profile", "make_pn_junction_profile"]
