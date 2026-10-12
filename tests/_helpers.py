"""Analytic fixtures shared across test packages."""

from __future__ import annotations


def draw_pn_rib(
    *,
    center_y: float = -20.0,
    rib_width: float = 0.4,
    zmax: float = 0.22,
    sigma_s_per_m: float = 1.6e3,
):
    """A rib whose two halves are the touching Regions ``n_rib`` | ``p_rib``.

    The smallest device with a metallurgical Junction on it, drawn by
    hand on a doped cross-section stack: the n half on the low side of
    ``center_y``, the p half on the high side, on a 90 nm slab.

    Returns:
        ``(component, stack)``.
    """
    import gdsfactory as gf

    from gsim.common.cross_section import build_doped_cross_section
    from gsim.common.stack.extractor import Layer
    from gsim.common.stack.materials import make_doped_materials

    gf.gpdk.PDK.activate()
    comp = gf.Component()
    wg = comp << gf.c.rectangle((10.0, rib_width), centered=True, layer=(1, 0))
    wg.y = center_y
    slab = comp << gf.c.rectangle((10.0, 100.0), centered=True, layer=(3, 0))
    slab.y = -5.0

    halves = {
        "n_rib": ((20, 0), (center_y - rib_width / 2, center_y)),
        "p_rib": ((21, 0), (center_y, center_y + rib_width / 2)),
    }
    layer_specs = {}
    for name, (gds_layer, (y0, y1)) in halves.items():
        comp.add_polygon(
            [(-5.0, y0), (5.0, y0), (5.0, y1), (-5.0, y1)], layer=gds_layer
        )
        layer_specs[name] = Layer(
            name=name,
            gds_layer=gds_layer,
            zmin=0.0,
            zmax=zmax,
            thickness=zmax,
            material=name,
            layer_type="dielectric",
            mesh_resolution="fine",
        )
    materials = make_doped_materials(
        [(name, sigma_s_per_m) for name in halves], permittivity=11.9
    )
    stack, _section = build_doped_cross_section(
        comp,
        axis="x",
        value=0.0,
        substrate_thickness=2.0,
        doping={"layer_specs": layer_specs, "materials": materials},
        verbose=False,
    )
    return comp, stack
