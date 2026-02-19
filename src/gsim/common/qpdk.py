"""QPDK utilities — etch-to-conductor conversion.

QPDK represents conductors as *etching* layers: the metallisation is
defined by the area of ``SIM_AREA`` that is **not** etched.  This module
converts that representation into explicit conductor, substrate and
vacuum layers suitable for meshing and EM simulation.

Usage::

    from gsim.common.qpdk import create_etched_component

    etched = create_etched_component(component, cpw_layers)
"""

from __future__ import annotations

import gdsfactory as gf
import klayout.db as kdb

from gsim.common.polygon_utils import decimate


def create_etched_component(
    component: gf.Component,
    cpw_layers: dict[str, tuple[int, int]],
    *,
    sim_area_layer: tuple[int, int] | None = None,
    etch_layer: tuple[int, int] | None = None,
) -> gf.Component:
    """Boolean-subtract the etch region from the simulation area.

    The resulting component contains three layers taken from *cpw_layers*:

    * ``CONDUCTOR`` — metallised area (SIM_AREA minus M1_ETCH)
    * ``SUBSTRATE`` — full simulation footprint
    * ``VACUUM``    — same footprint (used for the air volume above)

    Ports from the original *component* are copied over.

    Args:
        component: Source QPDK component (must have SIM_AREA & M1_ETCH layers).
        cpw_layers: Mapping with at least ``CONDUCTOR``, ``SUBSTRATE`` and
            ``VACUUM`` keys whose values are ``(layer, datatype)`` tuples.
        sim_area_layer: Override for the QPDK SIM_AREA layer index.
            Defaults to ``LAYER.SIM_AREA`` from ``qpdk.tech``.
        etch_layer: Override for the QPDK M1_ETCH layer index.
            Defaults to ``LAYER.M1_ETCH`` from ``qpdk.tech``.

    Returns:
        New ``gf.Component`` with CONDUCTOR / SUBSTRATE / VACUUM layers
        and the original ports.
    """
    # Resolve QPDK layer indices ----------------------------------------
    if sim_area_layer is None or etch_layer is None:
        from qpdk.tech import LAYER as QPDK_LAYER  # type: ignore[import-untyped]

        if sim_area_layer is None:
            sim_area_layer = (QPDK_LAYER.SIM_AREA[0], QPDK_LAYER.SIM_AREA[1])
        if etch_layer is None:
            etch_layer = (QPDK_LAYER.M1_ETCH[0], QPDK_LAYER.M1_ETCH[1])

    layout = component.kdb_cell.layout()

    sim_idx = layout.layer(sim_area_layer[0], sim_area_layer[1])
    etch_idx = layout.layer(etch_layer[0], etch_layer[1])

    sim_region = kdb.Region(component.kdb_cell.begin_shapes_rec(sim_idx))
    etch_region = kdb.Region(component.kdb_cell.begin_shapes_rec(etch_idx))

    # Decimate the (often very dense) etch polygons ----------------------
    etch_polys = decimate([p for p in etch_region.each()])
    etch_region = kdb.Region()
    for poly in etch_polys:
        etch_region.insert(poly)

    if sim_region.is_empty():
        print("Warning: no polygons found on SIM_AREA")
    if etch_region.is_empty():
        print("Warning: no polygons found on M1_ETCH")

    # Boolean subtraction: conductor = sim_area − etch -------------------
    conductor_region = sim_region - etch_region

    # Build result component ---------------------------------------------
    etched = gf.Component("etched_component")
    el = etched.kdb_cell.layout()

    for name, region in [
        ("CONDUCTOR", conductor_region),
        ("SUBSTRATE", sim_region),
        ("VACUUM", sim_region),
    ]:
        ly = cpw_layers[name]
        idx = el.layer(ly[0], ly[1])
        etched.kdb_cell.shapes(idx).insert(region)

    # Copy ports ---------------------------------------------------------
    if hasattr(component, "ports"):
        for port in component.ports:
            etched.add_port(name=port.name, port=port)

    return etched
