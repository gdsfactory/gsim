"""QPDK utilities — etch-to-conductor conversion and CPW layer stack.

QPDK represents conductors as *etching* layers: the metallisation is
defined by the area of ``SIM_AREA`` that is **not** etched.  This module
converts that representation into explicit conductor, substrate and
vacuum layers suitable for meshing and EM simulation.

It also provides :func:`cpw_layer_stack` which builds a complete gsim
:class:`~gsim.common.stack.extractor.LayerStack` for the three-layer
SUBSTRATE / SUPERCONDUCTOR / VACUUM topology used by QPDK components.

Usage::

    from gsim.common.stack.qpdk_utils import cpw_layer_stack, create_etched_component

    layer_stack, CPW_LAYER = cpw_layer_stack(
        substrate_thickness=500, vacuum_thickness=500
    )
    etched = create_etched_component(component, CPW_LAYER)

    sim.set_geometry(etched)
    sim.set_stack(layer_stack)
"""

from __future__ import annotations

import warnings

import gdsfactory as gf
import klayout.db as kdb

from gsim.common.polygon_utils import decimate
from gsim.common.stack.extractor import Layer, LayerStack
from gsim.common.stack.materials import MATERIALS_DB

# ---------------------------------------------------------------------------
# CPW layer stack
# ---------------------------------------------------------------------------

#: Default GDS layer numbers for the three CPW layers.
CPW_LAYERS: dict[str, tuple[int, int]] = {
    "SUBSTRATE": (1, 0),
    "SUPERCONDUCTOR": (2, 0),
    "VACUUM": (3, 0),
}


def cpw_layer_stack(
    conductor_thickness: float = 0,
    substrate_thickness: float = 1500,
    vacuum_thickness: float = 1500,
) -> tuple[LayerStack, dict[str, tuple[int, int]]]:
    """Return a CPW (coplanar-waveguide) layer stack ready for Palace.

    The returned :class:`LayerStack` contains three layers and three
    dielectric regions that match what the Palace mesher expects:

    * **SUBSTRATE** (1, 0) — dielectric below the conductor plane
    * **SUPERCONDUCTOR** (2, 0) — metal plane (zero-thickness by default for
      2-D boundary-element treatment)
    * **VACUUM** (3, 0) — air / vacuum above the conductor

    Args:
        conductor_thickness: Conductor metal thickness (µm).  Usually 0
            for sheet-impedance (PEC) models.
        substrate_thickness: Dielectric thickness (µm).
        vacuum_thickness: Air gap above conductor (µm).

    Returns:
        ``(layer_stack, LAYER)`` — a gsim *LayerStack* and a dict mapping
        layer names to ``(layer, datatype)`` GDS tuples.

    Example::

        layer_stack, CPW = cpw_layer_stack(
            substrate_thickness=500, vacuum_thickness=500
        )
        sim.set_stack(layer_stack)
    """
    LAYER = dict(CPW_LAYERS)  # copy so callers can't mutate the default

    stack = LayerStack(pdk_name="qpdk")

    # --- layers (conductor / via entries read by the mesher) ---------------
    stack.layers["SUBSTRATE"] = Layer(
        name="SUBSTRATE",
        gds_layer=LAYER["SUBSTRATE"],
        zmin=0.0,
        zmax=substrate_thickness,
        thickness=substrate_thickness,
        material="sapphire",
        layer_type="dielectric",
    )

    stack.layers["SUPERCONDUCTOR"] = Layer(
        name="SUPERCONDUCTOR",
        gds_layer=LAYER["SUPERCONDUCTOR"],
        zmin=substrate_thickness,
        zmax=substrate_thickness + conductor_thickness,
        thickness=conductor_thickness,
        material="aluminum",
        layer_type="conductor",
    )

    stack.layers["VACUUM"] = Layer(
        name="VACUUM",
        gds_layer=LAYER["VACUUM"],
        zmin=substrate_thickness + conductor_thickness,
        zmax=substrate_thickness + conductor_thickness + vacuum_thickness,
        thickness=vacuum_thickness,
        material="vacuum",
        layer_type="dielectric",
    )

    # --- dielectric regions (used by the config generator) -----------------
    stack.dielectrics = [
        {
            "name": "substrate",
            "zmin": 0.0,
            "zmax": substrate_thickness,
            "material": "sapphire",
        },
        {
            "name": "vacuum",
            "zmin": substrate_thickness + conductor_thickness,
            "zmax": substrate_thickness + conductor_thickness + vacuum_thickness,
            "material": "vacuum",
        },
    ]

    # --- materials ---------------------------------------------------------
    stack.materials["sapphire"] = MATERIALS_DB["sapphire"].to_dict()
    stack.materials["aluminum"] = MATERIALS_DB["aluminum"].to_dict()
    stack.materials["vacuum"] = MATERIALS_DB["vacuum"].to_dict()

    # --- simulation metadata -----------------------------------------------
    stack.simulation = {
        "boundary_margin": 30.0,
        "air_above": vacuum_thickness,
        "substrate_thickness": substrate_thickness,
        "include_substrate": False,
    }

    return stack, LAYER


def create_etched_component(
    component: gf.Component,
    cpw_layers: dict[str, tuple[int, int]] | None = None,
    *,
    sim_area_layer: tuple[int, int] | None = None,
    etch_layer: tuple[int, int] | None = None,
) -> gf.Component:
    """Boolean-subtract the etch region from the simulation area.

    The resulting component contains three layers taken from *cpw_layers*:

    * ``SUPERCONDUCTOR`` — metallised area (SIM_AREA minus M1_ETCH)
    * ``SUBSTRATE``       — full simulation footprint
    * ``VACUUM``          — same footprint (used for the air volume above)

    Ports from the original *component* are copied over.

    Args:
        component: Source QPDK component (must have SIM_AREA & M1_ETCH layers).
        cpw_layers: Mapping with at least ``SUPERCONDUCTOR``, ``SUBSTRATE`` and
            ``VACUUM`` keys whose values are ``(layer, datatype)`` tuples.
            Defaults to :data:`CPW_LAYERS`.
        sim_area_layer: Override for the QPDK SIM_AREA layer index.
            Defaults to ``LAYER.SIM_AREA`` from ``qpdk.tech``.
        etch_layer: Override for the QPDK M1_ETCH layer index.
            Defaults to ``LAYER.M1_ETCH`` from ``qpdk.tech``.

    Returns:
        New ``gf.Component`` with SUPERCONDUCTOR / SUBSTRATE / VACUUM layers
        and the original ports.
    """
    if cpw_layers is None:
        cpw_layers = dict(CPW_LAYERS)

    # Resolve QPDK layer indices ----------------------------------------
    if sim_area_layer is None or etch_layer is None:
        from qpdk.tech import LAYER as QPDK_LAYER

        if sim_area_layer is None:
            sim_area_layer = (QPDK_LAYER.SIM_AREA[0], QPDK_LAYER.SIM_AREA[1])
        if etch_layer is None:
            etch_layer = (QPDK_LAYER.M1_ETCH[0], QPDK_LAYER.M1_ETCH[1])

    layout = component.kdb_cell.layout()

    sim_idx = layout.layer(sim_area_layer[0], sim_area_layer[1])  # type: ignore[index]
    etch_idx = layout.layer(etch_layer[0], etch_layer[1])  # type: ignore[index]

    sim_region = kdb.Region(component.kdb_cell.begin_shapes_rec(sim_idx))
    etch_region = kdb.Region(component.kdb_cell.begin_shapes_rec(etch_idx))

    # Decimate the (often very dense) etch polygons ----------------------
    etch_polys = decimate(list(etch_region.each()))
    etch_region = kdb.Region()
    for poly in etch_polys:
        etch_region.insert(poly)

    if sim_region.is_empty():
        warnings.warn("No polygons found on SIM_AREA", stacklevel=2)
    if etch_region.is_empty():
        warnings.warn("No polygons found on M1_ETCH", stacklevel=2)

    # Boolean subtraction: conductor = sim_area - etch -------------------
    conductor_region = sim_region - etch_region

    # Build result component ---------------------------------------------
    etched = gf.Component("etched_component")
    el = etched.kdb_cell.layout()

    for name, region in [
        ("SUPERCONDUCTOR", conductor_region),
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
