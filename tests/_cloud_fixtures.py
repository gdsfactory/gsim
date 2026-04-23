"""Shared fixtures for cloud-based simulation tests (meep + palace).

Each ``make_*`` builder returns a configured simulation object. Callers
layer test-specific state (output dir, etc.) on top.
"""

from __future__ import annotations

from pathlib import Path

import gdsfactory as gf
import pytest


def require_regression_reference(request: pytest.FixtureRequest) -> None:
    """Fail fast if the ``ndarrays_regression`` reference file is missing.

    pytest-regressions' default is to run the test, write the reference,
    then fail. Cloud sims are expensive, so we refuse to run when the
    baseline is missing — pass ``--force-regen`` to generate it deliberately.
    """
    if request.config.getoption("--force-regen", default=False):
        return
    test_file: Path = request.node.path
    func_name = request.node.originalname or request.node.name
    ref_path = test_file.parent / test_file.stem / f"{func_name}.npz"
    if not ref_path.exists():
        rel = ref_path.relative_to(test_file.parent.parent.parent)
        pytest.fail(
            f"Reference file missing: {rel}\n"
            "Refusing to run cloud simulation without a baseline. "
            "Run with --force-regen to generate it."
        )


# ---------------------------------------------------------------------------
# Meep fixtures
# ---------------------------------------------------------------------------


def make_sbend_sim():
    """Tiny S-bend MEEP simulation (3D, 2-port, very low resolution)."""
    from gsim.meep import Simulation

    gf.gpdk.PDK.activate()

    # Small S-bend. bend_s `size` is (length, height). Keep both small
    # so the sim cell stays tiny, but big enough to satisfy the default
    # gpdk strip cross-section min bend radius (3.5 um).
    component = gf.components.bend_s(size=(7.0, 2.0), npoints=41)

    sim = Simulation()
    sim.geometry.component = component
    sim.materials = {"si": 3.47, "SiO2": 1.44}
    sim.source.port = "o1"
    sim.source.wavelength = 1.55
    sim.source.wavelength_span = 0.05
    sim.num_freqs = 5
    sim.monitors = ["o2"]
    sim.domain.pml = 1.0
    sim.domain.margin = 0.5
    sim.solver.resolution = 10
    sim.solver.stop_when_energy_decayed(dt=15.0, decay_by=0.05)
    return sim


def make_2d_xz_gc_sim():
    """Tiny 2D XZ grating-coupler sim with a Gaussian fiber source.

    Mirrors ``nbs/meep_2d_xz_gc.ipynb`` but with the smallest grating
    coupler that still produces a coupled mode and a feed monitor.
    """
    from gsim.meep import Simulation

    gf.gpdk.PDK.activate()

    c = gf.Component()
    gc = gf.components.grating_coupler_elliptical(fiber_angle=0.0)
    gc_r = c.add_ref(gc)
    s_r = c.add_ref(gf.components.straight(length=3.0))
    s_r.connect("o1", gc_r.ports["o1"])
    c.add_port("o2", port=s_r.ports["o2"])

    sim = Simulation()
    sim.geometry.component = c
    sim.materials = {"si": 3.47, "SiO2": 1.44}

    sim.solver.is_3d = False
    sim.solver.plane = "xz"
    sim.solver.resolution = 25
    sim.solver.stop_when_energy_decayed(dt=15.0, decay_by=0.05)

    sim.source_fiber(
        x=25.0,
        z=2.0,
        angle_deg=-6.0,
        waist=5.2,
        wavelength=1.55,
        wavelength_span=0.06,
        polarization="TE",
    )

    sim.monitors = ["o2"]
    sim.domain.pml = 1.0
    sim.domain.margin = 0.5
    sim.num_freqs = 11
    return sim


# ---------------------------------------------------------------------------
# Palace fixtures
# ---------------------------------------------------------------------------


def _gsg_electrode(
    length: float = 200.0,
    s_width: float = 20.0,
    g_width: float = 40.0,
    gap_width: float = 15.0,
):
    """Minimal GSG (ground-signal-ground) CPW electrode for palace tests.

    Defaults mirror the CPW notebooks (shrunk length for fast CI).
    """
    gf.gpdk.PDK.activate()
    layer = gf.gpdk.LAYER.M1

    @gf.cell
    def gsg_electrode_cell() -> gf.Component:
        c = gf.Component()
        r1 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        r1.move((0, (g_width + s_width) / 2 + gap_width))
        c << gf.c.rectangle((length, s_width), centered=True, layer=layer)
        r3 = c << gf.c.rectangle((length, g_width), centered=True, layer=layer)
        r3.move((0, -(g_width + s_width) / 2 - gap_width))
        c.add_port(
            name="o1",
            center=(-length / 2, 0),
            width=s_width,
            orientation=180,
            port_type="electrical",
            layer=layer,
        )
        c.add_port(
            name="o2",
            center=(length / 2, 0),
            width=s_width,
            orientation=0,
            port_type="electrical",
            layer=layer,
        )
        return c

    return gsg_electrode_cell()


def make_driven_cpw_lumped_sim(output_dir: str | Path):
    """DrivenSim with CPW lumped ports on a GSG electrode."""
    from gsim.palace import DrivenSim

    component = _gsg_electrode()

    sim = DrivenSim()
    sim.set_output_dir(str(output_dir))
    sim.set_geometry(component)
    sim.set_stack(substrate_thickness=2.0, air_above=100.0, air_below=100.0)
    sim.add_cpw_port("o1", layer="metal1", s_width=20, gap_width=15, excited=True)
    sim.add_cpw_port("o2", layer="metal1", s_width=20, gap_width=15, excited=False)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=11)
    return sim


def make_driven_cpw_waveport_sim(output_dir: str | Path):
    """DrivenSim with wave ports on a GSG electrode."""
    from gsim.palace import DrivenSim

    component = _gsg_electrode()

    sim = DrivenSim()
    sim.set_output_dir(str(output_dir))
    sim.set_geometry(component)
    sim.set_stack(substrate_thickness=2.0, air_above=100.0, air_below=100.0)
    sim.add_wave_port("o1", layer="metal1", max_size=True, mode=1, excited=True)
    sim.add_wave_port("o2", layer="metal1", max_size=True, mode=1, excited=False)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=11)
    return sim


def _qpdk_resonator_component(coupling_gap: float = 20.0):
    """Build the qpdk coupled resonator used by the eigenmode fixture.

    Matches the geometry used in ``nbs/_palace_qpdk_eigenmode.ipynb``.
    The SIM_AREA - M1_ETCH region arithmetic is then done by the caller
    to produce the conductor/substrate/vacuum layout consumed by Palace.
    """
    import gdsfactory as gf
    from qpdk import PDK, cells
    from qpdk.cells.airbridge import cpw_with_airbridges
    from qpdk.tech import LAYER, route_bundle_sbend_cpw

    PDK.activate()

    @gf.cell
    def resonator_compact() -> gf.Component:
        c = gf.Component()
        res = c << cells.resonator_coupled(
            coupling_straight_length=200,
            coupling_gap=coupling_gap,
            length=1000,
            meanders=2,
        )
        res.movex(-res.size_info.width / 4)

        left = c << cells.straight()
        right = c << cells.straight()
        w = res.size_info.width + 100
        left.move((-w, 0))
        right.move((w, 0))

        route_bundle_sbend_cpw(
            c,
            [left["o2"], right["o1"]],
            [res["coupling_o1"], res["coupling_o2"]],
            cross_section=cpw_with_airbridges(
                airbridge_spacing=250.0, airbridge_padding=20.0
            ),
        )

        sim_area_idx = c.kdb_cell.layout().layer(LAYER.SIM_AREA[0], LAYER.SIM_AREA[1])
        c.kdb_cell.shapes(sim_area_idx).insert(c.bbox().enlarged(0, 100))

        c.add_port(name="o1", port=left["o1"])
        c.add_port(name="o2", port=right["o2"])
        return c

    return resonator_compact()


def _etch_to_conductor(component):
    """Convert a qpdk component's SIM_AREA + M1_ETCH layers into a
    three-layer (SUBSTRATE, SUPERCONDUCTOR, VACUUM) component suitable
    for Palace. Lifted from ``nbs/_palace_qpdk_eigenmode.ipynb``.
    """
    import gdsfactory as gf
    import klayout.db as kdb
    from qpdk.tech import LAYER as QPDK_LAYER

    from gsim.common.polygon_utils import decimate

    sim_area_layer = (QPDK_LAYER.SIM_AREA[0], QPDK_LAYER.SIM_AREA[1])
    etch_layer = (QPDK_LAYER.M1_ETCH[0], QPDK_LAYER.M1_ETCH[1])
    cpw_layers = {"SUBSTRATE": (1, 0), "SUPERCONDUCTOR": (2, 0), "VACUUM": (3, 0)}

    layout = component.kdb_cell.layout()
    sim_region = kdb.Region(
        component.kdb_cell.begin_shapes_rec(layout.layer(*sim_area_layer))
    )
    etch_region_raw = kdb.Region(
        component.kdb_cell.begin_shapes_rec(layout.layer(*etch_layer))
    )
    etch_region = kdb.Region()
    for poly in decimate(list(etch_region_raw.each())):
        etch_region.insert(poly)

    conductor_region = sim_region - etch_region

    etched = gf.Component("etched_component")
    el = etched.kdb_cell.layout()
    for name, region in [
        ("SUPERCONDUCTOR", conductor_region),
        ("SUBSTRATE", sim_region),
        ("VACUUM", sim_region),
    ]:
        idx = el.layer(*cpw_layers[name])
        etched.kdb_cell.shapes(idx).insert(region)

    for port in component.ports:
        etched.add_port(name=port.name, port=port)
    return etched


def _qpdk_cpw_stack(
    substrate_thickness: float = 500.0, vacuum_thickness: float = 500.0
):
    """Three-layer sapphire/aluminum/vacuum stack from the qpdk notebook."""
    from gsim.common.stack import Layer, LayerStack
    from gsim.common.stack.materials import MATERIALS_DB

    stack = LayerStack(pdk_name="qpdk")
    stack.layers["SUBSTRATE"] = Layer(
        name="SUBSTRATE",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=substrate_thickness,
        thickness=substrate_thickness,
        material="sapphire",
        layer_type="dielectric",
    )
    stack.layers["SUPERCONDUCTOR"] = Layer(
        name="SUPERCONDUCTOR",
        gds_layer=(2, 0),
        zmin=substrate_thickness,
        zmax=substrate_thickness,
        thickness=0,
        material="aluminum",
        layer_type="conductor",
    )
    stack.layers["VACUUM"] = Layer(
        name="VACUUM",
        gds_layer=(3, 0),
        zmin=substrate_thickness,
        zmax=substrate_thickness + vacuum_thickness,
        thickness=vacuum_thickness,
        material="vacuum",
        layer_type="dielectric",
    )
    stack.dielectrics = [
        {
            "name": "substrate",
            "zmin": 0.0,
            "zmax": substrate_thickness,
            "material": "sapphire",
        },
        {
            "name": "vacuum",
            "zmin": substrate_thickness,
            "zmax": substrate_thickness + vacuum_thickness,
            "material": "vacuum",
        },
    ]
    stack.materials = {
        "sapphire": MATERIALS_DB["sapphire"].to_dict(),
        "aluminum": MATERIALS_DB["aluminum"].to_dict(),
        "vacuum": MATERIALS_DB["vacuum"].to_dict(),
    }
    return stack


def make_eigenmode_cavity_sim(output_dir: str | Path):
    """EigenmodeSim on a qpdk coupled CPW resonator.

    With ``length=1000`` um (vs the notebook's 4000), the lambda/4
    fundamental on a sapphire CPW shifts from ~8 GHz to ~32 GHz; the
    solver ``target`` is set accordingly.
    """
    pytest.importorskip("qpdk")
    from gsim.palace import EigenmodeSim

    component = _qpdk_resonator_component(coupling_gap=20.0)
    etched = _etch_to_conductor(component)
    stack = _qpdk_cpw_stack()

    sim = EigenmodeSim()
    sim.set_output_dir(str(output_dir))
    sim.set_geometry(etched)
    sim.set_stack(stack)
    sim.add_cpw_port(
        "o1", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5
    )
    sim.add_cpw_port(
        "o2", layer="SUPERCONDUCTOR", s_width=10.0, gap_width=6.0, offset=2.5
    )
    sim.set_eigenmode(num_modes=1, target=32e9, save=1)
    return sim


__all__ = [
    "make_2d_xz_gc_sim",
    "make_driven_cpw_lumped_sim",
    "make_driven_cpw_waveport_sim",
    "make_eigenmode_cavity_sim",
    "make_sbend_sim",
    "require_regression_reference",
]
