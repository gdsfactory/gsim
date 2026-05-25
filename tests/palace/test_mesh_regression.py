"""Mesh generation regression tests for real-world Palace geometries.

Each test meshes a geometry taken directly from the Palace notebooks and
snapshots the physical-group names (exact match) and mesh element counts
(1% tolerance) via the ``mesh_regression`` fixture.  Group names are fully
deterministic (they don't depend on gmsh version or mesh density), so any
labelling regression — wrong layer mapping, missing conductor surface, lost
port — will be caught immediately.  Element counts are platform-variable at
the ~0.2% level due to GMSH non-determinism across OS/hardware.

Generate / refresh reference YAML files:

    uv run pytest tests/palace/test_mesh_regression.py --force-regen

Tests are skipped automatically when ``ihp`` is not installed.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _mesh_snapshot(sim) -> dict:
    """Return group names and mesh counts from _last_mesh_result."""
    result = sim._last_mesh_result
    groups = result.groups
    stats = result.mesh_stats
    return {
        "groups": {
            "volumes": sorted(groups["volumes"].keys()),
            "conductor_surfaces": sorted(groups["conductor_surfaces"].keys()),
            "pec_surfaces": sorted(groups["pec_surfaces"].keys()),
            "port_surfaces": sorted(groups["port_surfaces"].keys()),
            "boundary_surfaces": sorted(groups["boundary_surfaces"].keys()),
        },
        "mesh": {
            "nodes": stats.get("nodes"),
            "elements": stats.get("elements"),
            "tetrahedra": stats.get("tetrahedra"),
            "invalid_elements": stats.get("sicn", {}).get("invalid", 0),
        },
    }


# ---------------------------------------------------------------------------
# 1. CPW Waveport — IHP GSG electrode, wave ports
#    Notebook: nbs/palace_cpw_waveport.ipynb
# ---------------------------------------------------------------------------


def _make_cpw_waveport_sim(tmp_path):
    """Build a CPW waveport DrivenSim without meshing."""
    import gdsfactory as gf
    from ihp import LAYER, PDK

    from gsim.palace import DrivenSim

    PDK.activate()

    @gf.cell
    def _gsg_waveport(
        length: float = 200,
        s_width: float = 20,
        g_width: float = 40,
        gap_width: float = 15,
        layer=LAYER.TopMetal2drawing,
    ) -> gf.Component:
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

    c = _gsg_waveport()
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(c)
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=0, margin_y=0, z_above=100.0, z_below=100.0)
    sim.add_wave_port("o1", layer="topmetal2", max_size=True, mode=1, excited=True)
    sim.add_wave_port("o2", layer="topmetal2", max_size=True, mode=1, excited=False)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=10)
    return sim


@pytest.fixture(scope="module")
def cpw_waveport_sim(tmp_path_factory):
    """Mesh the CPW waveport geometry once per test session."""
    pytest.importorskip("ihp")
    tmp_path = tmp_path_factory.mktemp("waveport")
    sim = _make_cpw_waveport_sim(tmp_path)
    sim.mesh(preset="coarse", margin_x=0, margin_y=50.0)
    return sim


def test_cpw_waveport_mesh(mesh_regression, cpw_waveport_sim):
    mesh_regression.check(_mesh_snapshot(cpw_waveport_sim))


# ---------------------------------------------------------------------------
# 2. CPW Via Transition — IHP multi-layer TM2 -> TM1 -> M5 -> TM1 -> TM2
#    Notebook: nbs/palace_cpw_via.ipynb
# ---------------------------------------------------------------------------

_VIA_RULES = {
    "TopVia2": {"size": 0.9, "spacing": 1.06, "enclosure": 0.5},
    "TopVia1": {"size": 0.42, "spacing": 0.42, "enclosure": 0.42},
}


def _via_pad_width(cols: int, via_type: str) -> float:
    """Return total pad width for a via array including enclosure."""
    rules = _VIA_RULES[via_type]
    pitch = rules["size"] + rules["spacing"]
    return (cols - 1) * pitch + rules["size"] + 2 * rules["enclosure"]


def _make_cpw_via_transition_sim(tmp_path):
    """Build a CPW via-transition DrivenSim without meshing."""
    import gdsfactory as gf
    from ihp import LAYER, PDK

    from gsim.palace import DrivenSim

    PDK.activate()

    via_layers = {
        "TopVia2": LAYER.TopVia2drawing,
        "TopVia1": LAYER.TopVia1drawing,
    }

    def _via_block(cols: int, rows: int, via_type: str) -> gf.Component:
        rules = _VIA_RULES[via_type]
        size = rules["size"]
        pitch = size + rules["spacing"]
        c = gf.Component()
        via = gf.c.rectangle((size, size), layer=via_layers[via_type])
        for col in range(cols):
            for row in range(rows):
                ref = c << via
                ref.move((col * pitch, row * pitch))
        return c

    def _place_via_block(c, vb_comp, x, y_ctr, via_type, rows):
        rules = _VIA_RULES[via_type]
        pitch = rules["size"] + rules["spacing"]
        vb = c << vb_comp
        vb.move(
            (x + rules["enclosure"], y_ctr - ((rows - 1) * pitch + rules["size"]) / 2)
        )

    @gf.cell
    def _gsg_via_transition(
        tm2_length: float = 50,
        tm1_length: float = 30,
        m5_length: float = 80,
        s_width: float = 20,
        g_width: float = 40,
        gap_width: float = 15,
        tv2_cols: int = 2,
        tv2_rows: int = 10,
        tv1_cols: int = 4,
        tv1_rows: int = 10,
    ) -> gf.Component:
        tv2_w = _via_pad_width(tv2_cols, "TopVia2")
        tv1_w = _via_pad_width(tv1_cols, "TopVia1")
        total = 2 * tm2_length + 2 * tv2_w + 2 * tm1_length + 2 * tv1_w + m5_length

        x = -total / 2
        sections = [
            ("tm2_l", tm2_length),
            ("tv2_l", tv2_w),
            ("tm1_l", tm1_length),
            ("tv1_l", tv1_w),
            ("m5", m5_length),
            ("tv1_r", tv1_w),
            ("tm1_r", tm1_length),
            ("tv2_r", tv2_w),
            ("tm2_r", tm2_length),
        ]
        xs: dict[str, float] = {}
        for name, width in sections:
            xs[name] = x
            x += width

        tv2_block = _via_block(tv2_cols, tv2_rows, "TopVia2")
        tv1_block = _via_block(tv1_cols, tv1_rows, "TopVia1")

        TM2 = LAYER.TopMetal2drawing
        TM1 = LAYER.TopMetal1drawing
        M5 = LAYER.Metal5drawing

        traces = [
            (0, s_width),
            (s_width / 2 + gap_width + g_width / 2, g_width),
            (-(s_width / 2 + gap_width + g_width / 2), g_width),
        ]

        c = gf.Component()
        for y_ctr, w in traces:
            yb = y_ctr - w / 2
            (c << gf.c.rectangle((tm2_length, w), layer=TM2)).move((xs["tm2_l"], yb))
            (c << gf.c.rectangle((tv2_w, w), layer=TM2)).move((xs["tv2_l"], yb))
            (c << gf.c.rectangle((tv2_w, w), layer=TM1)).move((xs["tv2_l"], yb))
            _place_via_block(c, tv2_block, xs["tv2_l"], y_ctr, "TopVia2", tv2_rows)
            (c << gf.c.rectangle((tm1_length, w), layer=TM1)).move((xs["tm1_l"], yb))
            (c << gf.c.rectangle((tv1_w, w), layer=TM1)).move((xs["tv1_l"], yb))
            (c << gf.c.rectangle((tv1_w, w), layer=M5)).move((xs["tv1_l"], yb))
            _place_via_block(c, tv1_block, xs["tv1_l"], y_ctr, "TopVia1", tv1_rows)
            (c << gf.c.rectangle((m5_length, w), layer=M5)).move((xs["m5"], yb))
            (c << gf.c.rectangle((tv1_w, w), layer=TM1)).move((xs["tv1_r"], yb))
            (c << gf.c.rectangle((tv1_w, w), layer=M5)).move((xs["tv1_r"], yb))
            _place_via_block(c, tv1_block, xs["tv1_r"], y_ctr, "TopVia1", tv1_rows)
            (c << gf.c.rectangle((tm1_length, w), layer=TM1)).move((xs["tm1_r"], yb))
            (c << gf.c.rectangle((tv2_w, w), layer=TM2)).move((xs["tv2_r"], yb))
            (c << gf.c.rectangle((tv2_w, w), layer=TM1)).move((xs["tv2_r"], yb))
            _place_via_block(c, tv2_block, xs["tv2_r"], y_ctr, "TopVia2", tv2_rows)
            (c << gf.c.rectangle((tm2_length, w), layer=TM2)).move((xs["tm2_r"], yb))

        c.add_port(
            name="o1",
            center=(-total / 2, 0),
            width=s_width,
            orientation=180,
            port_type="electrical",
            layer=TM2,
        )
        c.add_port(
            name="o2",
            center=(total / 2, 0),
            width=s_width,
            orientation=0,
            port_type="electrical",
            layer=TM2,
        )
        return c

    comp = _gsg_via_transition()
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(comp)
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=0, margin_y=0, z_above=300.0, z_below=300.0)
    sim.add_cpw_port("o1", layer="topmetal2", s_width=20, gap_width=15)
    sim.add_cpw_port("o2", layer="topmetal2", s_width=20, gap_width=15)
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=10)
    return sim


@pytest.fixture(scope="module")
def cpw_via_transition_sim(tmp_path_factory):
    """Mesh the CPW via-transition geometry once per test session."""
    pytest.importorskip("ihp")
    tmp_path = tmp_path_factory.mktemp("via_transition")
    sim = _make_cpw_via_transition_sim(tmp_path)
    sim.mesh(preset="coarse", margin_x=0, margin_y=0, planar_conductors=False)
    return sim


def test_cpw_via_transition_mesh(mesh_regression, cpw_via_transition_sim):
    mesh_regression.check(_mesh_snapshot(cpw_via_transition_sim))


# ---------------------------------------------------------------------------
# 3. Microstrip — IHP straight_metal with Metal1 ground, via ports
#    Notebook: nbs/palace_microstrip.ipynb
# ---------------------------------------------------------------------------


def _make_microstrip_sim(tmp_path):
    """Build a microstrip DrivenSim without meshing."""
    import gdsfactory as gf
    from ihp import LAYER, PDK, cells

    from gsim.palace import DrivenSim

    PDK.activate()

    c = gf.Component()
    r1 = c << cells.straight_metal(length=200, width=14)
    r = c.get_region(layer=LAYER.TopMetal2drawing)
    r_sized = r.sized(+20000)
    c.add_polygon(r_sized, layer=LAYER.Metal1drawing)
    c.add_ports(r1.ports)

    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(c)
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=0, margin_y=0, z_above=300.0, z_below=300.0)
    for port in c.ports:
        assert port.name is not None
        sim.add_port(
            port.name, from_layer="metal1", to_layer="topmetal2", geometry="via"
        )
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=10)
    return sim


@pytest.fixture(scope="module")
def microstrip_sim(tmp_path_factory):
    """Mesh the microstrip geometry once per test session."""
    pytest.importorskip("ihp")
    tmp_path = tmp_path_factory.mktemp("microstrip")
    sim = _make_microstrip_sim(tmp_path)
    sim.mesh(preset="coarse")
    return sim


def test_microstrip_mesh(mesh_regression, microstrip_sim):
    mesh_regression.check(_mesh_snapshot(microstrip_sim))


# ---------------------------------------------------------------------------
# 4. Branch-line coupler — IHP, 4-port, Metal3 ground + TM2 signal, via ports
#    Cell inlined from https://github.com/gdsfactory/IHP/pull/99 (not yet in
#    ihp-gdsfactory 0.2.8).
# ---------------------------------------------------------------------------


def _make_branch_line_coupler_sim(tmp_path):
    """Build a branch-line coupler DrivenSim without meshing."""
    import gdsfactory as gf
    from ihp import LAYER, PDK

    from gsim.palace import DrivenSim

    PDK.activate()

    signal_cs = "topmetal2_routing"
    ground_cs = "metal3_routing"
    signal_layer = gf.get_cross_section(signal_cs).layer

    @gf.cell
    def _tline1(
        length: float = 100,
        width: float = 14,
    ) -> gf.Component:
        c = gf.Component()
        signal = c.add_ref(
            gf.c.straight(length=length, cross_section=signal_cs, width=width)
        )
        c.add_ports(signal.ports)
        ground = c.add_ref(
            gf.c.straight(
                length=length + 6 * width, cross_section=ground_cs, width=7 * width
            )
        )
        ground.move((-3 * width, 0))
        return c

    @gf.cell
    def _branch_line_coupler(
        width: float = 8.85,
        width_coupled: float = 14.96,
        quarter_wave_length: float = 200,
        connection_length: float = 50,
    ) -> gf.Component:
        c = gf.Component()
        corner = gf.Component()
        corner.add_polygon(
            points=[
                (0, 0),
                (0, width),
                (width - (width_coupled - width), width),
                (width, width_coupled),
                (width, 0),
            ],
            layer=signal_layer,
        )
        corner.add_port(
            name="e1",
            center=(width / 2, 0),
            width=width,
            orientation=270,
            port_type="electrical",
            layer=signal_layer,
        )
        corner.add_port(
            name="e2",
            center=(width, width_coupled / 2),
            width=width_coupled,
            orientation=0,
            port_type="electrical",
            layer=signal_layer,
        )
        corner.add_port(
            name="e3",
            center=(0, width / 2),
            width=width,
            orientation=180,
            port_type="electrical",
            layer=signal_layer,
        )

        corner_nw = c.add_ref(corner)
        tline_top = c.add_ref(
            _tline1(length=quarter_wave_length - width, width=width_coupled)
        )
        tline_top.connect("e1", corner_nw.ports["e2"])
        corner_ne = c.add_ref(corner).mirror(p1=(0, 0), p2=(0, 1))
        corner_ne.connect("e2", tline_top.ports["e2"])
        tline_left = c.add_ref(
            _tline1(length=quarter_wave_length - width_coupled, width=width)
        )
        tline_left.connect("e1", corner_nw.ports["e1"])
        corner_sw = c.add_ref(corner).mirror(p1=(0, 0), p2=(1, 0))
        corner_sw.connect("e1", tline_left.ports["e2"])
        tline_bottom = c.add_ref(
            _tline1(length=quarter_wave_length - width, width=width_coupled)
        )
        tline_bottom.connect("e1", corner_sw.ports["e2"])
        corner_se = (
            c.add_ref(corner).mirror(p1=(0, 0), p2=(1, 0)).mirror(p1=(0, 0), p2=(0, 1))
        )
        corner_se.connect("e2", tline_bottom.ports["e2"])
        tline_right = c.add_ref(
            _tline1(length=quarter_wave_length - width_coupled, width=width)
        )
        tline_right.connect("e1", corner_ne.ports["e1"])

        for port, name in [
            (corner_nw.ports["e3"], "e1"),
            (corner_ne.ports["e3"], "e2"),
            (corner_se.ports["e3"], "e3"),
            (corner_sw.ports["e3"], "e4"),
        ]:
            feed = c.add_ref(_tline1(length=connection_length, width=width))
            feed.connect("e1", port)
            c.add_port(name=name, port=feed.ports["e2"])

        c.move((0, -width))
        return c

    c = gf.Component()
    ref = c << _branch_line_coupler()
    c.add_ports(ref.ports)
    ports = [(p.name, p.center, p.width, p.orientation, p.layer) for p in c.ports]
    c.flatten()
    comp = c

    r = comp.get_region(layer=LAYER.Metal3drawing)
    r_filled = gf.kdb.Region(
        [gf.kdb.Polygon(list(p.each_point_hull())) for p in r.each()]
    )
    comp.remove_layers(layers=[LAYER.Metal3drawing])
    comp.add_polygon(r_filled, layer=LAYER.Metal3drawing)

    for name, center, width, orientation, layer in ports:
        comp.add_port(
            name=name, center=center, width=width, orientation=orientation, layer=layer
        )

    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(comp)
    sim.set_stack(substrate_thickness=2.0)
    sim.set_airbox(margin_x=0, margin_y=0, z_above=300.0, z_below=300.0)
    for port in comp.ports:
        assert port.name is not None
        sim.add_port(
            port.name, from_layer="metal3", to_layer="topmetal2", geometry="via"
        )
    sim.set_driven(fmin=1e9, fmax=100e9, num_points=10)
    return sim


@pytest.fixture(scope="module")
def branch_line_coupler_sim(tmp_path_factory):
    """Mesh the branch-line coupler geometry once per test session."""
    pytest.importorskip("ihp")
    tmp_path = tmp_path_factory.mktemp("branch_coupler")
    sim = _make_branch_line_coupler_sim(tmp_path)
    sim.mesh(preset="coarse")
    return sim


def test_branch_line_coupler_mesh(mesh_regression, branch_line_coupler_sim):
    mesh_regression.check(_mesh_snapshot(branch_line_coupler_sim))
