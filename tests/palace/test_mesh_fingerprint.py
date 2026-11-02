"""The fingerprint of the inputs a mesh is built from (mesh reuse, gsim#389).

The pure functions need no mesh. The ``mesh()`` tests replace
``generate_mesh`` with a fake, so they need no Gmsh either.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gsim.palace import BoundaryModeSim, DrivenSim, EigenmodeSim, ElectrostaticSim
from gsim.palace._mesh_fingerprint import (
    describe_changes,
    mesh_inputs_fingerprint,
    port_geometry,
)
from gsim.palace.models import PECBlockConfig, SymmetryPlaneConfig
from gsim.palace.models.ports import (
    CPWPortConfig,
    PortConfig,
    TwoTerminalPortConfig,
    WavePortConfig,
)
from tests._electrode_layouts import FULL, component, stack


def _fingerprint(c=None, st=None, ports=(), **kwargs) -> dict[str, str]:
    """The fingerprint of a layout on the generic stack, with nothing else set."""
    return mesh_inputs_fingerprint(
        component=component() if c is None else c,
        stack=stack() if st is None else st,
        airbox_config=kwargs.get("airbox_config", {}),
        pec_blocks=kwargs.get("pec_blocks", []),
        symmetry_planes=kwargs.get("symmetry_planes", []),
        port_configs=list(ports),
    )


def _two_ports() -> list[PortConfig]:
    return [
        PortConfig(name="PLUS", layer="m1", length=1.0),
        PortConfig(name="MINUS", layer="m1", length=1.0),
    ]


# ----- keys and values ---------------------------------------------------------------


def test_fingerprint_has_one_key_per_mesh_input() -> None:
    fp = _fingerprint(ports=_two_ports())

    assert set(fp) == {
        "geometry",
        "component_ports",
        "layer:m1",
        "layer:v1",
        "layer:m2",
        "dielectrics",
        # "air" is in the stack materials but nothing references it
        "material:aluminum",
        "material:tungsten",
        "material:sio2",
        "airbox",
        "pec_blocks",
        "symmetry_plane",
        "port:PLUS",
        "port:MINUS",
    }
    assert all(isinstance(value, str) for value in fp.values())


def test_fingerprint_values() -> None:
    fp = _fingerprint(ports=_two_ports())

    assert len(fp["geometry"]) == 64
    assert json.loads(fp["component_ports"]) == [
        ["PLUS", 2.0, 2.0, 0.0, 1.0],
        ["MINUS", 8.0, 2.0, 0.0, 1.0],
    ]
    assert json.loads(fp["layer:v1"]) == {
        "name": "v1",
        "gds_layer": [3, 0],
        "zmin": 1.0,
        "zmax": 2.0,
        "thickness": 1.0,
        "material": "tungsten",
        "layer_type": "via",
        "sidewall_angle": 0.0,
        "mesh_order": None,
        "via_conductive": True,
    }
    assert "via_conductive" not in json.loads(fp["layer:m1"])
    assert json.loads(fp["dielectrics"]) == [
        {"name": "ox", "zmin": -2.0, "zmax": 5.0, "material": "sio2"}
    ]
    assert json.loads(fp["material:sio2"]) is False
    assert json.loads(fp["airbox"]) == {}
    assert json.loads(fp["pec_blocks"]) == []
    assert json.loads(fp["symmetry_plane"]) == []
    assert json.loads(fp["port:PLUS"]) == [
        ["lumped", "inplane", "m1", None, None, 1.0, 0.0]
    ]


def test_fingerprint_without_a_component() -> None:
    fp = mesh_inputs_fingerprint(
        component=None,
        stack=stack(),
        airbox_config={},
        pec_blocks=[],
        symmetry_planes=[],
        port_configs=[],
    )

    assert fp["geometry"] == "none"
    assert json.loads(fp["component_ports"]) == []


def test_mesh_resolution_and_permittivity_are_left_out() -> None:
    """Element sizing is free after reuse; only the air-like test is kept."""
    st = stack()
    st.layers["m1"].mesh_resolution = "fine"
    st.materials["sio2"]["permittivity"] = 4.1

    assert _fingerprint(st=st) == _fingerprint()


def test_dielectric_at_eps_one_counts_as_air() -> None:
    st = stack()
    st.materials["sio2"]["permittivity"] = 1.0

    fp = _fingerprint(st=st)

    assert json.loads(fp["material:sio2"]) is True


def test_via_without_conductivity_is_not_conductive() -> None:
    st = stack()
    st.materials["tungsten"]["conductivity"] = 0.0

    fp = _fingerprint(st=st)

    assert json.loads(fp["layer:v1"])["via_conductive"] is False


def test_airbox_pec_and_symmetry_enter_whole() -> None:
    fp = _fingerprint(
        airbox_config={"margin_x": 5.0, "material": "air"},
        pec_blocks=[PECBlockConfig(gds_layer=(9, 0), from_layer="m1", to_layer="m2")],
        symmetry_planes=[SymmetryPlaneConfig(axis="x", position=5.0)],
    )

    assert json.loads(fp["airbox"]) == {"margin_x": 5.0, "material": "air"}
    assert json.loads(fp["pec_blocks"]) == [
        {"gds_layer": [9, 0], "from_layer": "m1", "to_layer": "m2"}
    ]
    assert json.loads(fp["symmetry_plane"]) == [
        {"axis": "x", "position": 5.0, "kind": "pmc", "keep": "positive"}
    ]


# ----- port_geometry -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        (
            PortConfig(name="o1", layer="m1", length=2.0, offset=0.5, excited=False),
            ("o1", ("lumped", "inplane", "m1", None, None, 2.0, 0.5)),
        ),
        (
            PortConfig(
                name="feed", from_layer="m1", to_layer="m2", geometry="interlayer"
            ),
            ("feed", ("lumped", "interlayer", None, "m1", "m2", None, 0.0)),
        ),
        (
            CPWPortConfig(name="c1", layer="m2", s_width=10.0, gap_width=6.0),
            ("c1", ("cpw", "m2", 10.0, 6.0, 2.0, 1.0)),
        ),
        (
            WavePortConfig(
                name="w1", layer="m1", z_margin=1.0, modes=(1, 2), active_mode=1
            ),
            ("w1", ("wave", "m1", 1.0, 0.0, False)),
        ),
        (
            TwoTerminalPortConfig(plus_port="PLUS", minus_port="MINUS", layer="m1"),
            ("PLUS", ("two_terminal", "MINUS", "m1")),
        ),
    ],
)
def test_port_geometry_keeps_only_what_shapes_the_port_surface(cfg, expected) -> None:
    assert port_geometry(cfg) == expected


def test_port_settings_leave_the_fingerprint_unchanged() -> None:
    other = [
        PortConfig(
            name="PLUS",
            layer="m1",
            length=1.0,
            excited=False,
            impedance=25.0,
            resistance=10.0,
        ),
        PortConfig(name="MINUS", layer="m1", length=1.0, capacitance=1e-15),
    ]

    assert _fingerprint(ports=other) == _fingerprint(ports=_two_ports())


# ----- F1: deterministic geometry -----------------------------------------------------


def test_component_copy_has_the_same_fingerprint() -> None:
    c = component()

    assert _fingerprint(c=c.copy()) == _fingerprint(c=c)


def test_shape_order_does_not_change_the_fingerprint() -> None:
    reordered = component(boxes=list(reversed(FULL)))

    assert _fingerprint(c=reordered) == _fingerprint(c=component())


def test_moved_shape_changes_only_the_geometry() -> None:
    moved = [("m1", 0, 6, 10, 8) if box[0:3] == ("m1", 0, 6) else box for box in FULL]

    before = _fingerprint()
    after = _fingerprint(c=component(boxes=moved))

    assert {key for key in before if before[key] != after[key]} == {"geometry"}


# ----- F2: ports keyed by name --------------------------------------------------------


def test_redeclaring_a_port_with_add_port_keeps_the_fingerprint() -> None:
    sim = DrivenSim()
    sim.add_port("PLUS", layer="m1", length=1.0)
    sim.add_port("MINUS", layer="m1", length=1.0)
    before = _fingerprint(ports=sim.ports)

    # add_port replaces a same-named port and moves it to the end of the list
    sim.add_port("PLUS", layer="m1", length=1.0, excited=False, impedance=25.0)

    assert [p.name for p in sim.ports] == ["MINUS", "PLUS"]
    assert _fingerprint(ports=sim.ports) == before


def test_port_layer_change_changes_its_entry() -> None:
    ports = _two_ports()
    before = _fingerprint(ports=ports)
    ports[0].layer = "m2"

    after = _fingerprint(ports=ports)

    assert {key for key in before if before[key] != after[key]} == {"port:PLUS"}


# ----- F3: describe_changes -----------------------------------------------------------


def test_describe_changes_of_equal_fingerprints_is_empty() -> None:
    fp = _fingerprint(ports=_two_ports())

    assert describe_changes(fp, dict(fp)) == []


def test_describe_changes_labels_every_key_in_section_order() -> None:
    keys = [
        "port:o1",
        "symmetry_plane",
        "material:sio2",
        "layer:metal1",
        "pec_blocks",
        "airbox",
        "dielectrics",
        "component_ports",
        "geometry",
    ]
    recorded = dict.fromkeys(keys, "a")
    current = dict.fromkeys(keys, "b")

    assert describe_changes(recorded, current) == [
        "geometry (layout polygons)",
        "component ports (names, positions, orientations, widths)",
        "stack layer 'metal1'",
        "stack dielectric regions",
        "material 'sio2' (air-like or not)",
        "airbox (set_airbox)",
        "PEC blocks (add_pec)",
        "symmetry plane",
        "port 'o1' geometry",
    ]


def test_describe_changes_names_added_and_removed_ports() -> None:
    recorded = _fingerprint(ports=_two_ports())
    st = stack()
    st.layers["m2"].zmax = 3.5
    current = _fingerprint(
        st=st,
        ports=[
            PortConfig(name="PLUS", layer="m1", length=2.0),
            PortConfig(name="EXTRA", layer="m2", length=1.0),
        ],
        airbox_config={"margin_x": 5.0},
    )

    assert describe_changes(recorded, current) == [
        "stack layer 'm2'",
        "airbox (set_airbox)",
        "port 'PLUS' geometry",
        "port 'MINUS' (removed)",
        "port 'EXTRA' (added)",
    ]


# ----- recording in mesh() ------------------------------------------------------------


@pytest.fixture
def fake_generate_mesh(monkeypatch, tmp_path):
    """Replace Gmsh meshing with a result that has no groups."""

    def _fake(**_kwargs):
        return SimpleNamespace(
            mesh_path=tmp_path / "palace.msh",
            config_path=None,
            port_info=[],
            mesh_stats={},
            metadata={},
            groups={},
        )

    monkeypatch.setattr("gsim.palace.mesh.generator.generate_mesh", _fake)


def _meshable(sim_class, out_dir):
    """A simulation of the electrode layout that passes validate_config()."""
    sim = sim_class()
    sim.set_output_dir(out_dir)
    sim.set_geometry(component())
    sim.set_stack(stack())
    if sim_class is ElectrostaticSim:
        sim.add_terminal("PLUS", layer="m1")
        sim.add_terminal("MINUS", layer="m2")
    elif sim_class is BoundaryModeSim:
        sim.set_cross_section("x=2")
    else:
        sim.add_port("PLUS", layer="m1", length=1.0)
        sim.add_port("MINUS", layer="m1", length=1.0)
    if sim_class is EigenmodeSim:
        sim.set_eigenmode(target=5e9)
    return sim


@pytest.mark.usefixtures("fake_generate_mesh")
@pytest.mark.parametrize("sim_class", [DrivenSim, EigenmodeSim])
def test_mesh_records_the_fingerprint_on_reuse_capable_sims(
    sim_class, tmp_path
) -> None:
    sim = _meshable(sim_class, tmp_path / "sim")
    sim._mesh_reused = True

    sim.mesh(verbose=False)

    assert sim._mesh_reused is False
    assert sim._mesh_fingerprint is not None
    assert {"geometry", "port:PLUS", "port:MINUS", "layer:m1"} <= set(
        sim._mesh_fingerprint
    )
    assert sim._mesh_fingerprint == sim._current_mesh_fingerprint(sim.stack)


@pytest.mark.usefixtures("fake_generate_mesh")
@pytest.mark.parametrize("sim_class", [ElectrostaticSim, BoundaryModeSim])
def test_mesh_records_nothing_on_other_sims(sim_class, tmp_path) -> None:
    sim = _meshable(sim_class, tmp_path / "sim")

    sim.mesh(verbose=False)

    assert sim.has_mesh
    assert sim._mesh_fingerprint is None
    assert sim._mesh_reused is False
