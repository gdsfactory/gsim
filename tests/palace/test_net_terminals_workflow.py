"""Full workflow of net terminals: mesh, write_config, read config.json (no Palace).

Layout of ``_electrode_layouts``: electrodes PLUS and MINUS on m1, each with an
m2 bar tied by a via, and a loose island net3 on m1.
"""

from __future__ import annotations

import json

import pytest

from tests._electrode_layouts import (
    FULL,
    MESH,
    PORTS,
    A,
    B,
    attribute_names,
    make_sim,
)


def _run(path, boxes, terminals, *, unassigned="ground", ports=PORTS):
    out = path / "es"
    sim = make_sim(out, boxes, ports)
    for name, kwargs in terminals:
        sim.add_terminal(name, **kwargs)
    sim.set_unassigned_conductors(unassigned)
    sim.mesh(**MESH)
    sim.write_config()
    config = json.loads((out / "config.json").read_text())
    return sim, config, out


def _sets(config, out):
    return attribute_names(config, out / "palace.msh")


def _numbers(config) -> tuple[list[set[int]], set[int]]:
    boundaries = config["Boundaries"]
    terminals = [set(t["Attributes"]) for t in boundaries["Terminal"]]
    ground = set(boundaries.get("Ground", {}).get("Attributes", []))
    return terminals, ground


def test_two_electrodes_on_one_layer_are_two_terminals(tmp_path) -> None:
    _, config, out = _run(
        tmp_path, [A, B], [("L", {"net": "PLUS"}), ("R", {"net": "MINUS"})]
    )
    found = _sets(config, out)
    assert found["Terminal"] == [
        {"m1_xy@PLUS", "m1_z@PLUS"},
        {"m1_xy@MINUS", "m1_z@MINUS"},
    ]
    assert found["Ground"] == []
    terminals, _ground = _numbers(config)
    assert terminals[0].isdisjoint(terminals[1])
    assert [t["Index"] for t in config["Boundaries"]["Terminal"]] == [1, 2]


def test_via_connected_electrode_takes_its_via_and_both_layers(tmp_path) -> None:
    _, config, out = _run(
        tmp_path, FULL, [("TOP", {"net": "PLUS"}), ("LEFT", {"net": "MINUS"})]
    )
    found = _sets(config, out)
    assert found["Terminal"][0] == {
        "m1_xy@PLUS",
        "m1_z@PLUS",
        "m2_xy@PLUS",
        "m2_z@PLUS",
        "sio2__v1@PLUS",
    }
    assert found["Terminal"][1] == {
        "m1_xy@MINUS",
        "m1_z@MINUS",
        "m2_xy@MINUS",
        "m2_z@MINUS",
        "sio2__v1@MINUS",
    }
    assert "sio2__v1@PLUS" not in found["Terminal"][1]
    terminals, ground = _numbers(config)
    assert terminals[0].isdisjoint(terminals[1])
    assert ground.isdisjoint(terminals[0] | terminals[1])


def test_the_island_is_ground_by_default(tmp_path) -> None:
    _, config, out = _run(
        tmp_path, FULL, [("TOP", {"net": "PLUS"}), ("LEFT", {"net": "MINUS"})]
    )
    assert _sets(config, out)["Ground"] == [{"m1_xy@net3", "m1_z@net3"}]
    assert len(config["Boundaries"]["Terminal"]) == 2


def test_the_island_is_an_extra_terminal_when_floating(tmp_path) -> None:
    sim, config, out = _run(
        tmp_path,
        FULL,
        [("TOP", {"net": "PLUS"}), ("LEFT", {"net": "MINUS"})],
        unassigned="floating",
    )
    found = _sets(config, out)
    assert len(found["Terminal"]) == 3
    assert found["Terminal"][2] == {"m1_xy@net3", "m1_z@net3"}
    assert found["Ground"] == []
    assert [t.name for t in sim._resolved_terminals] == ["TOP", "LEFT", "net3"]


def test_an_explicit_floating_terminal_is_a_terminal_too(tmp_path) -> None:
    _, config, out = _run(
        tmp_path,
        FULL,
        [
            ("TOP", {"net": "PLUS"}),
            ("LEFT", {"net": "MINUS"}),
            ("SHIELD", {"net": "net3", "floating": True}),
        ],
    )
    assert _sets(config, out)["Terminal"][2] == {"m1_xy@net3", "m1_z@net3"}


def test_region_terminals_select_whole_nets(tmp_path) -> None:
    _, config, out = _run(
        tmp_path,
        FULL,
        [
            ("L", {"region": (7, 0, 11, 5), "layer": "m1"}),
            ("T", {"net": "PLUS"}),
        ],
    )
    found = _sets(config, out)
    assert "sio2__v1@MINUS" in found["Terminal"][0]
    assert "m2_xy@MINUS" in found["Terminal"][0]
    assert "sio2__v1@PLUS" in found["Terminal"][1]


def test_terminals_json_lists_the_terminals(tmp_path) -> None:
    _, _, out = _run(
        tmp_path,
        FULL,
        [("TOP", {"net": "PLUS"}), ("S", {"net": "net3", "floating": True})],
    )
    data = json.loads((out / "terminals.json").read_text())
    assert data == [
        {
            "index": 1,
            "name": "TOP",
            "nets": [0],
            "net_names": ["PLUS"],
            "floating": False,
        },
        {
            "index": 2,
            "name": "S",
            "nets": [2],
            "net_names": ["net3"],
            "floating": True,
        },
    ]


def test_layer_terminals_write_no_terminals_json(tmp_path) -> None:
    _, config, out = _run(
        tmp_path, FULL, [("T1", {"layer": "m1"}), ("T2", {"layer": "m2"})]
    )
    assert not (out / "terminals.json").exists()
    assert len(config["Boundaries"]["Terminal"]) == 2


# ----- switching modes without meshing again -----------------------------------


def test_switching_from_layers_to_nets_needs_a_new_mesh(tmp_path) -> None:
    sim, _, _out = _run(
        tmp_path, FULL, [("T1", {"layer": "m1"}), ("T2", {"layer": "m2"})]
    )
    sim.add_terminal("T1", net="PLUS")
    with pytest.raises(
        ValueError, match=r"one physical group per layer.*mesh\(\) again"
    ):
        sim.write_config()


def test_switching_from_nets_to_layers_needs_a_new_mesh(tmp_path) -> None:
    sim, _, _out = _run(
        tmp_path, FULL, [("TOP", {"net": "PLUS"}), ("LEFT", {"net": "MINUS"})]
    )
    sim.add_terminal("TOP", layer="m1")
    sim.add_terminal("LEFT", layer="m2")
    with pytest.raises(ValueError, match=r"one physical group per net.*mesh\(\) again"):
        sim.write_config()


def test_meshing_again_in_layer_mode_removes_a_stale_terminals_json(tmp_path) -> None:
    sim, _, out = _run(
        tmp_path, FULL, [("TOP", {"net": "PLUS"}), ("LEFT", {"net": "MINUS"})]
    )
    assert (out / "terminals.json").exists()
    sim.add_terminal("TOP", layer="m1")
    sim.add_terminal("LEFT", layer="m2")
    sim.mesh(**MESH)
    sim.write_config()
    assert not (out / "terminals.json").exists()
    assert sim._resolved_terminals is None


def test_a_changed_geometry_after_meshing_is_caught(tmp_path) -> None:
    sim, _, _out = _run(
        tmp_path, FULL, [("TOP", {"net": "PLUS"}), ("LEFT", {"net": "MINUS"})]
    )
    # a port moved to the island renames a net; the mesh still has the old names
    from tests._electrode_layouts import component

    sim.set_geometry(component(FULL, [("PLUS", 2, 2, "m1"), ("MINUS", 5, 6.5, "m1")]))
    sim.add_terminal("LEFT", net="MINUS")
    with pytest.raises(ValueError, match="differ from the nets of the mesh"):
        sim.write_config()
