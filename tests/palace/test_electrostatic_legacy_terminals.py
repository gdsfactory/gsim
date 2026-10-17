"""Pin today's layer-terminal behaviour of ElectrostaticSim (gsim#273).

A terminal given by ``layer=`` takes every conductor shape on that layer, and a
via is attached to a terminal by z-overlap only. These tests assert that by
physical group names, so they keep holding when terminals can also select
connected nets, and layer-only simulations stay as they are.
"""

from __future__ import annotations

import json

import pytest

from tests._electrode_layouts import (
    MESH,
    attribute_names,
    make_sim,
    physical_names,
)


@pytest.fixture(scope="module")
def legacy(tmp_path_factory):
    out = tmp_path_factory.mktemp("legacy") / "es"
    sim = make_sim(out)
    sim.add_terminal("T1", layer="m1")
    sim.add_terminal("T2", layer="m2")
    sim.mesh(**MESH)
    sim.write_config()
    config = json.loads((out / "config.json").read_text())
    return sim, config, out / "palace.msh"


def test_each_terminal_takes_its_whole_layer(legacy) -> None:
    _, config, msh = legacy
    found = attribute_names(config, msh)
    assert found["Terminal"][0] == {"m1_xy", "m1_z", "sio2__v1"}
    assert found["Terminal"][1] == {"m2_xy", "m2_z", "sio2__v1"}


def test_there_is_no_ground_when_every_conductor_layer_is_a_terminal(legacy) -> None:
    _, config, _ = legacy
    assert "Ground" not in config["Boundaries"]


def test_terminal_indices_follow_the_order_added(legacy) -> None:
    _, config, _ = legacy
    assert [t["Index"] for t in config["Boundaries"]["Terminal"]] == [1, 2]


def test_the_mesh_has_one_group_per_layer_and_no_net_groups(legacy) -> None:
    sim, _, msh = legacy
    names = set(physical_names(msh).values())
    assert {"m1_xy", "m1_z", "m2_xy", "m2_z", "sio2__v1"} <= names
    assert not any("@" in name for name in names)
    assert "nets" not in sim._last_mesh_result.groups


def test_no_terminals_file_is_written_for_layer_terminals(legacy) -> None:
    _, _, msh = legacy
    assert not (msh.parent / "terminals.json").exists()
