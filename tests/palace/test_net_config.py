"""Tests for Terminal and Ground entries built from nets (no gmsh, no mesh)."""

from __future__ import annotations

import json

import pytest

from gsim.palace.mesh.config_generator import (
    _electrostatic_net_boundaries,
    generate_palace_config,
)
from gsim.palace.terminals import ResolvedTerminal
from tests._electrode_layouts import stack


def _groups() -> dict:
    """Two via-connected electrodes (nets 0, 1) and an island (net 2).

    Group numbers: PLUS 12 15 18 20 22, MINUS 13 16 19 21 23, island 14 17; 24 is
    a PEC block group that belongs to no net.
    """
    cond = {
        "m1_xy@PLUS": 12,
        "m1_xy@MINUS": 13,
        "m1_xy@net3": 14,
        "m1_z@PLUS": 15,
        "m1_z@MINUS": 16,
        "m1_z@net3": 17,
        "m2_xy@PLUS": 18,
        "m2_xy@MINUS": 19,
        "m2_z@PLUS": 20,
        "m2_z@MINUS": 21,
    }
    return {
        "volumes": {"sio2": {"phys_group": 2, "tags": [1]}},
        "conductor_surfaces": {
            name: {"phys_group": pg, "tags": []} for name, pg in cond.items()
        },
        "pec_surfaces": {"shield_xy": {"phys_group": 24, "tags": []}},
        "via_boundary_surfaces": {"v1@PLUS": [22], "v1@MINUS": [23]},
        "port_surfaces": {},
        "boundary_surfaces": {},
        "nets": [
            {"name": "PLUS", "phys_groups": [12, 15, 18, 20, 22]},
            {"name": "MINUS", "phys_groups": [13, 16, 19, 21, 23]},
            {"name": "net3", "phys_groups": [14, 17]},
        ],
    }


def _t(name: str, nets: tuple[int, ...], floating: bool = False) -> ResolvedTerminal:
    return ResolvedTerminal(name, nets, floating)


def test_terminals_and_ground_are_disjoint_and_cover_all_groups() -> None:
    boundaries = _electrostatic_net_boundaries(
        _groups(), [_t("TOP", (0,)), _t("LEFT", (1,))]
    )
    terminals = boundaries["Terminal"]
    assert terminals == [
        {"Index": 1, "Attributes": [12, 15, 18, 20, 22]},
        {"Index": 2, "Attributes": [13, 16, 19, 21, 23]},
    ]
    ground = boundaries["Ground"]["Attributes"]
    assert ground == [14, 17, 24]  # the island and the PEC block
    everything = {12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}
    used = [a for t in terminals for a in t["Attributes"]] + ground
    assert sorted(used) == sorted(everything)  # no overlap, nothing missing


def test_via_group_is_only_in_its_own_nets_terminal() -> None:
    boundaries = _electrostatic_net_boundaries(
        _groups(), [_t("TOP", (0,)), _t("LEFT", (1,))]
    )
    assert 22 in boundaries["Terminal"][0]["Attributes"]
    assert 22 not in boundaries["Terminal"][1]["Attributes"]
    assert 23 in boundaries["Terminal"][1]["Attributes"]


def test_index_order_follows_the_resolved_order() -> None:
    boundaries = _electrostatic_net_boundaries(
        _groups(), [_t("LEFT", (1,)), _t("TOP", (0,))]
    )
    assert [t["Index"] for t in boundaries["Terminal"]] == [1, 2]
    assert boundaries["Terminal"][0]["Attributes"][0] == 13


def test_floating_terminals_are_terminals_like_the_others() -> None:
    boundaries = _electrostatic_net_boundaries(
        _groups(), [_t("TOP", (0,)), _t("LEFT", (1,)), _t("net3", (2,), True)]
    )
    assert boundaries["Terminal"][2] == {"Index": 3, "Attributes": [14, 17]}
    assert boundaries["Ground"]["Attributes"] == [24]


def test_a_terminal_with_several_nets_takes_the_union() -> None:
    boundaries = _electrostatic_net_boundaries(_groups(), [_t("BOTH", (0, 1))])
    assert boundaries["Terminal"][0]["Attributes"] == [
        12,
        13,
        15,
        16,
        18,
        19,
        20,
        21,
        22,
        23,
    ]


def test_no_ground_entry_when_every_group_is_used() -> None:
    groups = _groups()
    del groups["pec_surfaces"]["shield_xy"]
    boundaries = _electrostatic_net_boundaries(
        groups, [_t("A", (0,)), _t("B", (1,)), _t("C", (2,))]
    )
    assert "Ground" not in boundaries


def test_missing_resolved_terminals_raise() -> None:
    with pytest.raises(ValueError, match="must be resolved to nets"):
        _electrostatic_net_boundaries(_groups(), None)
    with pytest.raises(ValueError, match="must be resolved to nets"):
        _electrostatic_net_boundaries(_groups(), [])


def test_a_terminal_with_no_surfaces_raises() -> None:
    groups = _groups()
    groups["nets"].append({"name": "vias_only", "phys_groups": []})
    with pytest.raises(ValueError, match="'D' has no surfaces to drive"):
        _electrostatic_net_boundaries(groups, [_t("T", (0,)), _t("D", (3,))])


def test_a_net_position_beyond_the_mesh_raises() -> None:
    with pytest.raises(ValueError, match="Run mesh\\(\\) again"):
        _electrostatic_net_boundaries(_groups(), [_t("T", (7,))])


def test_generate_palace_config_writes_the_net_boundaries(tmp_path) -> None:
    path = generate_palace_config(
        _groups(),
        [],
        [],
        stack(),
        tmp_path,
        "palace",
        100e9,
        simulation_type="electrostatic",
        resolved_terminals=[_t("TOP", (0,)), _t("LEFT", (1,))],
    )
    boundaries = json.loads(path.read_text())["Boundaries"]
    assert [t["Index"] for t in boundaries["Terminal"]] == [1, 2]
    assert boundaries["Ground"]["Attributes"] == [14, 17, 24]


def test_generate_palace_config_needs_resolved_terminals_for_net_groups(
    tmp_path,
) -> None:
    from gsim.palace.models import TerminalConfig

    with pytest.raises(ValueError, match="must be resolved to nets"):
        generate_palace_config(
            _groups(),
            [],
            [],
            stack(),
            tmp_path,
            "palace",
            100e9,
            simulation_type="electrostatic",
            terminals=[TerminalConfig(name="T1", layer="m1")],
        )


def test_layer_groups_still_take_the_layer_path(tmp_path) -> None:
    groups = {
        "volumes": {},
        "conductor_surfaces": {
            "m1_xy": {"phys_group": 4, "tags": []},
            "m1_z": {"phys_group": 5, "tags": []},
            "m2_xy": {"phys_group": 6, "tags": []},
            "m2_z": {"phys_group": 7, "tags": []},
        },
        "pec_surfaces": {},
        "via_boundary_surfaces": {"v1": [8]},
        "port_surfaces": {},
        "boundary_surfaces": {},
    }
    from gsim.palace.models import TerminalConfig

    path = generate_palace_config(
        groups,
        [],
        [],
        stack(),
        tmp_path,
        "palace",
        100e9,
        simulation_type="electrostatic",
        terminals=[
            TerminalConfig(name="T1", layer="m1"),
            TerminalConfig(name="T2", layer="m2"),
        ],
    )
    boundaries = json.loads(path.read_text())["Boundaries"]
    assert boundaries == {
        "Terminal": [
            {"Index": 1, "Attributes": [4, 5, 8]},
            {"Index": 2, "Attributes": [6, 7, 8]},
        ]
    }
