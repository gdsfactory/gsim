"""Tests for resolving terminals to nets (pure Python, no mesh)."""

from __future__ import annotations

import pytest

from gsim.palace.models import TerminalConfig
from gsim.palace.terminals import ResolvedTerminal, resolve_terminals
from tests._electrode_layouts import (
    BAR_A,
    BAR_B,
    ISLAND,
    VIA_A,
    VIA_B,
    A,
    B,
    make_sim,
    nets_of,
)

# the full layout: nets PLUS (A, bar, via), MINUS (B, bar, via) and net3 (island)


@pytest.fixture(scope="module")
def nets():
    return nets_of()


def _t(name: str, **kwargs) -> TerminalConfig:
    return TerminalConfig(name=name, **kwargs)


def test_two_electrodes_on_one_layer_give_two_terminals(nets) -> None:
    resolved = resolve_terminals([_t("L", net="PLUS"), _t("R", net="MINUS")], nets)
    assert resolved == [
        ResolvedTerminal("L", (0,), False, ("PLUS",)),
        ResolvedTerminal("R", (1,), False, ("MINUS",)),
    ]


def test_terminal_order_is_the_order_given(nets) -> None:
    resolved = resolve_terminals([_t("R", net="MINUS"), _t("L", net="PLUS")], nets)
    assert [t.name for t in resolved] == ["R", "L"]
    assert [t.nets for t in resolved] == [(1,), (0,)]


def test_a_via_connected_electrode_is_one_net_by_name_or_by_layer() -> None:
    boxes = [A, BAR_A, VIA_A, ("m1", 20, 0, 24, 4)]
    nets = nets_of(boxes, ports=[("PLUS", 2, 2, "m1")])
    assert len(nets) == 2
    by_net = resolve_terminals([_t("T", net="PLUS")], nets)
    by_m2 = resolve_terminals([_t("T", layer="m2")], nets)
    assert by_net[0].nets == by_m2[0].nets == (0,)
    # layer="m1" selects every net with an m1 shape, so both
    both = resolve_terminals([_t("T", layer="m1")], nets)
    assert both[0].nets == (0, 1)


def test_region_selects_whole_nets(nets) -> None:
    # the box only covers the m2 bar of MINUS
    resolved = resolve_terminals(
        [_t("R", region=(7.2, 1.2, 8.8, 2.8), layer="m2"), _t("L", net="PLUS")], nets
    )
    assert resolved[0].nets == (1,)
    assert resolved[0].net_names == ("MINUS",)


def test_region_without_a_layer_looks_at_every_layer(nets) -> None:
    resolved = resolve_terminals([_t("both", region=(-1, -1, 11, 5))], nets)
    assert resolved[0].nets == (0, 1)


def test_region_polygon_selects_by_overlap(nets) -> None:
    triangle = [(-1, -1), (5, -1), (-1, 5)]
    assert resolve_terminals([_t("T", region=triangle)], nets)[0].nets == (0,)


def test_the_island_goes_to_no_terminal_with_ground(nets) -> None:
    resolved = resolve_terminals(
        [_t("L", net="PLUS"), _t("R", net="MINUS")], nets, unassigned="ground"
    )
    assert [t.name for t in resolved] == ["L", "R"]
    assert all(2 not in t.nets for t in resolved)


def test_the_island_becomes_a_floating_terminal_with_floating(nets) -> None:
    resolved = resolve_terminals(
        [_t("L", net="PLUS"), _t("R", net="MINUS")], nets, unassigned="floating"
    )
    assert [t.name for t in resolved] == ["L", "R", "net3"]
    assert resolved[2] == ResolvedTerminal("net3", (2,), True, ("net3",))


def test_floating_flag_is_carried(nets) -> None:
    resolved = resolve_terminals(
        [_t("L", net="PLUS"), _t("S", net="net3", floating=True)], nets
    )
    assert [t.floating for t in resolved] == [False, True]


def test_a_net_in_two_terminals_raises_naming_both(nets) -> None:
    with pytest.raises(ValueError, match="'PLUS' is in two terminals, 'A' and 'B'"):
        resolve_terminals([_t("A", net="PLUS"), _t("B", layer="m2")], nets)


def test_a_net_reached_by_port_and_by_name_in_two_terminals_raises(nets) -> None:
    with pytest.raises(ValueError, match="in two terminals"):
        resolve_terminals([_t("A", net="PLUS"), _t("B", region=(0, 0, 4, 4))], nets)


def test_unknown_net_raises_with_the_available_names(nets) -> None:
    with pytest.raises(ValueError, match=r"Terminal 'X': no net or port named 'NOPE'"):
        resolve_terminals([_t("X", net="NOPE"), _t("Y", net="PLUS")], nets)


def test_empty_region_raises(nets) -> None:
    with pytest.raises(ValueError, match="Terminal 'X': no net found at the region"):
        resolve_terminals([_t("X", region=(50, 50, 60, 60))], nets)


def test_layer_without_nets_raises(nets) -> None:
    with pytest.raises(ValueError, match="Terminal 'X'"):
        resolve_terminals([_t("X", layer="nope")], nets)


def test_region_on_an_unknown_layer_raises(nets) -> None:
    with pytest.raises(ValueError, match=r"Terminal 'X'.*unknown conductor or via"):
        resolve_terminals([_t("X", region=(0, 0, 1, 1), layer="nope")], nets)


def test_a_floating_name_clash_raises(nets) -> None:
    with pytest.raises(ValueError, match=r"'net3'.*name of another terminal"):
        resolve_terminals(
            [_t("net3", net="PLUS"), _t("R", net="MINUS")], nets, unassigned="floating"
        )


def test_two_unassigned_nets_with_one_name_cannot_both_float() -> None:
    nets = nets_of([A, B, ISLAND], ports=[("SIG", 2, 2, "m1"), ("SIG", 8, 2, "m1")])
    with pytest.raises(ValueError, match="'SIG'"):
        resolve_terminals([_t("T", net="net3")], nets, unassigned="floating")


def test_all_floating_terminals_raise(nets) -> None:
    with pytest.raises(ValueError, match="Every terminal is floating"):
        resolve_terminals(
            [_t("A", net="PLUS", floating=True), _t("B", net="MINUS", floating=True)],
            nets,
        )


# ----- ElectrostaticSim.resolve_terminals and validate_config ------------------


def test_sim_resolve_terminals_valid_configuration(tmp_path) -> None:
    sim = make_sim(tmp_path / "es")
    sim.add_terminal("TOP", net="PLUS")
    sim.add_terminal("LEFT", region=(7, 0, 11, 5), layer="m1")
    resolved = sim.resolve_terminals()
    assert [(t.name, t.net_names, t.floating) for t in resolved] == [
        ("TOP", ("PLUS",), False),
        ("LEFT", ("MINUS",), False),
    ]
    assert sim.validate_config().valid


def test_sim_resolve_terminals_adds_floating_terminals(tmp_path) -> None:
    sim = make_sim(tmp_path / "es")
    sim.add_terminal("TOP", net="PLUS")
    sim.add_terminal("LEFT", net="MINUS")
    sim.set_unassigned_conductors("floating")
    assert [t.name for t in sim.resolve_terminals()] == ["TOP", "LEFT", "net3"]


def test_sim_unknown_net_fails_validation_before_meshing(tmp_path) -> None:
    sim = make_sim(tmp_path / "es")
    sim.add_terminal("TOP", net="PLUS")
    sim.add_terminal("LEFT", net="NOPE")
    result = sim.validate_config()
    assert not result.valid
    expected = "Terminals: Terminal 'LEFT': no net or port named 'NOPE'"
    assert any(expected in e for e in result.errors)
    with pytest.raises(ValueError, match="no net or port named 'NOPE'"):
        sim.mesh()


def test_sim_empty_region_fails_validation(tmp_path) -> None:
    sim = make_sim(tmp_path / "es")
    sim.add_terminal("TOP", net="PLUS")
    sim.add_terminal("LEFT", region=(50, 50, 60, 60))
    result = sim.validate_config()
    assert any("no net found at the region" in e for e in result.errors)


def test_sim_net_in_two_terminals_fails_validation(tmp_path) -> None:
    sim = make_sim(tmp_path / "es")
    sim.add_terminal("TOP", net="PLUS")
    sim.add_terminal("BOTTOM", layer="m2")
    assert any("in two terminals" in e for e in sim.validate_config().errors)


def test_layer_only_validation_does_not_resolve_nets(tmp_path) -> None:
    """Layer terminals keep the old validation: no net extraction, no new errors."""
    sim = make_sim(tmp_path / "es")
    sim.add_terminal("T1", layer="m1")
    sim.add_terminal("T2", layer="m2")
    assert sim.validate_config().valid


def test_two_nets_with_the_same_port_name_go_into_one_terminal() -> None:
    nets = nets_of(
        [A, B, BAR_A, VIA_A, BAR_B, VIA_B],
        ports=[("SIG", 2, 2, "m1"), ("SIG", 8, 2, "m1")],
    )
    resolved = resolve_terminals([_t("T", net="SIG")], nets)
    assert resolved[0].nets == (0, 1)
