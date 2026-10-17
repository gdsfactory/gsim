"""Tests for selecting electrostatic terminals by layer, net or region (gsim#273)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gsim.palace import ElectrostaticSim
from gsim.palace.models import TerminalConfig

# ----- TerminalConfig ---------------------------------------------------------


def _area(terminal: TerminalConfig) -> float:
    polygon = terminal.region_polygon()
    assert polygon is not None
    return polygon.area


def test_layer_terminal_is_still_valid() -> None:
    terminal = TerminalConfig(name="T1", layer="m5")
    assert terminal.layer == "m5"
    assert terminal.net is None
    assert terminal.region is None
    assert terminal.floating is False


def test_old_dict_still_loads() -> None:
    terminal = TerminalConfig.model_validate({"name": "T1", "layer": "m5"})
    assert (terminal.name, terminal.layer) == ("T1", "m5")


def test_net_terminal_is_valid() -> None:
    terminal = TerminalConfig(name="TOP", net="PLUS")
    assert terminal.net == "PLUS"
    assert terminal.layer is None


def test_region_box_terminal_is_valid() -> None:
    terminal = TerminalConfig(name="L", region=(0, 0, 20, 20), layer="m1")
    assert terminal.region == (0, 0, 20, 20)
    assert terminal.layer == "m1"
    assert _area(terminal) == pytest.approx(400.0)


def test_region_polygon_terminal_is_valid() -> None:
    terminal = TerminalConfig(name="L", region=[(0, 0), (4, 0), (0, 3)])
    assert _area(terminal) == pytest.approx(6.0)


def test_region_box_given_as_a_list_is_a_box() -> None:
    terminal = TerminalConfig(name="L", region=[0, 0, 2, 5])
    assert _area(terminal) == pytest.approx(10.0)


def test_floating_flag_is_kept() -> None:
    assert TerminalConfig(name="S", net="net3", floating=True).floating is True


def test_a_terminal_without_a_selection_is_rejected() -> None:
    with pytest.raises(ValidationError, match="needs layer, net or region"):
        TerminalConfig(name="T1")


@pytest.mark.parametrize("extra", [{"region": (0, 0, 1, 1)}, {"layer": "m1"}])
def test_net_cannot_be_combined_with_region_or_layer(extra) -> None:
    with pytest.raises(ValidationError, match="net cannot be combined"):
        TerminalConfig(name="T1", net="PLUS", **extra)


@pytest.mark.parametrize(
    "box", [(1, 0, 1, 2), (2, 0, 1, 2), (0, 3, 1, 3), (0, 0, float("nan"), 1)]
)
def test_a_box_needs_min_below_max(box) -> None:
    with pytest.raises(ValidationError, match="region"):
        TerminalConfig(name="T1", region=box)


def test_a_polygon_needs_three_points() -> None:
    with pytest.raises(ValidationError, match="at least 3 points"):
        TerminalConfig(name="T1", region=[(0, 0), (1, 1)])


def test_a_degenerate_or_self_intersecting_polygon_is_rejected() -> None:
    with pytest.raises(ValidationError, match="degenerate or self-intersecting"):
        TerminalConfig(name="T1", region=[(0, 0), (1, 1), (2, 2)])
    with pytest.raises(ValidationError, match="degenerate or self-intersecting"):
        TerminalConfig(name="T1", region=[(0, 0), (2, 2), (2, 0), (0, 2)])


def test_assignment_is_validated_too() -> None:
    terminal = TerminalConfig(name="T1", layer="m1")
    with pytest.raises(ValidationError):
        terminal.net = "PLUS"  # layer is still set
    terminal.layer = None
    terminal.net = "PLUS"
    assert terminal.net == "PLUS"


def test_region_polygon_is_none_without_a_region() -> None:
    assert TerminalConfig(name="T1", layer="m1").region_polygon() is None


# ----- ElectrostaticSim.add_terminal and _mesh_by_net ---------------------------


def test_add_terminal_keeps_the_layer_form() -> None:
    sim = ElectrostaticSim()
    sim.add_terminal("T1", layer="m1")
    assert sim.terminals[0].layer == "m1"


def test_add_terminal_replaces_a_terminal_of_the_same_name() -> None:
    sim = ElectrostaticSim()
    sim.add_terminal("T1", layer="m1")
    sim.add_terminal("T1", net="PLUS")
    assert [(t.name, t.layer, t.net) for t in sim.terminals] == [("T1", None, "PLUS")]


def test_add_terminal_without_a_selection_raises() -> None:
    with pytest.raises(ValueError, match="needs layer, net or region"):
        ElectrostaticSim().add_terminal("T1")


def test_default_unassigned_conductors_are_grounded() -> None:
    assert ElectrostaticSim().unassigned_conductors == "ground"


def test_set_unassigned_conductors_validates() -> None:
    sim = ElectrostaticSim()
    sim.set_unassigned_conductors("floating")
    assert sim.unassigned_conductors == "floating"
    with pytest.raises(ValidationError):
        sim.set_unassigned_conductors("open")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("terminals", "unassigned", "expected"),
    [
        ([("T1", {"layer": "m1"}), ("T2", {"layer": "m2"})], "ground", False),
        ([("T1", {"net": "PLUS"}), ("T2", {"layer": "m2"})], "ground", True),
        ([("T1", {"layer": "m1"}), ("T2", {"region": (0, 0, 1, 1)})], "ground", True),
        (
            [("T1", {"layer": "m1"}), ("T2", {"layer": "m2", "floating": True})],
            "ground",
            True,
        ),
        ([("T1", {"layer": "m1"}), ("T2", {"layer": "m2"})], "floating", True),
        ([], "ground", False),
        ([], "floating", True),
    ],
)
def test_mesh_by_net_truth_table(terminals, unassigned, expected) -> None:
    sim = ElectrostaticSim()
    for name, kwargs in terminals:
        sim.add_terminal(name, **kwargs)
    sim.set_unassigned_conductors(unassigned)
    assert sim._mesh_by_net() is expected


def test_layer_terminals_on_one_layer_warn_in_validate_config() -> None:
    sim = ElectrostaticSim()
    sim.add_terminal("T1", layer="m1")
    sim.add_terminal("T2", layer="m1")
    warnings = sim.validate_config().warnings
    assert any("'T1', 'T2'" in w and "layer 'm1'" in w for w in warnings)


def test_layer_terminals_on_different_layers_do_not_warn() -> None:
    sim = ElectrostaticSim()
    sim.add_terminal("T1", layer="m1")
    sim.add_terminal("T2", layer="m2")
    assert not any("same conductors" in w for w in sim.validate_config().warnings)


def test_net_and_region_terminals_on_one_layer_do_not_warn() -> None:
    sim = ElectrostaticSim()
    sim.add_terminal("T1", net="PLUS")
    sim.add_terminal("T2", region=(0, 0, 1, 1), layer="m1")
    assert not any("same conductors" in w for w in sim.validate_config().warnings)


def test_validate_config_reports_terminals_without_a_selection() -> None:
    sim = ElectrostaticSim()
    sim.add_terminal("T1", layer="m1")
    sim.add_terminal("T2", layer="m2")
    # a model built without validation, as model_construct does
    sim.terminals[1] = TerminalConfig.model_construct(name="T2")
    errors = sim.validate_config().errors
    assert any("Terminal 'T2': needs layer, net or region" in e for e in errors)
