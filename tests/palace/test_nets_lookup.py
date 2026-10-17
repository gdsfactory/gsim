"""Tests for the lookups on Nets: by name, by region, by distance to a point."""

from __future__ import annotations

import gdsfactory as gf
import klayout.db as kdb
import pytest
from shapely import box

from gsim.palace.mesh.geometry import extract_geometry
from gsim.palace.mesh.nets import extract_nets
from tests._electrode_layouts import (
    BAR_A,
    BAR_B,
    ISLAND,
    PORTS,
    VIA_A,
    VIA_B,
    A,
    B,
    nets_of,
    stack,
)

# nets of the full layout, in layout order: PLUS (A + bar + via), MINUS, net3 (island)


@pytest.fixture(scope="module")
def nets():
    return nets_of()


def test_the_full_layout_has_three_nets(nets) -> None:
    assert [n.name for n in nets] == ["PLUS", "MINUS", "net3"]


# ----- find ---------------------------------------------------------------------


def test_has_layer_tells_stack_layers_from_other_names(nets) -> None:
    assert nets.has_layer("m1")
    assert nets.has_layer("v1")
    assert not nets.has_layer("ox")


def test_find_by_net_name(nets) -> None:
    assert [n.name for n in nets.find("net3")] == ["net3"]


def test_find_by_port_name(nets) -> None:
    assert nets.find("MINUS") == (nets[1],)


def test_find_joined_name_of_a_net_with_two_ports() -> None:
    joined = nets_of(
        [A, B, BAR_A, VIA_A, BAR_B, VIA_B, ("m2", 2.5, 1.5, 7.5, 2.5)],
        ports=[("X", 2, 2, "m1"), ("Y", 8, 2, "m1")],
    )
    assert len(joined) == 1
    assert joined.find("X+Y") == (joined[0],)
    assert joined.find("Y") == (joined[0],)


def test_find_unknown_name_lists_the_available_names(nets) -> None:
    with pytest.raises(KeyError) as error:
        nets.find("NOPE")
    message = str(error.value)
    assert "NOPE" in message
    assert "PLUS" in message
    assert "net3" in message


def test_two_nets_with_the_same_port_name_are_both_found() -> None:
    both = nets_of([A, B], ports=[("SIG", 2, 2, "m1"), ("SIG", 8, 2, "m1")])
    assert len(both) == 2
    assert both.find("SIG") == (both[0], both[1])


# ----- nets_in -----------------------------------------------------------------


def test_nets_in_a_box_over_one_electrode(nets) -> None:
    assert nets.nets_in(box(-1, -1, 5, 5)) == (nets[0],)


def test_nets_in_a_box_over_two_electrodes_keeps_net_order(nets) -> None:
    assert nets.nets_in(box(-1, -1, 11, 5)) == (nets[0], nets[1])


def test_nets_in_follows_a_net_to_its_other_layers(nets) -> None:
    """The box touches only the m2 bar of PLUS; the whole net is selected."""
    assert nets.nets_in(box(1.2, 1.2, 2.8, 2.8), layers=["m2"]) == (nets[0],)


def test_nets_in_limited_to_a_layer(nets) -> None:
    # the box covers A on m1 and its bar on m2; on m2 alone only the bar counts
    region = box(-1, -1, 5, 5)
    assert nets.nets_in(region, layers=["m2"]) == (nets[0],)
    assert nets.nets_in(box(-1, 5, 11, 8), layers=["m2"]) == ()
    assert nets.nets_in(box(-1, 5, 11, 8), layers=["m1"]) == (nets[2],)


def test_nets_in_an_empty_region_is_empty(nets) -> None:
    assert nets.nets_in(box(20, 20, 30, 30)) == ()


def test_a_region_that_only_touches_a_shape_selects_nothing(nets) -> None:
    # A spans x in [0, 4]; the box starts at x = 4
    assert nets.nets_in(box(4, 0, 5, 4), layers=["m1"]) == ()


def test_nets_in_an_unknown_layer_raises(nets) -> None:
    with pytest.raises(KeyError, match="unknown conductor or via layer"):
        nets.nets_in(box(0, 0, 1, 1), layers=["nope"])


def test_nets_in_counts_via_shapes(nets) -> None:
    # over the via of PLUS but not its metal on m2 (bar is 1..3; the box is
    # inside the via footprint, which is also under the bar)
    assert nets.nets_in(box(1.6, 1.6, 2.4, 2.4), layers=["v1"]) == (nets[0],)


# ----- nets_near ---------------------------------------------------------------


def test_nets_near_a_point_on_a_shape_has_distance_zero(nets) -> None:
    found = nets.nets_near(2, 2, "m1", 0.05)
    assert found == [(0.0, nets[0])]


def test_nets_near_a_point_in_the_gap_lists_both_sides_nearest_first(nets) -> None:
    found = nets.nets_near(4.5, 2, "m1", 2.0)
    assert [net.name for _, net in found] == ["PLUS", "MINUS"]
    assert found[0][0] == pytest.approx(0.5)
    assert found[1][0] == pytest.approx(1.5)


def test_nets_near_respects_the_distance(nets) -> None:
    assert nets.nets_near(4.5, 2, "m1", 0.4) == []
    assert [n.name for _, n in nets.nets_near(4.5, 2, "m1", 1.0)] == ["PLUS"]


def test_nets_near_only_looks_at_the_given_layer(nets) -> None:
    assert nets.nets_near(5, 2, "m2", 0.5) == []


def test_nets_near_a_net_with_two_shapes_lists_it_once() -> None:
    two = nets_of([("m1", 0, 0, 1, 1), ("m1", 3, 0, 4, 1), ("m1", 1, 0, 3, 0.5)])
    assert len(two) == 1
    found = two.nets_near(2, 2, "m1", 3.0)
    assert len(found) == 1
    # the nearest of its shapes is the corner (1, 1) of the first, 2 ** 0.5 away,
    # not the bridge 1.5 below
    assert found[0][0] == pytest.approx(2**0.5)


def test_nets_near_unknown_layer_raises(nets) -> None:
    with pytest.raises(KeyError, match="unknown conductor or via layer"):
        nets.nets_near(0, 0, "nope", 1.0)


def test_nets_near_on_a_layer_without_shapes_is_empty() -> None:
    only_m1 = nets_of([A], ports=[])
    assert only_m1.nets_near(1, 1, "m2", 5.0) == []


def test_tree_is_cached(nets) -> None:
    nets.nets_near(2, 2, "m1", 0.1)
    first = nets._trees["m1"]
    nets.nets_near(3, 3, "m1", 0.1)
    assert nets._trees["m1"] is first


# ----- polygons with holes -------------------------------------------------------


def _ring_with_island():
    """A 4 x 4 m1 ring with a 2 x 2 hole and a 1 x 1 island inside the hole."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    ring = kdb.DPolygon(kdb.DBox(0, 0, 4, 4))
    ring.insert_hole(
        [kdb.DPoint(1, 1), kdb.DPoint(1, 3), kdb.DPoint(3, 3), kdb.DPoint(3, 1)]
    )
    c.shapes(c.kcl.layer(1, 0)).insert(ring)
    c.add_polygon([(1.5, 1.5), (2.5, 1.5), (2.5, 2.5), (1.5, 2.5)], layer=(1, 0))
    st = stack()
    return extract_nets(extract_geometry(c, st), st, c.ports)


def test_a_point_in_the_hole_is_near_the_island_and_the_ring_not_on_either() -> None:
    nets = _ring_with_island()
    assert len(nets) == 2
    found = nets.nets_near(1.25, 2, "m1", 1.0)
    assert [round(gap, 3) for gap, _ in found] == [0.25, 0.25]
    assert {net.name for _, net in found} == {"net1", "net2"}
    assert nets.net_at(1.25, 2, "m1") is None


def test_nets_in_a_region_inside_the_hole_selects_only_the_island() -> None:
    nets = _ring_with_island()
    inside = nets.nets_in(box(1.1, 1.1, 2.9, 2.9))
    assert len(inside) == 1
    assert inside[0] is nets.net_at(2, 2, "m1")


def test_ports_fixture_is_the_plus_minus_pair() -> None:
    assert [p[0] for p in PORTS] == ["PLUS", "MINUS"]
    assert ISLAND[0] == "m1"
