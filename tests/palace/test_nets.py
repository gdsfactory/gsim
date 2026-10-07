"""Tests for finding the electrical nets of a layout.

Two conductor shapes are on the same net when metal touches metal on one layer,
or when a via joins the layers it spans. Every expectation below follows from
that rule and not from what the code does.

Layers of the synthetic stack (z in um): ``m1`` [0, 1], ``v1`` [1, 2],
``m2`` [2, 3], ``v2`` [3, 4], ``m3`` [4, 5], ``vgap`` [1.3, 1.7], a via that
touches no metal, ``m1b`` [0.5, 1.5], a conductor whose z-range overlaps ``m1``,
and ``ox`` [0, 5], a dielectric.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace

import gdsfactory as gf
import klayout.db as kdb
import pytest
from shapely import Polygon

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace import ElectrostaticSim
from gsim.palace.mesh.geometry import extract_geometry
from gsim.palace.mesh.nets import Nets, extract_nets

Box = tuple[str, float, float, float, float]
PortSpec = tuple[str, float, float, str]

LAYERS = {
    "m1": (1, 0.0, 1.0, "conductor"),
    "v1": (3, 1.0, 2.0, "via"),
    "m2": (2, 2.0, 3.0, "conductor"),
    "v2": (5, 3.0, 4.0, "via"),
    "m3": (4, 4.0, 5.0, "conductor"),
    "vgap": (6, 1.3, 1.7, "via"),
    "m1b": (7, 0.5, 1.5, "conductor"),
    "ox": (8, 0.0, 5.0, "dielectric"),
}


def _stack(layers: dict[str, tuple] | None = None) -> LayerStack:
    return LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=(number, 0),
                zmin=zmin,
                zmax=zmax,
                thickness=zmax - zmin,
                material="aluminum",
                layer_type=kind,
            )
            for name, (number, zmin, zmax, kind) in (layers or LAYERS).items()
        }
    )


def _component(
    boxes: Sequence[Box],
    ports: Sequence[PortSpec] = (),
    layers: dict[str, tuple] | None = None,
) -> gf.Component:
    """Rectangles on the stack layers; ports sit on the pin datatype, as in IHP."""
    gf.gpdk.PDK.activate()
    table = layers or LAYERS
    c = gf.Component()
    for layer, x0, y0, x1, y1 in boxes:
        c.add_polygon(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=(table[layer][0], 0)
        )
    for name, x, y, layer in ports:
        c.add_port(
            name=name,
            center=(x, y),
            width=1.0,
            orientation=0,
            layer=(table[layer][0], 2),
            port_type="electrical",
        )
    return c


def _nets(
    boxes: Sequence[Box],
    ports: Sequence[PortSpec] = (),
    layers: dict[str, tuple] | None = None,
) -> Nets:
    stack = _stack(layers)
    c = _component(boxes, ports, layers)
    return extract_nets(extract_geometry(c, stack), stack, c.ports)


def _count(boxes: Sequence[Box]) -> int:
    return len(_nets(boxes))


# ----- metal touching metal on one layer -------------------------------------


def test_separate_shapes_on_one_layer_are_separate_nets() -> None:
    assert _count([("m1", 0, 0, 1, 1), ("m1", 3, 0, 4, 1)]) == 2


def test_shapes_sharing_an_edge_are_one_net() -> None:
    assert _count([("m1", 0, 0, 1, 1), ("m1", 1, 0, 2, 1)]) == 1


def test_overlapping_shapes_are_one_net() -> None:
    assert _count([("m1", 0, 0, 2, 1), ("m1", 1, 0, 3, 1)]) == 1


def test_shapes_touching_only_at_a_corner_are_not_connected() -> None:
    """A point has no width, so no current passes through it."""
    assert _count([("m1", 0, 0, 1, 1), ("m1", 1, 1, 2, 2)]) == 2


def test_a_gap_of_one_grid_step_keeps_shapes_apart() -> None:
    assert _count([("m1", 0, 0, 1, 1), ("m1", 1.001, 0, 2, 1)]) == 2


def test_connection_is_transitive_along_a_chain() -> None:
    """A touches B and B touches C, so A and C are on one net too."""
    chain = [("m1", 0, 0, 1, 1), ("m1", 1, 0, 2, 1), ("m1", 2, 0, 3, 1)]
    assert _count(chain) == 1


def test_metal_on_different_layers_is_not_connected_without_a_via() -> None:
    assert _count([("m1", 0, 0, 2, 2), ("m2", 1, 1, 3, 3)]) == 2


# ----- vias ------------------------------------------------------------------


def test_a_via_over_both_metals_joins_the_layers() -> None:
    boxes = [("m1", 0, 0, 2, 2), ("m2", 1, 1, 3, 3), ("v1", 1.2, 1.2, 1.8, 1.8)]
    assert _count(boxes) == 1
    assert _count(boxes[:2]) == 2


def test_a_via_that_misses_the_upper_metal_in_plan_does_not_join() -> None:
    boxes = [("m1", 0, 0, 2, 2), ("m2", 1, 1, 3, 3), ("v1", 0.2, 0.2, 0.8, 0.8)]
    nets = _nets(boxes)
    assert len(nets) == 2
    assert nets.net_at(0.5, 0.5, "v1") is nets.net_at(0.5, 0.5, "m1")
    assert nets.net_at(2.5, 2.5, "m2") is not nets.net_at(0.5, 0.5, "m1")


def test_a_via_touching_a_metal_only_along_an_edge_does_not_join_it() -> None:
    """The via lies over m2 and abuts the edge of m1: no contact area with m1."""
    boxes = [("m1", 0, 0, 2, 2), ("m2", 1, 0, 3, 2), ("v1", 2, 0, 3, 2)]
    nets = _nets(boxes)
    assert len(nets) == 2
    assert nets.net_at(2.5, 1, "v1") is nets.net_at(2.5, 1, "m2")


def test_a_via_that_does_not_reach_the_metal_in_z_does_not_join() -> None:
    """vgap sits between the metals in plan but spans z 1.3 to 1.7 only."""
    boxes = [("m1", 0, 0, 2, 2), ("m2", 1, 1, 3, 3), ("vgap", 1.2, 1.2, 1.8, 1.8)]
    nets = _nets(boxes)
    assert len(nets) == 3
    orphan = nets.net_at(1.5, 1.5, "vgap")
    assert orphan is not None
    assert orphan.layers == ("vgap",)


def test_a_via_whose_z_range_only_overlaps_the_metal_still_joins() -> None:
    """PDK stacks overlap vias into the metal by a few nm; that is contact."""
    layers = {
        "lo": (1, 0.0, 1.0, "conductor"),
        "via": (2, 0.9, 2.1, "via"),
        "hi": (3, 2.0, 3.0, "conductor"),
    }
    boxes = [("lo", 0, 0, 2, 2), ("hi", 0, 0, 2, 2), ("via", 0.5, 0.5, 1.5, 1.5)]
    assert len(_nets(boxes, layers=layers)) == 1


def test_z_ranges_that_meet_up_to_float_noise_still_touch() -> None:
    """0.1 + 0.2 is 0.30000000000000004: a via ending at 0.3 reaches that metal."""
    layers = {
        "via": (1, 0.0, 0.3, "via"),
        "top": (2, 0.1 + 0.2, 1.0, "conductor"),
    }
    boxes = [("via", 0, 0, 1, 1), ("top", 0, 0, 1, 1)]
    assert len(_nets(boxes, layers=layers)) == 1


def test_a_via_only_joins_the_layers_it_touches() -> None:
    """v1 spans m1 and m2, so it cannot connect m3 even with the same footprint."""
    boxes = [
        ("m1", 0, 0, 2, 2),
        ("m3", 0, 0, 2, 2),
        ("v1", 0.5, 0.5, 1.5, 1.5),
    ]
    nets = _nets(boxes)
    assert len(nets) == 2
    upper = nets.net_at(1, 1, "m3")
    assert upper is not None
    assert upper.layers == ("m3",)


def test_a_stack_of_metals_and_vias_is_one_net() -> None:
    boxes = [
        ("m1", 0, 0, 2, 2),
        ("v1", 0.5, 0.5, 1.5, 1.5),
        ("m2", 0, 0, 2, 2),
        ("v2", 0.5, 0.5, 1.5, 1.5),
        ("m3", 0, 0, 2, 2),
    ]
    nets = _nets(boxes)
    assert len(nets) == 1
    assert nets[0].layers == ("m1", "m2", "m3", "v1", "v2")


def test_conductors_whose_z_ranges_overlap_are_not_connected_without_a_via() -> None:
    """A gate over active silicon overlaps it in plan and in z, and is insulated."""
    assert _count([("m1", 0, 0, 2, 2), ("m1b", 1, 1, 3, 3)]) == 2


def test_dielectric_shapes_are_not_nets_and_join_nothing() -> None:
    boxes = [("m1", 0, 0, 2, 2), ("m2", 1, 1, 3, 3), ("ox", 0, 0, 3, 3)]
    assert _count(boxes) == 2


# ----- the situation of gsim#273 ---------------------------------------------

A = ("m1", 0, 0, 1, 1)
B = ("m1", 3, 0, 4, 1)
BAR = ("m2", 0.5, 0, 3.5, 1)
VIA_A = ("v1", 0.6, 0.2, 0.9, 0.8)
VIA_B = ("v1", 3.1, 0.2, 3.4, 0.8)


def test_disconnected_electrodes_on_one_layer_stay_separate() -> None:
    assert _count([A, B]) == 2


def test_a_bar_overlapping_both_electrodes_without_vias_joins_nothing() -> None:
    assert _count([A, B, BAR]) == 3


def test_a_via_path_to_one_electrode_leaves_the_other_apart() -> None:
    nets = _nets([A, B, BAR, VIA_A])
    assert len(nets) == 2
    assert nets.net_at(0.5, 0.5, "m1") is nets.net_at(2, 0.5, "m2")
    assert nets.net_at(3.5, 0.5, "m1") is not nets.net_at(0.5, 0.5, "m1")


def test_a_metal_and_via_path_between_the_electrodes_joins_them() -> None:
    nets = _nets([A, B, BAR, VIA_A, VIA_B])
    assert len(nets) == 1
    assert nets.net_at(0.5, 0.5, "m1") is nets.net_at(3.5, 0.5, "m1")


# ----- polygons with holes -----------------------------------------------------


def _ring_with_island(island: tuple[float, float, float, float]) -> Nets:
    """A 4 x 4 m1 ring with a 2 x 2 hole, plus a rectangle on m1."""
    gf.gpdk.PDK.activate()
    c = gf.Component()
    ring = kdb.DPolygon(kdb.DBox(0, 0, 4, 4))
    ring.insert_hole(
        [kdb.DPoint(1, 1), kdb.DPoint(1, 3), kdb.DPoint(3, 3), kdb.DPoint(3, 1)]
    )
    c.shapes(c.kcl.layer(1, 0)).insert(ring)
    x0, y0, x1, y1 = island
    c.add_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], layer=(1, 0))
    stack = _stack()
    return extract_nets(extract_geometry(c, stack), stack, c.ports)


def test_an_island_inside_the_hole_of_a_ring_is_a_separate_net() -> None:
    nets = _ring_with_island((1.5, 1.5, 2.5, 2.5))
    assert len(nets) == 2
    assert nets.net_at(0.5, 0.5, "m1") is not nets.net_at(2, 2, "m1")
    assert nets.net_at(2, 2, "m1") is not None
    assert nets.net_at(1.2, 2, "m1") is None  # in the hole, no metal


def test_an_island_touching_the_edge_of_the_hole_joins_the_ring() -> None:
    assert len(_ring_with_island((1, 1.5, 2, 2.5))) == 1


# ----- ports name the nets -----------------------------------------------------


def test_a_port_names_the_net_it_sits_on() -> None:
    nets = _nets(
        [A, B, BAR, VIA_A, VIA_B, ("m3", 10, 10, 11, 11)],
        ports=[("PLUS", 0.5, 0.5, "m1")],
    )
    assert [net.name for net in nets] == ["PLUS", "net2"]
    assert nets[0].ports == ("PLUS",)
    assert nets[1].ports == ()


def test_ports_match_by_layer_number_whatever_the_datatype() -> None:
    """The polygons are on datatype 0 and the port on the pin datatype 2."""
    nets = _nets([A], ports=[("P", 0.5, 0.5, "m1")])
    assert nets[0].name == "P"


def test_a_port_on_the_edge_of_a_shape_belongs_to_it() -> None:
    nets = _nets([A], ports=[("edge", 1.0, 0.5, "m1"), ("corner", 0.0, 0.0, "m1")])
    assert nets[0].ports == ("edge", "corner")


def test_two_ports_on_one_net_name_it_together() -> None:
    nets = _nets(
        [A, B, BAR, VIA_A, VIA_B],
        ports=[("X", 0.5, 0.5, "m1"), ("Y", 3.5, 0.5, "m1")],
    )
    assert len(nets) == 1
    assert nets[0].name == "X+Y"
    assert nets[0].ports == ("X", "Y")


def test_ports_with_one_name_on_one_net_list_it_once() -> None:
    """A ground plane can carry several ports with the same name."""
    wide = ("m1", 0, 0, 4, 1)
    nets = _nets([wide], ports=[("gnd", 0.5, 0.5, "m1"), ("gnd", 3.5, 0.5, "m1")])
    assert nets[0].ports == ("gnd",)
    assert nets[0].name == "gnd"


def test_a_port_on_another_layer_than_the_metal_under_it_is_unmatched() -> None:
    nets = _nets([A], ports=[("wrong", 0.5, 0.5, "m2")])
    assert nets.unmatched_ports == ("wrong",)
    assert nets[0].ports == ()


def test_a_port_over_no_metal_is_unmatched() -> None:
    nets = _nets([A], ports=[("lost", 8, 8, "m1")])
    assert nets.unmatched_ports == ("lost",)


# ----- layouts with nothing to connect -----------------------------------------


def test_a_layout_with_only_dielectric_shapes_has_no_nets() -> None:
    nets = _nets([("ox", 0, 0, 3, 3)])
    assert len(nets) == 0
    assert str(nets) == "0 nets"


def test_ports_of_a_layout_without_shapes_are_unmatched() -> None:
    nets = _nets([], ports=[("P", 0.5, 0.5, "m1")])
    assert len(nets) == 0
    assert nets.unmatched_ports == ("P",)
    assert nets.net_at(0.5, 0.5, "m1") is None


# ----- looking up the net at a point ---------------------------------------------


def test_net_at_finds_the_net_under_a_point_on_a_layer() -> None:
    nets = _nets([A, B])
    assert nets.net_at(0.5, 0.5, "m1") is nets[0]
    assert nets.net_at(3.5, 0.5, "m1") is nets[1]
    assert nets.net_at(2, 0.5, "m1") is None  # between the electrodes
    assert nets.net_at(0.5, 0.5, "m2") is None  # nothing on that layer there
    assert nets.net_at(1.0, 0.5, "m1") is nets[0]  # on the edge of the first


def test_net_at_rejects_a_layer_that_is_not_conductor_or_via() -> None:
    with pytest.raises(KeyError, match="unknown conductor or via layer 'm9'"):
        _nets([A]).net_at(0.5, 0.5, "m9")


def test_net_at_accepts_a_layer_with_no_shapes() -> None:
    assert _nets([A]).net_at(0.5, 0.5, "m3") is None


def test_str_lists_each_net_with_its_shapes_per_layer() -> None:
    nets = _nets(
        [A, B, BAR, VIA_A],
        ports=[("PLUS", 0.5, 0.5, "m1"), ("lost", 9, 9, "m1")],
    )
    assert str(nets) == (
        "2 nets\n"
        "  PLUS: m1 x1, m2 x1, v1 x1\n"
        "  net2: m1 x1\n"
        "  ports on no conductor: lost"
    )


def test_a_port_where_two_unconnected_shapes_meet_names_both_nets() -> None:
    """Shapes that meet at one point are two nets, and the port sits on both."""
    nets = _nets(
        [("m1", 0, 0, 1, 1), ("m1", 1, 1, 2, 2)], ports=[("P", 1.0, 1.0, "m1")]
    )
    assert len(nets) == 2
    assert [net.ports for net in nets] == [("P",), ("P",)]


def test_two_stack_layers_on_one_gds_layer_make_one_net_of_the_shape() -> None:
    """A thick metal split into two levels of the stack is still one conductor."""
    layers = {
        "thick_lo": (9, 0.0, 0.5, "conductor"),
        "thick_hi": (9, 0.5, 1.0, "conductor"),
    }
    nets = _nets([("thick_lo", 0, 0, 1, 1)], layers=layers)
    assert len(nets) == 1
    assert nets[0].layers == ("thick_hi", "thick_lo")


# ----- properties of the result, on random layouts ---------------------------------


def _random_boxes(seed: int) -> list[Box]:
    rng = random.Random(seed)  # noqa: S311
    boxes: list[Box] = []
    for layer in ("m1", "v1", "m2", "v2", "m3", "vgap"):
        for _ in range(12):
            x, y = rng.randrange(0, 20), rng.randrange(0, 20)
            w, h = rng.randrange(1, 6), rng.randrange(1, 6)
            boxes.append((layer, x, y, x + w, y + h))
    return boxes


def _flood_fill(boxes: Sequence[Box]) -> set[frozenset[tuple]]:
    """Reference: test every pair of boxes, then flood fill the graph.

    Boxes are identified by their layer number and corners, since identical
    boxes are one polygon in the layout.
    """
    shapes = [
        Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]) for _, x0, y0, x1, y1 in boxes
    ]

    def joined(i: int, j: int) -> bool:
        (_, i_lo, i_hi, i_kind) = LAYERS[boxes[i][0]]
        (_, j_lo, j_hi, j_kind) = LAYERS[boxes[j][0]]
        common = shapes[i].intersection(shapes[j])
        if boxes[i][0] == boxes[j][0]:
            return common.length > 0
        if {i_kind, j_kind} != {"conductor", "via"}:
            return False
        return i_lo <= j_hi and j_lo <= i_hi and common.area > 0

    seen: set[int] = set()
    parts: set[frozenset[tuple]] = set()
    for start in range(len(boxes)):
        if start in seen:
            continue
        part, todo = {start}, [start]
        while todo:
            i = todo.pop()
            for j in range(len(boxes)):
                if j not in part and joined(i, j):
                    part.add(j)
                    todo.append(j)
        seen |= part
        parts.add(frozenset(_box_key(boxes[i]) for i in part))
    return parts


def _box_key(box: Box) -> tuple:
    layer, x0, y0, x1, y1 = box
    return (LAYERS[layer][0], x0, y0, x1, y1)


def _polygon_key(geometry, index: int) -> tuple:
    layernum, xs, ys, _ = geometry.polygons[index]
    return (layernum, min(xs), min(ys), max(xs), max(ys))


@pytest.mark.parametrize("seed", range(8))
def test_nets_match_a_brute_force_flood_fill_on_random_layouts(seed: int) -> None:
    boxes = _random_boxes(seed)
    stack = _stack()
    c = _component(boxes)
    geometry = extract_geometry(c, stack)
    nets = extract_nets(geometry, stack, c.ports)
    got = {frozenset(_polygon_key(geometry, i) for _, i in net.members) for net in nets}
    assert got == _flood_fill(boxes)


def test_every_conductor_and_via_shape_is_on_exactly_one_net() -> None:
    stack = _stack()
    c = _component(_random_boxes(3))
    geometry = extract_geometry(c, stack)
    nets = extract_nets(geometry, stack, c.ports)
    members = [member for net in nets for member in net.members]
    assert len(members) == len(set(members)) == len(geometry.polygons)


def test_the_partition_does_not_depend_on_the_order_of_the_shapes() -> None:
    stack = _stack()
    c = _component(_random_boxes(5))
    geometry = extract_geometry(c, stack)

    def partition(geom) -> set[frozenset[tuple]]:
        nets = extract_nets(geom, stack, c.ports)
        return {
            frozenset((layer, _polygon_key(geom, i)) for layer, i in net.members)
            for net in nets
        }

    shuffled = list(geometry.polygons)
    random.Random(1).shuffle(shuffled)  # noqa: S311
    assert partition(replace(geometry, polygons=shuffled)) == partition(geometry)


# ----- the simulation object -------------------------------------------------------


def test_electrostatic_sim_reports_the_nets_of_its_geometry() -> None:
    sim = ElectrostaticSim()
    sim.set_geometry(
        _component(
            [A, B, BAR, VIA_A],
            ports=[("PLUS", 0.5, 0.5, "m1"), ("MINUS", 3.5, 0.5, "m1")],
        )
    )
    sim.set_stack(_stack())
    assert [net.name for net in sim.nets()] == ["PLUS", "MINUS"]


def test_electrostatic_sim_names_nets_by_electrical_ports_only() -> None:
    """An optical port sits on no conductor and must not be listed as lost."""
    component = _component([A], ports=[("PLUS", 0.5, 0.5, "m1")])
    component.add_port(
        name="o1",
        center=(9, 9),
        width=1.0,
        orientation=0,
        layer=(LAYERS["m1"][0], 2),
        port_type="optical",
    )
    sim = ElectrostaticSim()
    sim.set_geometry(component)
    sim.set_stack(_stack())
    nets = sim.nets()
    assert [net.name for net in nets] == ["PLUS"]
    assert nets.unmatched_ports == ()


def test_electrostatic_sim_without_geometry_says_so() -> None:
    with pytest.raises(ValueError, match="set_geometry"):
        ElectrostaticSim().nets()


# ----- real layouts from the IHP SG13G2 cells ---------------------------------------


@pytest.fixture
def _ihp() -> None:
    ihp = pytest.importorskip("ihp")
    ihp.PDK.activate()


def _ihp_nets(cell: str) -> tuple[Nets, int]:
    from ihp import cells

    from gsim.common.stack import get_stack

    stack = get_stack(substrate_thickness=2.0)
    component = getattr(cells, cell)()
    geometry = extract_geometry(component, stack)
    return extract_nets(geometry, stack, component.ports), len(geometry.polygons)


@pytest.mark.usefixtures("_ihp")
def test_the_ihp_mim_capacitor_has_two_electrodes_named_by_its_ports() -> None:
    nets, _ = _ihp_nets("cmim")
    by_name = {net.name: net for net in nets}
    assert sorted(by_name) == ["MINUS", "PLUS"]
    # bottom plate is Metal5 alone; top plate is TopMetal1 with all its vmim vias
    assert by_name["MINUS"].layers == ("metal5",)
    assert by_name["PLUS"].layers == ("topmetal1", "vmim")
    assert len(by_name["PLUS"].members) == 1 + 36
    assert nets.unmatched_ports == ()


@pytest.mark.usefixtures("_ihp")
def test_the_ihp_interdigitated_capacitor_has_two_electrodes() -> None:
    nets, polygons = _ihp_nets("cmom")
    assert sorted(net.name for net in nets) == ["MINUS", "PLUS"]
    assert sum(len(net.members) for net in nets) == polygons
    assert nets.unmatched_ports == ()
