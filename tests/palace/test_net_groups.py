"""Tests for the per-net physical groups of electrostatic meshes (gsim#273).

The pure parts (via-merge check, classification of points) need no mesh. The
gmsh part runs through ``ElectrostaticSim.mesh`` on the synthetic stack of
``_electrode_layouts`` (m1, via v1, m2 in SiO2).
"""

from __future__ import annotations

import json

import gdsfactory as gf
import pytest

from gsim.palace.mesh.geometry import extract_geometry
from gsim.palace.mesh.net_groups import (
    _net_of_points,
    check_via_merging,
    net_labels,
)
from gsim.palace.mesh.nets import extract_nets
from tests._electrode_layouts import (
    FULL,
    PORTS,
    A,
    B,
    component,
    make_sim,
    nets_of,
    physical_names,
    stack,
    surface_nodes_by_group,
)

# ----- check_via_merging ---------------------------------------------------------


def _check(boxes, distance: float) -> None:
    c = component(boxes, ports=[])
    st = stack()
    geometry = extract_geometry(c, st)
    nets = extract_nets(geometry, st, c.ports)
    check_via_merging(geometry, st, nets, distance)


# electrodes 1 um apart on m1; each has a via at its edge, the vias 1 um apart
TWO_NETS = [
    ("m1", 0, 0, 4, 4),
    ("m1", 5, 0, 9, 4),
    ("v1", 3.5, 1, 4, 2),
    ("v1", 5, 1, 5.5, 2),
]


def test_vias_of_two_nets_closer_than_the_merge_distance_raise() -> None:
    with pytest.raises(ValueError, match=r"Via layer 'v1'.*different nets") as error:
        _check(TWO_NETS, 2.0)
    message = str(error.value)
    assert "'net1'" in message
    assert "'net2'" in message
    assert "1 um apart" in message
    assert "merge_via_distance" in message


def test_vias_of_two_nets_pass_with_a_smaller_merge_distance() -> None:
    _check(TWO_NETS, 0.5)


def test_merging_off_never_raises() -> None:
    _check(TWO_NETS, 0.0)


def test_vias_of_one_net_closer_than_the_merge_distance_pass() -> None:
    _check([("m1", 0, 0, 4, 4), ("v1", 0.5, 1, 1, 2), ("v1", 2, 1, 2.5, 2)], 2.0)


def test_a_single_via_never_raises() -> None:
    _check([("m1", 0, 0, 4, 4), ("v1", 0.5, 1, 1, 2)], 2.0)


def test_far_apart_vias_of_two_nets_pass_with_the_default_distance() -> None:
    _check(
        [
            ("m1", 0, 0, 4, 4),
            ("m1", 20, 0, 24, 4),
            ("v1", 1, 1, 2, 2),
            ("v1", 21, 1, 22, 2),
        ],
        2.0,
    )


# ----- _net_of_points ------------------------------------------------------------


def _l_shape_and_box():
    gf.gpdk.PDK.activate()
    c = gf.Component()
    c.add_polygon([(0, 0), (4, 0), (4, 1), (1, 1), (1, 4), (0, 4)], layer=(1, 0))
    c.add_polygon([(6, 0), (10, 0), (10, 4), (6, 4)], layer=(1, 0))
    st = stack()
    return extract_nets(extract_geometry(c, st), st, c.ports)


def test_points_on_an_l_shape_pick_its_net() -> None:
    nets = _l_shape_and_box()
    assert len(nets) == 2
    # inside both arms, on the outer edge, on the inner corner
    pts = [(0.5, 3.0), (3.0, 0.5), (0.0, 2.0), (1.0, 1.0)]
    assert _net_of_points(nets, "m1", pts, 0.05) == 0
    assert _net_of_points(nets, "m1", [(7, 2)], 0.05) == 1


def test_a_point_in_the_notch_of_the_l_is_off_every_shape() -> None:
    """The centre of mass of the L is (1.36, 1.36), outside it; a point there fails."""
    nets = _l_shape_and_box()
    with pytest.raises(ValueError, match=r"not within 0\.05 um of any drawn shape"):
        _net_of_points(nets, "m1", [(1.36, 1.36)], 0.05)


def test_points_that_pick_two_nets_raise() -> None:
    nets = _l_shape_and_box()
    with pytest.raises(ValueError, match=r"more than one net.*'net1', 'net2'"):
        _net_of_points(nets, "m1", [(0.5, 0.5), (7, 2)], 0.05)


def test_the_message_points_to_the_fix() -> None:
    nets = _l_shape_and_box()
    with pytest.raises(ValueError, match='curve_fit_mode="line"'):
        _net_of_points(nets, "m1", [(30, 30)], 0.05)


def test_a_point_on_the_shared_edge_of_one_nets_shapes_is_that_net() -> None:
    nets = nets_of([("m1", 0, 0, 2, 2), ("m1", 2, 0, 4, 2), ("m1", 9, 0, 10, 1)], [])
    assert len(nets) == 2
    assert _net_of_points(nets, "m1", [(2.0, 1.0)], 0.05) == 0


def test_the_nearest_net_wins_inside_the_tolerance() -> None:
    # two electrodes 0.06 um apart; a point on the first one is 0.06 from the other
    nets = nets_of([("m1", 0, 0, 1, 1), ("m1", 1.06, 0, 2, 1)], [])
    assert _net_of_points(nets, "m1", [(1.0, 0.5)], 0.1) == 0
    assert _net_of_points(nets, "m1", [(1.06, 0.5)], 0.1) == 1


def test_via_tolerance_reaches_the_rim_of_a_merged_via() -> None:
    nets = nets_of([("m1", 0, 0, 4, 4), ("v1", 0.5, 1, 1, 2), ("v1", 2, 1, 2.5, 2)], [])
    # in the filled gap between the two vias: 0.5 um from both, tolerance 1.06
    assert _net_of_points(nets, "v1", [(1.5, 1.5)], 1.06) == 0
    with pytest.raises(ValueError, match="not within"):
        _net_of_points(nets, "v1", [(1.5, 1.5)], 0.06)


# ----- the meshes ----------------------------------------------------------------

MESH = {"refined_mesh_size": 0.5, "max_mesh_size": 3.0, "merge_via_distance": 0}


def _mesh(path, boxes, terminals, *, unassigned="ground", ports=PORTS):
    sim = make_sim(path / "es", boxes, ports)
    for name, kwargs in terminals:
        sim.add_terminal(name, **kwargs)
    sim.set_unassigned_conductors(unassigned)
    result = sim.mesh(**MESH)
    return sim, sim._last_mesh_result.groups, result


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    """The probe layout in net mode: PLUS, MINUS and the loose island net3."""
    path = tmp_path_factory.mktemp("full")
    sim, groups, result = _mesh(
        path, FULL, [("TOP", {"net": "PLUS"}), ("LEFT", {"net": "MINUS"})]
    )
    return sim, groups, result, nets_of(FULL, PORTS)


def _names_of(groups, net: int) -> set[str]:
    """Names of the physical groups of one net, from the ``.msh`` order of tags."""
    return {
        name
        for section in ("conductor_surfaces", "pec_surfaces")
        for name, info in groups[section].items()
        if info.get("net") == groups["nets"][net]["label"]
    }


def test_each_net_has_its_own_conductor_groups(full) -> None:
    _, groups, _, _ = full
    assert [n["name"] for n in groups["nets"]] == ["PLUS", "MINUS", "net3"]
    assert _names_of(groups, 0) >= {"m1_xy@PLUS", "m1_z@PLUS"}
    assert _names_of(groups, 1) >= {"m1_xy@MINUS", "m1_z@MINUS"}
    assert _names_of(groups, 2) == {"m1_xy@net3", "m1_z@net3"}


def test_a_via_connected_net_covers_both_layers_and_its_via(full) -> None:
    _sim, groups, result, _ = full
    names = physical_names(result.mesh_path)
    for k, label in enumerate(["PLUS", "MINUS"]):
        entry = groups["nets"][k]
        assert entry["layers"] == ["m1", "m2", "v1"]
        assert {names[pg] for pg in entry["phys_groups"]} == {
            f"m1_xy@{label}",
            f"m1_z@{label}",
            f"m2_xy@{label}",
            f"m2_z@{label}",
            f"sio2__v1@{label}",
        }
    via = groups["via_boundary_surfaces"]
    assert set(via) == {"v1@PLUS", "v1@MINUS"}
    for label, pgs in via.items():
        assert [names[pg] for pg in pgs] == [f"sio2__{label}"]


def test_the_floating_island_is_its_own_net_with_groups(full) -> None:
    _, groups, result, _ = full
    names = physical_names(result.mesh_path)
    island = groups["nets"][2]
    assert island["ports"] == []
    assert island["layers"] == ["m1"]
    assert {names[pg] for pg in island["phys_groups"]} == {"m1_xy@net3", "m1_z@net3"}


def test_every_split_group_is_in_the_msh_file(full) -> None:
    _, groups, result, _ = full
    names = set(physical_names(result.mesh_path).values())
    split = {
        name
        for section in ("conductor_surfaces", "pec_surfaces")
        for name in groups[section]
    } | {f"sio2__{key}" for key in groups["via_boundary_surfaces"]}
    assert split <= names
    assert {"m1_xy", "m1_z", "m2_xy", "m2_z", "sio2__v1"}.isdisjoint(names)


def test_no_surface_is_in_two_groups(full) -> None:
    _, groups, _, _ = full
    seen: dict[int, str] = {}
    for section in ("conductor_surfaces", "pec_surfaces"):
        for name, info in groups[section].items():
            for tag in info["tags"]:
                assert tag not in seen, (tag, name, seen[tag])
                seen[tag] = name
    tags = [pg for entry in groups["nets"] for pg in entry["phys_groups"]]
    assert len(tags) == len(set(tags))


def test_physical_group_numbers_stay_unique_across_dimensions(full) -> None:
    _, groups, _result, _ = full
    new = {pg for entry in groups["nets"] for pg in entry["phys_groups"]}
    volumes = {info["phys_group"] for info in groups["volumes"].values()}
    assert not new & volumes


def test_the_mesh_nodes_of_each_group_lie_on_its_net(full) -> None:
    """Independent of gmsh: read the written file and compare with the shapes."""
    import shapely

    _, groups, result, nets = full
    nodes = surface_nodes_by_group(result.mesh_path)
    checked = 0
    for entry in groups["nets"]:
        net = next(n for n in nets if n.name == entry["name"])
        union = shapely.unary_union(
            [
                shape
                for items in nets._shapes.values()
                for shape, owner in items
                if owner is net
            ]
        )
        for name, xy in nodes.items():
            if not name.endswith(f"@{entry['label']}"):
                continue
            distances = shapely.distance(union, shapely.points(xy))
            assert distances.max() < 0.06, (name, distances.max())
            checked += 1
    assert checked == 5 + 5 + 2


def test_splitting_does_not_change_the_mesh(full, tmp_path) -> None:
    _, _, result, _ = full
    _, _, layer_result = _mesh(
        tmp_path, FULL, [("T1", {"layer": "m1"}), ("T2", {"layer": "m2"})]
    )
    assert result.mesh_stats["tetrahedra"] == layer_result.mesh_stats["tetrahedra"]
    assert result.mesh_stats["nodes"] == layer_result.mesh_stats["nodes"]


def test_without_net_terminals_there_are_no_net_groups(tmp_path) -> None:
    _sim, groups, _ = _mesh(
        tmp_path, FULL, [("T1", {"layer": "m1"}), ("T2", {"layer": "m2"})]
    )
    assert "nets" not in groups
    assert set(groups["conductor_surfaces"]) == {"m1_xy", "m1_z", "m2_xy", "m2_z"}


def test_two_separate_electrodes_on_one_layer(tmp_path) -> None:
    _sim, groups, result = _mesh(
        tmp_path,
        [A, B],
        [("L", {"net": "PLUS"}), ("R", {"net": "MINUS"})],
    )
    names = physical_names(result.mesh_path)
    assert len(groups["nets"]) == 2
    for entry, label in zip(groups["nets"], ["PLUS", "MINUS"], strict=True):
        assert {names[pg] for pg in entry["phys_groups"]} == {
            f"m1_xy@{label}",
            f"m1_z@{label}",
        }
    assert "m1_xy" not in groups["conductor_surfaces"]


def test_a_floating_island_alone_gets_groups_in_floating_mode(tmp_path) -> None:
    _, groups, result = _mesh(
        tmp_path,
        FULL,
        [("TOP", {"net": "PLUS"}), ("LEFT", {"net": "MINUS"})],
        unassigned="floating",
    )
    names = physical_names(result.mesh_path)
    assert {names[pg] for pg in groups["nets"][2]["phys_groups"]} == {
        "m1_xy@net3",
        "m1_z@net3",
    }


def test_planar_conductors_are_split_by_net_too(tmp_path) -> None:
    """Conductors as 2D PEC surfaces: the groups are ``<layer>_pec`` per layer."""
    sim = make_sim(tmp_path / "es", FULL)
    sim.add_terminal("TOP", net="PLUS")
    sim.add_terminal("LEFT", net="MINUS")
    result = sim.mesh(planar_conductors=True, **MESH)
    groups = sim._last_mesh_result.groups
    names = physical_names(result.mesh_path)
    assert groups["conductor_surfaces"] == {}
    assert set(groups["pec_surfaces"]) == {
        "m1@PLUS",
        "m1@MINUS",
        "m1@net3",
        "m2@PLUS",
        "m2@MINUS",
    }
    plus = groups["nets"][0]
    assert {names[pg] for pg in plus["phys_groups"]} == {
        "m1_pec@PLUS",
        "m2_pec@PLUS",
        "sio2__v1@PLUS",
    }
    # the nodes of each planar group lie on its own net
    import shapely

    nets = nets_of(FULL, PORTS)
    nodes = surface_nodes_by_group(result.mesh_path)
    for k, entry in enumerate(groups["nets"]):
        union = shapely.unary_union(
            [s for items in nets._shapes.values() for s, o in items if o is nets[k]]
        )
        for name, xy in nodes.items():
            if name.endswith(f"@{entry['label']}"):
                assert shapely.distance(union, shapely.points(xy)).max() < 0.06
    sim.write_config()
    config = json.loads((tmp_path / "es" / "config.json").read_text())
    boundaries = config["Boundaries"]
    assert len(boundaries["Terminal"]) == 2
    island = sorted(groups["nets"][2]["phys_groups"])
    assert boundaries["Ground"]["Attributes"] == island


def test_shapes_touching_at_a_corner_stay_two_nets_in_the_mesh(tmp_path) -> None:
    """A single shared point does not conduct; OCC keeps their faces apart."""
    boxes = [("m1", 0, 0, 4, 4), ("m1", 4, 4, 8, 8)]
    ports = [("A", 2, 2, "m1"), ("B", 6, 6, "m1")]
    _, groups, result = _mesh(
        tmp_path, boxes, [("A", {"net": "A"}), ("B", {"net": "B"})], ports=ports
    )
    names = physical_names(result.mesh_path)
    assert [n["name"] for n in groups["nets"]] == ["A", "B"]
    for entry, label in zip(groups["nets"], ["A", "B"], strict=True):
        assert {names[pg] for pg in entry["phys_groups"]} == {
            f"m1_xy@{label}",
            f"m1_z@{label}",
        }


def test_vias_of_one_net_merge_and_still_get_their_net(tmp_path) -> None:
    """Default merge distance: two vias 1 um apart become one volume of the net."""
    boxes = [
        ("m1", 0, 0, 4, 4),
        ("v1", 0.5, 1, 1, 2),
        ("v1", 2, 1, 2.5, 2),
        ("m2", 0, 0, 4, 4),
    ]
    # a second, separate net so that there are two terminals
    sim = make_sim(
        tmp_path / "es", [*boxes, ("m1", 10, 0, 14, 4)], [("PLUS", 2, 2, "m1")]
    )
    sim.add_terminal("A", net="PLUS")
    sim.add_terminal("B", region=(10, 0, 14, 4), layer="m1")
    result = sim.mesh(refined_mesh_size=0.5, max_mesh_size=3.0, merge_via_distance=2.0)
    groups = sim._last_mesh_result.groups
    names = physical_names(result.mesh_path)
    assert [names[pg] for pg in groups["via_boundary_surfaces"]["v1@PLUS"]] == [
        "sio2__v1@PLUS"
    ]
    assert result.mesh_stats["tetrahedra"] > 0


def test_via_merge_conflict_fails_before_meshing(tmp_path) -> None:
    sim = make_sim(tmp_path / "es", TWO_NETS, [])
    sim.add_terminal("L", region=(0, 0, 4, 4), layer="m1")
    sim.add_terminal("R", region=(5, 0, 9, 4), layer="m1")
    with pytest.raises(ValueError, match=r"Via layer 'v1'.*different nets"):
        sim.mesh(refined_mesh_size=0.5, max_mesh_size=3.0, merge_via_distance=2.0)
    assert not (tmp_path / "es" / "palace.msh").exists()


def test_generate_mesh_refuses_to_write_the_config_with_net_groups(tmp_path) -> None:
    from gsim.palace.mesh.generator import generate_mesh

    with pytest.raises(ValueError, match="needs write_config=False"):
        generate_mesh(
            component(FULL, PORTS),
            stack(),
            [],
            tmp_path,
            simulation_type="electrostatic",
            split_conductors_by_net=True,
            write_config=True,
        )


# ----- labels -------------------------------------------------------------------


def test_net_labels_are_the_names_when_unique() -> None:
    nets = nets_of()
    assert net_labels(nets) == ["PLUS", "MINUS", "net3"]


def test_net_labels_stay_unique_when_port_names_repeat() -> None:
    nets = nets_of(
        [("m1", 0, 0, 4, 4), ("m1", 6, 0, 10, 4)],
        [("SIG", 2, 2, "m1"), ("SIG", 8, 2, "m1")],
    )
    assert [n.name for n in nets] == ["SIG", "SIG"]
    assert net_labels(nets) == ["SIG", "SIG#2"]
