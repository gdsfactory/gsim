"""Review findings on ``clip_merged_vias`` (gsim PR #423).

1. In net mode a clipped via volume that overlaps the vias of one net, but
   reaches towards the vias of another, must be rejected before meshing.
2. One GDS layer that backs two via layers: each level starts from the drawn
   polygons, not from the merged and clipped result of the level before.
3. The sliver filter applies to added landing pieces, never to a drawn cut.
"""

from __future__ import annotations

import gmsh
import pytest
from shapely.geometry import box
from shapely.ops import unary_union

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace.mesh.geometry import (
    GeometryData,
    _clip_merged_vias,
    _merge_via_layer,
    _merge_via_polygons,
    _polygons_by_gds_layer,
    _via_landing_area,
    _via_shapes,
    add_metals,
    extract_geometry,
)
from gsim.palace.mesh.net_groups import check_via_merging
from gsim.palace.mesh.nets import extract_nets
from tests._electrode_layouts import component, make_sim, stack

# ----- 1. a clipped lobe towards another net ------------------------------------------


def _lobe_layout(reach: float):
    """Two nets side by side; the metal of the first reaches out to ``reach``.

    Net 1 has m1 and m2 from x = 0 to ``reach`` and a via at 3.5 .. 4; net 2 has
    m1 and m2 from x = 5 and a via at 5 .. 5.5. The vias are 1 um apart, so they
    merge at 2 um; the clipped volume of net 1 is the landing area that the
    merged slab covers, which reaches ``reach``.
    """
    return [
        ("m1", 0, 0, reach, 4),
        ("m1", 5, 0, 9, 4),
        ("m2", 0, 0, reach, 4),
        ("m2", 5, 0, 9, 4),
        ("v1", 3.5, 1, 4, 2),
        ("v1", 5, 1, 5.5, 2),
    ]


def _check(boxes, distance: float, **kwargs) -> None:
    c = component(boxes, ports=[])
    st = stack()
    geometry = extract_geometry(c, st)
    nets = extract_nets(geometry, st, c.ports)
    assert len(nets) == 2
    check_via_merging(geometry, st, nets, distance, **kwargs)


def test_a_lobe_nearer_to_the_other_net_is_rejected_before_meshing() -> None:
    """Metal out to 4.8 puts the lobe 0.2 um from net 2's via, 0.8 um from its own."""
    with pytest.raises(ValueError, match=r"Via layer 'v1'.*clip_merged_vias") as error:
        _check(_lobe_layout(4.8), 2.0, clip_merged_vias=True)
    message = str(error.value)
    assert "'net1'" in message
    assert "'net2'" in message
    assert "0.2 um from the vias of net 'net2'" in message
    assert "merge_via_distance" in message


def test_the_lobe_check_names_a_merge_distance_that_works() -> None:
    """The gap between the vias is 1 um: 0.9 keeps them apart and passes."""
    _check(_lobe_layout(4.8), 0.9, clip_merged_vias=True)


def test_a_lobe_nearer_to_its_own_net_passes() -> None:
    """Metal out to 4.3: the lobe stays 0.3 um from its own via, 0.7 from the other."""
    _check(_lobe_layout(4.3), 2.0, clip_merged_vias=True)


def test_without_clipping_the_slab_is_still_the_short_it_was() -> None:
    with pytest.raises(ValueError, match="different nets"):
        _check(_lobe_layout(4.8), 2.0, clip_merged_vias=False)


MESH = {"refined_mesh_size": 0.5, "max_mesh_size": 3.0}
PORTS = [("PLUS", 2, 2, "m1"), ("MINUS", 7, 2, "m1")]


def _net_sim(tmp_path, boxes):
    sim = make_sim(tmp_path / "es", boxes, PORTS)
    sim.add_terminal("TOP", net="PLUS")
    sim.add_terminal("LEFT", net="MINUS")
    return sim


def test_the_net_mesh_stops_at_the_check_not_at_the_face_assignment(tmp_path) -> None:
    sim = _net_sim(tmp_path, _lobe_layout(4.8))
    with pytest.raises(ValueError, match=r"with clip_merged_vias the via volume"):
        sim.mesh(**MESH, merge_via_distance=2.0, clip_merged_vias=True)


def test_a_safe_lobe_meshes_with_one_via_group_per_net(tmp_path) -> None:
    sim = _net_sim(tmp_path, _lobe_layout(4.3))
    sim.mesh(**MESH, merge_via_distance=2.0, clip_merged_vias=True)
    groups = sim._last_mesh_result.groups
    assert set(groups["via_boundary_surfaces"]) == {"v1@PLUS", "v1@MINUS"}


# ----- 2. one GDS layer, two via layers -----------------------------------------------

LEVELS = {
    "m1": (1, 0.0, 1.0, "conductor", "aluminum"),
    "v1": (3, 1.0, 2.0, "via", "tungsten"),
    "m2": (2, 2.0, 3.0, "conductor", "aluminum"),
    "v2": (3, 3.0, 4.0, "via", "tungsten"),  # the same GDS layer as v1
    "m3": (4, 4.0, 5.0, "conductor", "aluminum"),
}
CELLS = (0.0, 1.0)


def _two_level_stack() -> LayerStack:
    return LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=(number, 0),
                zmin=z0,
                zmax=z1,
                thickness=z1 - z0,
                material=material,
                layer_type=kind,
            )
            for name, (number, z0, z1, kind, material) in LEVELS.items()
        },
        materials={
            "aluminum": {"type": "conductor", "conductivity": 3.77e7},
            "tungsten": {"type": "conductor", "conductivity": 1.8e7},
        },
        dielectrics=[],
    )


def _rect(layer: int, x0: float, y0: float, x1: float, y1: float):
    return (layer, [x0, x1, x1, x0], [y0, y0, y1, y1], [])


def _two_level_polygons():
    """Two cuts on GDS layer 3 under one m2 sheet; m1 is a sheet, m3 has pads."""
    cuts = [_rect(3, x, 0, x + 0.4, 0.4) for x in CELLS]
    pads = [_rect(4, x - 0.1, -0.1, x + 0.5, 0.5) for x in CELLS]
    return [
        _rect(1, -0.5, -0.5, 2.0, 1.0),
        _rect(2, -0.5, -0.5, 2.0, 1.0),
        *cuts,
        *pads,
    ]


@pytest.fixture
def occ():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("levels")
    try:
        yield gmsh.model.occ
    finally:
        gmsh.finalize()


def _volume(occ, tags) -> float:
    return sum(occ.getMass(3, tag) for tag in tags)


def test_the_second_via_level_is_clipped_from_the_drawn_cuts(occ) -> None:
    polygons = _two_level_polygons()
    geometry = GeometryData(polygons=polygons, bbox=(0, 0, 2, 1), layer_bboxes={})
    tags = add_metals(
        occ, geometry, _two_level_stack(), merge_via_distance=2.0, clip_merged_vias=True
    )

    # what each level must give, from the drawn cuts and its own landing area
    by_gds = _polygons_by_gds_layer(polygons)
    drawn = by_gds[3]
    st = _two_level_stack()
    expected = {}
    for name in ("v1", "v2"):
        landing = _via_landing_area(by_gds, st, name)
        expected[name] = _merge_via_layer(drawn, 2.0, landing)

    # level 1 sits between two full sheets: one slab; level 2 has pads on m3
    assert len(expected["v1"]) == 1
    assert len(expected["v2"]) == 2
    for name in ("v1", "v2"):
        volumes = tags[name]["volumes"]
        assert len(volumes) == len(expected[name])
        area = unary_union(_via_shapes(expected[name])).area
        assert _volume(occ, volumes) == pytest.approx(area * 1.0, rel=1e-6)
    assert _volume(occ, tags["v1"]["volumes"]) == pytest.approx(0.56, rel=1e-6)
    assert _volume(occ, tags["v2"]["volumes"]) == pytest.approx(0.40, rel=1e-6)


def test_the_second_via_level_without_clipping_merges_the_drawn_cuts(occ) -> None:
    geometry = GeometryData(
        polygons=_two_level_polygons(), bbox=(0, 0, 2, 1), layer_bboxes={}
    )
    tags = add_metals(occ, geometry, _two_level_stack(), merge_via_distance=2.0)
    for name in ("v1", "v2"):
        assert len(tags[name]["volumes"]) == 1
        assert _volume(occ, tags[name]["volumes"]) == pytest.approx(0.56, rel=1e-6)


# ----- 3. the sliver filter and drawn cuts --------------------------------------------


def _cut(x: float, y: float, side: float):
    return ([x, x + side, x + side, x], [y, y, y + side, y + side], [])


@pytest.mark.parametrize("side", [0.001, 0.0005, 0.002])
def test_a_tiny_drawn_cut_is_never_dropped(side) -> None:
    """The cut is at most 1e-6 um^2 for the first two sides; the landing misses it."""
    drawn = [_cut(0.0, 0.0, 0.4), _cut(1.0, 0.0, side)]
    merged = _merge_via_polygons(drawn, 2.0)
    assert len(merged) == 1
    landing = box(-0.1, -0.1, 0.5, 0.5)  # only under the big cut
    clipped = _clip_merged_vias(drawn, merged, landing)
    assert len(clipped) == 2
    kept = unary_union(_via_shapes(clipped))
    for cut in _via_shapes(drawn):
        assert cut.difference(kept).area == pytest.approx(0.0, abs=1e-12)
    tiny = min(_via_shapes(clipped), key=lambda shape: shape.area)
    assert tiny.area == pytest.approx(side * side, rel=1e-6)


def test_a_sliver_of_landing_is_still_not_added() -> None:
    """A landing piece of 9e-7 um^2 that sticks out of a cut is left out."""
    drawn = [_cut(0.0, 0.0, 0.4), _cut(1.0, 0.0, 0.4)]
    merged = _merge_via_polygons(drawn, 2.0)
    landing = box(0.4 - 0.0005, 0.1, 0.4 + 0.0005, 0.1 + 0.0009)
    clipped = _clip_merged_vias(drawn, merged, landing)
    assert len(clipped) == 2
    assert [shape.area for shape in _via_shapes(clipped)] == pytest.approx(
        [0.16, 0.16], abs=1e-12
    )


def test_a_piece_above_the_sliver_limit_is_still_added() -> None:
    drawn = [_cut(0.0, 0.0, 0.4), _cut(1.0, 0.0, 0.4)]
    merged = _merge_via_polygons(drawn, 2.0)
    landing = box(0.4 - 0.1, 0.1, 0.4 + 0.1, 0.3)  # 0.04 um^2, half outside the cut
    clipped = _clip_merged_vias(drawn, merged, landing)
    assert len(clipped) == 2
    assert sorted(shape.area for shape in _via_shapes(clipped)) == pytest.approx(
        [0.16, 0.16 + 0.1 * 0.2], rel=1e-6
    )
