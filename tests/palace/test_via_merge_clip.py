"""Tests for clipping merged vias to the metal around them (gsim#411, option 1).

Merging joins the cuts of a via layer that are closer than
``merge_via_distance`` into one slab. ``_clip_merged_vias`` keeps the drawn
cuts and cuts the slab back to the overlap of the conductor layers directly
below and above the via layer (the *landing area*). No mesh is made here; the
meshes are in ``test_via_merge_clip_mesh.py``.
"""

from __future__ import annotations

import logging

import pytest
from shapely import GeometryCollection, LineString, MultiPolygon
from shapely import Polygon as ShapelyPolygon
from shapely.geometry import box
from shapely.ops import unary_union

from gsim.common.stack.extractor import Layer, LayerStack
from gsim.palace.mesh import geometry
from gsim.palace.mesh.geometry import (
    _clip_merged_vias,
    _merge_via_layer,
    _merge_via_polygons,
    _polygons_by_gds_layer,
    _shapes_to_via_polys,
    _union_area,
    _via_landing_area,
    _via_neighbour_conductors,
    _via_shapes,
    _warn_if_via_merge_inflates_area,
)
from tests._electrode_layouts import stack as generic_stack

LOGGER = geometry.logger.name
SIDE = 0.4  # um, the side of a drawn cut


def _cut(x: float, y: float, side: float = SIDE):
    """A square via polygon in the (pts_x, pts_y, holes) form of the mesher."""
    return ([x, x + side, x + side, x], [y, y, y + side, y + side], [])


def _rect(x0: float, y0: float, x1: float, y1: float):
    """A rectangular polygon in the form of the mesher."""
    return ([x0, x1, x1, x0], [y0, y0, y1, y1], [])


def row():
    """Ten cuts in a row at pitch 1.0 um."""
    return [_cut(i * 1.0, 0.0) for i in range(10)]


def diagonal():
    """Ten cuts in a zigzag of pitch 1.0 um."""
    return [_cut(i * 1.0, (i % 2) * 1.0) for i in range(10)]


def array():
    """A 2 x 10 array of cuts at pitch 1.5 um."""
    return [_cut(i * 1.5, j * 1.5) for i in range(10) for j in range(2)]


def _ihp_like_stack() -> LayerStack:
    """The z-ranges of the IHP stack around ``cont``, ``via1`` and ``vmim``."""
    rows = {
        "active": (1, 0.0, 0.201, "conductor"),
        "poly": (5, 0.2, 0.36, "conductor"),
        "polyres": (128, 0.2, 0.36, "conductor"),
        "cont": (6, 0.2, 0.84, "via"),
        "emwind": (33, 0.36, 0.39, "conductor"),
        "metal1": (8, 0.84, 1.26, "conductor"),
        "via1": (19, 1.26, 1.8, "via"),
        "metal2": (10, 1.8, 2.29, "conductor"),
        "metal5": (67, 4.89, 5.38, "conductor"),
        "vmim": (129, 5.58, 6.24, "via"),
        "topmetal1": (126, 6.23, 8.23, "conductor"),
    }
    return LayerStack(
        layers={
            name: Layer(
                name=name,
                gds_layer=(number, 0),
                zmin=z0,
                zmax=z1,
                thickness=z1 - z0,
                material="aluminum",
                layer_type=kind,
            )
            for name, (number, z0, z1, kind) in rows.items()
        },
        materials={"aluminum": {"type": "conductor", "conductivity": 3.77e7}},
        dielectrics=[],
    )


# ----- the polygon conversion ------------------------------------------------------


def test_shapes_to_via_polys_keeps_holes_and_drops_the_closing_vertex() -> None:
    shape = box(0, 0, 4, 4).difference(box(1, 1, 3, 3))
    ((xs, ys, holes),) = _shapes_to_via_polys(shape)
    assert len(xs) == len(ys) == 4
    assert len(holes) == 1
    assert len(holes[0][0]) == 4


def test_shapes_to_via_polys_takes_the_polygons_of_a_collection() -> None:
    parts = GeometryCollection(
        [box(0, 0, 1, 1), LineString([(2, 0), (3, 0)]), MultiPolygon([box(5, 0, 6, 1)])]
    )
    assert len(_shapes_to_via_polys(parts)) == 2
    assert _shapes_to_via_polys(ShapelyPolygon()) == []


def test_shapes_to_via_polys_drops_small_polygons() -> None:
    parts = MultiPolygon([box(0, 0, 1, 1), box(5, 0, 5.0005, 0.0005)])  # 2.5e-7 um^2
    assert len(_shapes_to_via_polys(parts)) == 2
    assert len(_shapes_to_via_polys(parts, min_area=1e-6)) == 1


# ----- the metal below and above a via layer ----------------------------------------


def test_cont_lands_on_active_poly_and_polyres_under_metal1() -> None:
    below, above = _via_neighbour_conductors(_ihp_like_stack(), "cont")
    assert below == ["active", "poly", "polyres"]
    assert above == ["metal1"]
    assert "emwind" not in below + above  # [0.36, 0.39] lies inside the cut


def test_a_via_between_two_metals_has_one_neighbour_on_each_side() -> None:
    assert _via_neighbour_conductors(_ihp_like_stack(), "via1") == (
        ["metal1"],
        ["metal2"],
    )


def test_vmim_has_nothing_below_and_topmetal1_above() -> None:
    # topmetal1 starts 0.01 um below the top of vmim: still touching it
    assert _via_neighbour_conductors(_ihp_like_stack(), "vmim") == ([], ["topmetal1"])


def test_generic_stack_neighbours() -> None:
    assert _via_neighbour_conductors(generic_stack(), "v1") == (["m1"], ["m2"])


# ----- the landing area -------------------------------------------------------------


def test_landing_is_the_overlap_of_the_metal_below_and_above() -> None:
    by_gds = _polygons_by_gds_layer(
        [(1, *_rect(0, 0, 4, 4)), (2, *_rect(2, 2, 6, 6)), (3, *_cut(0, 0))]
    )
    landing = _via_landing_area(by_gds, generic_stack(), "v1")
    assert landing.area == pytest.approx(4.0)


def test_landing_is_empty_when_a_side_has_no_polygons() -> None:
    only_m1 = _polygons_by_gds_layer([(1, *_rect(0, 0, 4, 4))])
    assert _via_landing_area(only_m1, generic_stack(), "v1").is_empty
    only_m2 = _polygons_by_gds_layer([(2, *_rect(0, 0, 4, 4))])
    assert _via_landing_area(only_m2, generic_stack(), "v1").is_empty


def test_landing_is_empty_when_a_side_has_no_layer() -> None:
    by_gds = _polygons_by_gds_layer(
        [(126, *_rect(0, 0, 4, 4)), (129, *_cut(1, 1))],
    )
    assert _via_landing_area(by_gds, _ihp_like_stack(), "vmim").is_empty


def test_landing_unites_the_layers_of_one_side() -> None:
    by_gds = _polygons_by_gds_layer(
        [
            (1, *_rect(0, 0, 2, 2)),  # active
            (5, *_rect(2, 0, 4, 2)),  # poly
            (8, *_rect(0, 0, 4, 2)),  # metal1
        ]
    )
    landing = _via_landing_area(by_gds, _ihp_like_stack(), "cont")
    assert landing.area == pytest.approx(8.0)


def test_polygons_are_grouped_by_gds_layer_in_order() -> None:
    grouped = _polygons_by_gds_layer(
        [(1, *_cut(0, 0)), (2, *_cut(1, 1)), (1, *_cut(2, 2))]
    )
    assert sorted(grouped) == [1, 2]
    assert [p[0][0] for p in grouped[1]] == [0, 2]


# ----- clipping ---------------------------------------------------------------------

EMPTY = ShapelyPolygon()


def _pads(cells, pad=0.6, margin=0.1):
    """One landing pad per cut (``cells`` are the lower-left corners)."""
    return unary_union(
        [
            box(x - margin, y - margin, x - margin + pad, y - margin + pad)
            for x, y in cells
        ]
    )


def _corners(polys):
    return [(p[0][0], p[1][0]) for p in polys]


def _landing(case: str, layout: str):
    """The landing area of a scenario over one of the issue layouts."""
    polys = {"row": row, "diagonal": diagonal, "array": array}[layout]()
    xs = [c for p in polys for c in p[0]]
    ys = [c for p in polys for c in p[1]]
    bar = box(min(xs) - 0.1, min(ys) - 0.1, max(xs) + 0.1, max(ys) + 0.1)
    if case == "enclosed":
        return bar
    if case == "empty":
        return EMPTY
    if case == "pads":
        return _pads(_corners(polys)).intersection(bar.buffer(1))
    raise AssertionError(case)


def _run(layout: str, landing, distance: float = 2.0):
    drawn = {"row": row, "diagonal": diagonal, "array": array}[layout]()
    merged = _merge_via_polygons(drawn, distance)
    clipped = _clip_merged_vias(drawn, merged, landing)
    return drawn, merged, clipped


def _ratio(drawn, polys) -> float:
    return _union_area(_via_shapes(polys)) / _union_area(_via_shapes(drawn))


def _check_invariants(drawn, merged, clipped, landing) -> None:
    drawn_u = unary_union(_via_shapes(drawn))
    merged_u = unary_union(_via_shapes(merged))
    clipped_u = unary_union(_via_shapes(clipped))
    assert clipped_u.difference(merged_u).area < 1e-6  # clipped is inside merged
    assert drawn_u.difference(clipped_u).area < 1e-6  # no drawn cut is lost
    # nothing outside the landing area except the drawn cuts themselves
    assert clipped_u.difference(unary_union([landing, drawn_u])).area < 1e-6


@pytest.mark.parametrize("layout", ["row", "diagonal", "array"])
def test_a_slab_inside_the_landing_area_is_unchanged(layout) -> None:
    """Fully enclosed by metal on both sides: the merged tuples come back."""
    landing = _landing("enclosed", layout)
    drawn, merged, clipped = _run(layout, landing)
    assert len(merged) == 1
    assert clipped == merged
    assert all(a is b for a, b in zip(clipped, merged, strict=True))
    clipped_u = unary_union(_via_shapes(clipped))
    merged_u = unary_union(_via_shapes(merged))
    assert clipped_u.symmetric_difference(merged_u).area == 0
    _check_invariants(drawn, merged, clipped, landing)


@pytest.mark.parametrize(
    ("layout", "n_cuts"), [("row", 10), ("diagonal", 10), ("array", 20)]
)
def test_with_no_metal_on_one_side_the_cuts_stay_as_drawn(layout, n_cuts) -> None:
    drawn, merged, clipped = _run(layout, _landing("empty", layout))
    assert len(merged) == 1
    assert len(clipped) == n_cuts
    assert _ratio(drawn, clipped) == pytest.approx(1.0, abs=1e-6)
    _check_invariants(drawn, merged, clipped, EMPTY)


@pytest.mark.parametrize(
    ("layout", "n_after", "ratio"),
    [("row", 10, 1.45), ("diagonal", 10, 1.69), ("array", 20, 1.81)],
)
def test_per_cut_landing_pads_keep_the_cuts_apart(layout, n_after, ratio) -> None:
    landing = _landing("pads", layout)
    drawn, merged, clipped = _run(layout, landing)
    assert len(merged) == 1
    assert len(clipped) == n_after
    assert _ratio(drawn, clipped) == pytest.approx(ratio, abs=0.01)
    assert _ratio(drawn, clipped) < _ratio(drawn, merged)
    _check_invariants(drawn, merged, clipped, landing)


def test_one_metal_strip_per_row_keeps_the_two_rows_apart() -> None:
    drawn = array()
    strips = unary_union([box(-0.1, -0.1, 13.95, 0.5), box(-0.1, 1.4, 13.95, 2.0)])
    merged = _merge_via_polygons(drawn, 2.0)
    clipped = _clip_merged_vias(drawn, merged, strips)
    assert len(merged) == 1
    assert len(clipped) == 2
    assert _ratio(drawn, clipped) == pytest.approx(4.34, abs=0.01)
    _check_invariants(drawn, merged, clipped, strips)


def test_a_landing_piece_without_a_cut_is_dropped() -> None:
    drawn = [_cut(0.0, 0.0), _cut(1.5, 0.0)]
    island = unary_union([box(-0.1, -0.1, 0.5, 0.5), box(0.6, 0.0, 0.9, 0.4)])
    merged = _merge_via_polygons(drawn, 2.0)
    clipped = _clip_merged_vias(drawn, merged, island)
    assert len(merged) == 1
    areas = sorted(
        round(ShapelyPolygon(list(zip(p[0], p[1], strict=True))).area, 4)
        for p in clipped
    )
    assert areas == [0.16, 0.2]  # the 0.12 um^2 island holds no cut
    _check_invariants(drawn, merged, clipped, island)


def test_a_ring_of_cuts_on_a_metal_frame_keeps_its_hole() -> None:
    cuts = (
        [_cut(x, 0) for x in range(5)]
        + [_cut(x, 4) for x in range(5)]
        + [_cut(0, y) for y in (1, 2, 3)]
        + [_cut(4, y) for y in (1, 2, 3)]
    )
    frame = box(-0.1, -0.1, 4.5, 4.5).difference(box(0.5, 0.5, 3.9, 3.9))
    merged = _merge_via_polygons(cuts, 2.0)
    clipped = _clip_merged_vias(cuts, merged, frame)
    assert len(merged) == 1
    assert len(merged[0][2]) == 1
    assert clipped == merged
    _check_invariants(cuts, merged, clipped, frame)


def test_nothing_merged_means_nothing_to_clip() -> None:
    drawn = [_cut(0, 0), _cut(10, 0)]
    merged = _merge_via_polygons(drawn, 2.0)
    assert len(merged) == 2
    clipped = _clip_merged_vias(drawn, merged, EMPTY)
    assert len(clipped) == 2
    assert _clip_merged_vias(drawn, drawn, EMPTY) is drawn  # distance 0 or one via


def test_touching_landing_edges_do_not_become_pieces() -> None:
    """A landing strip that only shares an edge with a cut gives a line: dropped."""
    drawn = [_cut(0.0, 0.0), _cut(1.0, 0.0)]
    landing = box(0.4, 0.0, 1.0, 0.4)  # fills the gap, shares edges with both cuts
    merged = _merge_via_polygons(drawn, 2.0)
    clipped = _clip_merged_vias(drawn, merged, landing)
    assert len(clipped) == 2
    assert _ratio(drawn, clipped) == pytest.approx(1.0, abs=1e-6)


# ----- the shared entry point -------------------------------------------------------


def test_merge_via_layer_without_landing_is_the_plain_merge() -> None:
    assert _merge_via_layer(array(), 2.0) == _merge_via_polygons(array(), 2.0)
    assert _merge_via_layer(array(), 2.0, None) == _merge_via_polygons(array(), 2.0)


def test_merge_via_layer_with_landing_is_the_clipped_merge() -> None:
    drawn = array()
    landing = _landing("pads", "array")
    merged = _merge_via_polygons(drawn, 2.0)
    assert _merge_via_layer(drawn, 2.0, landing) == _clip_merged_vias(
        drawn, merged, landing
    )


def test_merge_via_layer_calls_merge_via_polygons_by_its_module_name(
    monkeypatch,
) -> None:
    """The monkeypatch of the curved-meshing tests keeps working."""
    seen = []

    def _spy(polys, distance):
        seen.append((len(polys), distance))
        return polys

    monkeypatch.setattr(geometry, "_merge_via_polygons", _spy)
    drawn = array()
    assert _merge_via_layer(drawn, 2.0, EMPTY) is drawn
    assert seen == [(20, 2.0)]


def test_merge_distance_zero_keeps_the_input_even_with_a_landing() -> None:
    drawn = array()
    assert _merge_via_layer(drawn, 0.0, EMPTY) is drawn


def test_clipping_is_logged_at_info(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _merge_via_layer(array(), 2.0, EMPTY, label="'v1'")
    (record,) = [r for r in caplog.records if "Via clipping" in r.getMessage()]
    assert record.levelno == logging.INFO
    assert "'v1'" in record.getMessage()
    assert "1 merged polygons -> 20" in record.getMessage()


def test_nothing_is_logged_when_clipping_changes_nothing(caplog) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _merge_via_layer(row(), 2.0, _landing("enclosed", "row"))
    assert not [r for r in caplog.records if "Via clipping" in r.getMessage()]


# ----- the area ratio in the log (option 2) ----------------------------------------


def test_small_growth_is_logged_at_info(caplog) -> None:
    # two 1 um squares 0.01 um apart: they merge, the bridge adds 0.5 % area
    polys = [_rect(0, 0, 1, 1), _rect(1.01, 0, 2.01, 1)]
    merged = _merge_via_polygons(polys, 2.0)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        _warn_if_via_merge_inflates_area("via1", polys, merged, 2.0)
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    ratio_lines = [r for r in infos if "factor of 1.005" in r.getMessage()]
    assert len(ratio_lines) == 1
    assert "Via layer 'via1'" in ratio_lines[0].getMessage()
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_the_warning_points_to_clipping_only_when_it_is_off(caplog) -> None:
    drawn = array()
    merged = _merge_via_polygons(drawn, 2.0)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _warn_if_via_merge_inflates_area("v1", drawn, merged, 2.0)
        _warn_if_via_merge_inflates_area(
            "v1", drawn, merged, 2.0, clip_merged_vias=True
        )
    off, on = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert "Set clip_merged_vias=True" in off
    assert "clipping only keeps them" not in off
    assert "Set clip_merged_vias" not in on
    assert "clipping only keeps them inside the metal above and below" in on
