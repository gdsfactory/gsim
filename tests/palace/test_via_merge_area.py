"""Tests for the warning when via merging grows the via area (gsim#411).

``_merge_via_polygons`` fills the gaps between vias closer than the merge
distance, so a dense array of small cuts becomes one slab with several times
the drawn area. ``add_metals`` logs one warning per via layer when the merged
area is more than ``VIA_MERGE_AREA_WARN_RATIO`` times the drawn area.

The area checks need no mesh; the last two tests mesh a small layout of the
synthetic stack of ``_electrode_layouts``.
"""

from __future__ import annotations

import logging
import re

import pytest

from gsim.palace.mesh import geometry
from gsim.palace.mesh.geometry import (
    VIA_MERGE_AREA_WARN_RATIO,
    _merge_via_polygons,
    _smallest_via_gap,
    _via_shapes,
    _warn_if_via_merge_inflates_area,
)
from tests._electrode_layouts import make_sim

LOGGER = geometry.logger.name
SIDE = 0.4  # um, the side of a drawn cut


def _cut(x: float, y: float, side: float = SIDE):
    """A square via polygon in the (pts_x, pts_y, holes) form of the mesher."""
    return ([x, x + side, x + side, x], [y, y, y + side, y + side], [])


def _rect(x0: float, y0: float, x1: float, y1: float, holes=()):
    """A rectangular via polygon, optionally with holes ``(hx, hy)``."""
    return ([x0, x1, x1, x0], [y0, y0, y1, y1], list(holes))


def row():
    """Ten cuts in a row at pitch 1.0 um (gap 0.6 um)."""
    return [_cut(i * 1.0, 0.0) for i in range(10)]


def diagonal():
    """Ten cuts in a zigzag of pitch 1.0 um: neighbours 1.41 um apart."""
    return [_cut(i * 1.0, (i % 2) * 1.0) for i in range(10)]


def array():
    """A 2 x 10 array of cuts at pitch 1.5 um (gap 1.1 um)."""
    return [_cut(i * 1.5, j * 1.5) for i in range(10) for j in range(2)]


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def _check(caplog, polys, distance: float, name: str = "via1"):
    """Merge as ``add_metals`` does and run the area check; return its ratio."""
    merged = _merge_via_polygons(polys, distance)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        ratio = _warn_if_via_merge_inflates_area(name, polys, merged, distance)
    return ratio, merged


# (layout, merge distance, polygons after merging, area ratio or None if not merged)
CASES = [
    (row, 2.0, 1, 2.35),
    (row, 1.0, 1, 2.35),
    (row, 0.0, 10, None),
    (diagonal, 2.0, 1, 6.975),
    (diagonal, 1.0, 10, None),  # corner to corner the necks are too thin to join
    (diagonal, 0.0, 10, None),
    (array, 2.0, 1, 8.25),
    (array, 1.0, 20, None),  # the 1.1 um gaps are wider than 1.0 um
    (array, 0.0, 20, None),
]


@pytest.mark.parametrize(
    ("layout", "distance", "n_after", "expected"),
    CASES,
    ids=[f"{c[0].__name__}-{c[1]:g}" for c in CASES],
)
def test_warns_exactly_when_the_area_grows(
    caplog, layout, distance, n_after, expected
) -> None:
    polys = layout()
    ratio, merged = _check(caplog, polys, distance)
    assert len(merged) == n_after
    if expected is None:
        assert ratio is None
        assert _warnings(caplog) == []
        return
    assert ratio == pytest.approx(expected, abs=0.005)
    assert ratio > VIA_MERGE_AREA_WARN_RATIO
    assert len(_warnings(caplog)) == 1


def test_the_threshold_is_five_percent() -> None:
    assert VIA_MERGE_AREA_WARN_RATIO == 1.05


def test_message_names_layer_counts_ratio_and_fix(caplog) -> None:
    _check(caplog, array(), 2.0, name="via_cut")
    (record,) = _warnings(caplog)
    text = record.getMessage()
    assert "Via layer 'via_cut'" in text
    assert "merge_via_distance=2 um" in text
    assert "20 via polygons into 1 volume(s)" in text
    assert "factor of 8.25" in text
    assert "drawn 3.2 um^2" in text
    assert "smallest gap between vias, 1.1 um" in text
    assert "merge_via_distance=0 " in text
    assert "keep the cuts as drawn" in text


@pytest.mark.parametrize("layout", [row, diagonal, array])
def test_the_suggested_distance_keeps_every_cut(caplog, layout) -> None:
    polys = layout()
    _check(caplog, polys, 2.0)
    (record,) = _warnings(caplog)
    match = re.search(r"merge_via_distance=([0-9.]+) \(below", record.getMessage())
    assert match is not None
    suggested = float(match.group(1))
    assert 0 < suggested < 1.0
    assert len(_merge_via_polygons(polys, suggested)) == len(polys)


def test_touching_vias_suggest_turning_merging_off(caplog) -> None:
    # two touching 1 um squares and a third one 0.9 um further on
    polys = [_rect(0, 0, 1, 1), _rect(1, 0, 2, 1), _rect(2.9, 0, 4, 1)]
    _check(caplog, polys, 2.0)
    (record,) = _warnings(caplog)
    assert "merge_via_distance=0 (some vias touch" in record.getMessage()


def test_one_warning_per_layer(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for name in ("via1", "via2"):
            merged = _merge_via_polygons(row(), 2.0)
            _warn_if_via_merge_inflates_area(name, row(), merged, 2.0)
    texts = [r.getMessage() for r in _warnings(caplog)]
    assert len(texts) == 2
    assert "'via1'" in texts[0]
    assert "'via2'" in texts[1]


def test_a_single_via_never_warns(caplog) -> None:
    ratio, merged = _check(caplog, [_cut(0, 0)], 2.0)
    assert ratio is None
    assert len(merged) == 1
    assert _warnings(caplog) == []


def test_far_apart_vias_never_warn(caplog) -> None:
    ratio, merged = _check(caplog, [_cut(0, 0), _cut(10, 0), _cut(0, 10)], 2.0)
    assert ratio is None
    assert len(merged) == 3
    assert _warnings(caplog) == []


def test_merging_that_adds_little_area_does_not_warn(caplog) -> None:
    # two 1 um squares 0.01 um apart: they merge, the bridge adds 0.5 % area
    polys = [_rect(0, 0, 1, 1), _rect(1.01, 0, 2.01, 1)]
    ratio, merged = _check(caplog, polys, 2.0)
    assert len(merged) == 1
    assert ratio == pytest.approx(1.005, abs=1e-3)
    assert _warnings(caplog) == []


def test_a_hole_of_the_merged_polygon_is_not_counted(caplog) -> None:
    # four touching bars make a square frame: merging keeps its hole, so the
    # area is the frame's (20 um^2), not that of the outline (36 um^2)
    frame = [
        _rect(0, 0, 6, 1),
        _rect(0, 5, 6, 6),
        _rect(0, 1, 1, 5),
        _rect(5, 1, 6, 5),
    ]
    ratio, merged = _check(caplog, frame, 2.0)
    assert len(merged) == 1
    assert len(merged[0][2]) == 1  # the hole survived
    assert ratio == pytest.approx(1.0, abs=1e-6)
    assert _warnings(caplog) == []


def test_the_holes_of_drawn_vias_are_subtracted_from_the_drawn_area(caplog) -> None:
    # a ring via (4 x 4 um with a 2 x 2 um hole: 12 um^2) and a 1 um square
    # 1 um from it; merging joins them and fills the hole
    ring = _rect(0, 0, 4, 4, holes=[([1, 3, 3, 1], [1, 1, 3, 3])])
    square = _rect(5, 0, 6, 1)
    ratio, merged = _check(caplog, [ring, square], 2.0)
    assert len(merged) == 1
    (record,) = _warnings(caplog)
    assert "drawn 13 um^2" in record.getMessage()
    assert ratio > VIA_MERGE_AREA_WARN_RATIO


def test_overlapping_drawn_vias_are_counted_once(caplog) -> None:
    polys = [_rect(0, 0, 2, 2), _rect(1, 0, 3, 2)]  # overlap 1 x 2, union 6 um^2
    ratio, merged = _check(caplog, polys, 2.0)
    assert len(merged) == 1
    assert ratio == pytest.approx(1.0, abs=1e-6)
    assert _warnings(caplog) == []


def test_merge_via_polygons_is_unchanged(caplog) -> None:
    """The signature and the return value stay as they were."""
    polys = array()
    merged = _merge_via_polygons(polys, 2.0)
    assert isinstance(merged, list)
    assert len(merged) == 1
    xs, ys, holes = merged[0]
    assert len(xs) == len(ys)
    assert holes == []
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _merge_via_polygons(polys, 2.0)
    assert _warnings(caplog) == []  # the warning comes from add_metals


# ----- end to end ----------------------------------------------------------------

MESH = {"refined_mesh_size": 0.5, "max_mesh_size": 3.0}
# two vias 1 um apart under a bar of m2 on a bar of m1; a second electrode
BOXES = [
    ("m1", 0, 0, 4, 4),
    ("v1", 0.5, 1, 1, 2),
    ("v1", 2, 1, 2.5, 2),
    ("m2", 0, 0, 4, 4),
    ("m1", 10, 0, 14, 4),
]
PORTS = [("PLUS", 2, 2, "m1")]


def _mesh(tmp_path, distance: float) -> None:
    sim = make_sim(tmp_path / "es", BOXES, PORTS)
    sim.add_terminal("A", net="PLUS")
    sim.add_terminal("B", region=(10, 0, 14, 4), layer="m1")
    sim.mesh(**MESH, merge_via_distance=distance)


def test_meshing_warns_once_for_the_merged_via_layer(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _mesh(tmp_path, 2.0)
    texts = [r.getMessage() for r in _warnings(caplog) if "Via layer" in r.getMessage()]
    assert len(texts) == 1
    assert "Via layer 'v1'" in texts[0]
    assert "2 via polygons into 1 volume(s)" in texts[0]
    assert "factor of 2.00" in texts[0]


def test_meshing_without_merging_does_not_warn(tmp_path, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _mesh(tmp_path, 0)
    assert [r for r in _warnings(caplog) if "Via layer" in r.getMessage()] == []


def test_degenerate_polygons_are_skipped() -> None:
    line = ([0.0, 1.0], [0.0, 0.0], [])
    assert _via_shapes([line, _cut(0.0, 0.0)])[0].area == pytest.approx(SIDE**2)
    assert len(_via_shapes([line])) == 0


def test_self_intersecting_polygons_are_repaired() -> None:
    bowtie = ([0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [])
    (shape,) = _via_shapes([bowtie])
    assert shape.is_valid
    assert shape.area == pytest.approx(0.5)


def test_a_single_via_has_no_gap() -> None:
    assert _smallest_via_gap(_via_shapes([_cut(0.0, 0.0)])) == 0.0


def test_vias_with_no_drawn_area_do_not_warn(caplog) -> None:
    flat = ([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        ratio = _warn_if_via_merge_inflates_area("via1", [flat, flat], [flat], 2.0)
    assert ratio is None
    assert _warnings(caplog) == []
