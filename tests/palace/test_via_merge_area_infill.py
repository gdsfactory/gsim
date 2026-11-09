"""Area warning when via merging fills the recesses *inside* the vias.

Merging at a positive distance also closes the slot of a U-shaped via and fills
the hole of a ring, so the area grows although the number of via polygons stays
the same. The area check runs whenever positive-distance merging runs, and in
that case the only safe suggestion is ``merge_via_distance=0``: a smaller gap
between vias would not stop the infill.

The first tests need no mesh; the last two mesh a small layout of the synthetic
stack of ``_electrode_layouts``.
"""

from __future__ import annotations

import logging

import pytest

from gsim.palace import ElectrostaticSim
from gsim.palace.mesh import geometry
from gsim.palace.mesh.geometry import (
    VIA_MERGE_AREA_WARN_RATIO,
    _merge_via_polygons,
    _warn_if_via_merge_inflates_area,
)
from tests._electrode_layouts import LAYERS, component, stack

LOGGER = geometry.logger.name


def u_via(x0: float, y0: float = 0.0, slot: float = 2.0):
    """A U-shaped via polygon: 4 x 4 um outline with a slot 3 um deep.

    With the default 2 um slot the U covers 10 um^2 (outline 16, slot 6).
    """
    left = 2.0 - slot / 2
    right = 2.0 + slot / 2
    xs = [0, 4, 4, right, right, left, left, 0]
    ys = [0, 0, 4, 4, 1, 1, 4, 4]
    return ([x0 + x for x in xs], [y0 + y for y in ys], [])


def rect(x0: float, y0: float, x1: float, y1: float, holes=()):
    """A rectangular via polygon, optionally with holes ``(hx, hy)``."""
    return ([x0, x1, x1, x0], [y0, y0, y1, y1], list(holes))


def _warnings(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno == logging.WARNING]


def _check(caplog, polys, distance: float, name: str = "via1"):
    """Merge as ``add_metals`` does and run the area check."""
    merged = _merge_via_polygons(polys, distance)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        ratio = _warn_if_via_merge_inflates_area(name, polys, merged, distance)
    return ratio, merged


def test_two_separated_u_vias_warn_although_the_count_is_unchanged(caplog) -> None:
    """The review's case: drawn 20 um^2, merged 32 um^2, two polygons both ways."""
    polys = [u_via(0.0), u_via(10.0)]
    ratio, merged = _check(caplog, polys, 2.0)
    assert len(merged) == len(polys) == 2  # the old check stopped here
    assert ratio == pytest.approx(1.6, abs=1e-3)
    assert ratio > VIA_MERGE_AREA_WARN_RATIO
    (record,) = _warnings(caplog)
    text = record.getMessage()
    assert "Via layer 'via1'" in text
    assert "merge_via_distance=2 um" in text
    assert "factor of 1.60" in text
    assert "drawn 20 um^2" in text
    assert "merged 32 um^2" in text
    assert "recesses and holes within the vias" in text
    assert "Pass merge_via_distance=0 to keep the cuts as drawn" in text


def test_infill_within_polygons_gets_no_gap_suggestion(caplog) -> None:
    """The vias are 6 um apart: no gap between vias would prevent the infill."""
    _check(caplog, [u_via(0.0), u_via(10.0)], 2.0)
    (record,) = _warnings(caplog)
    text = record.getMessage()
    assert "smallest gap" not in text
    assert "merge_via_distance=0 " in text
    assert text.count("merge_via_distance=") == 2  # the setting used and the fix


@pytest.mark.parametrize("distance", [0.0, 1.0, 0.5])
def test_u_vias_that_stay_as_drawn_do_not_warn(caplog, distance) -> None:
    """A 2 um slot survives a merge distance of 1 um or less: nothing changes."""
    polys = [u_via(0.0), u_via(10.0)]
    ratio, merged = _check(caplog, polys, distance)
    assert len(merged) == 2
    assert ratio is None
    assert _warnings(caplog) == []


def test_a_single_u_via_never_warns(caplog) -> None:
    ratio, merged = _check(caplog, [u_via(0.0)], 2.0)
    assert ratio is None
    assert len(merged) == 1
    assert _warnings(caplog) == []


def test_far_apart_rectangles_do_not_warn(caplog) -> None:
    """Merging runs (two vias, distance > 0) but changes nothing."""
    polys = [rect(0, 0, 1, 1), rect(10, 0, 11, 1), rect(0, 10, 1, 11)]
    ratio, merged = _check(caplog, polys, 2.0)
    assert len(merged) == 3
    assert ratio is None
    assert _warnings(caplog) == []


def test_separated_rings_warn_when_merging_fills_their_holes(caplog) -> None:
    """Same polygon count, the holes of the drawn rings are filled."""
    hole1 = ([1, 3, 3, 1], [1, 1, 3, 3])
    hole2 = ([11, 13, 13, 11], [1, 1, 3, 3])
    polys = [rect(0, 0, 4, 4, holes=[hole1]), rect(10, 0, 14, 4, holes=[hole2])]
    ratio, merged = _check(caplog, polys, 2.0)
    assert len(merged) == 2
    assert ratio == pytest.approx(16 / 12, abs=1e-3)
    (record,) = _warnings(caplog)
    assert "kept 2 via polygons" in record.getMessage()
    assert "smallest gap" not in record.getMessage()


def test_the_gap_suggestion_stays_when_vias_were_joined(caplog) -> None:
    """Joined vias without recesses: the suggestion is unchanged by the fix."""
    polys = [rect(i * 1.0, 0.0, i * 1.0 + 0.4, 0.4) for i in range(10)]
    ratio, merged = _check(caplog, polys, 2.0)
    assert len(merged) == 1
    assert ratio > VIA_MERGE_AREA_WARN_RATIO
    (record,) = _warnings(caplog)
    text = record.getMessage()
    assert "merged 10 via polygons into 1 volume(s)" in text
    assert "smallest gap between vias, 0.6 um" in text
    assert "merge_via_distance=0.2 (below" in text


def test_joined_vias_with_a_narrow_slot_suggest_only_turning_merging_off(
    caplog,
) -> None:
    """A gap-based distance would still close the 0.3 um slot, so it is not offered."""
    # a thin U (1.5 x 4 um, 0.3 um slot, 4.95 um^2) and a 0.2 um square 1 um away
    thin_u = (
        [0, 1.5, 1.5, 0.9, 0.9, 0.6, 0.6, 0],
        [0, 0, 4, 4, 0.5, 0.5, 4, 4],
        [],
    )
    polys = [thin_u, rect(2.5, 0.0, 2.7, 0.2)]  # gap 1 um
    ratio, merged = _check(caplog, polys, 2.0)
    assert len(merged) == 1  # the square joined the U
    assert ratio > VIA_MERGE_AREA_WARN_RATIO
    (record,) = _warnings(caplog)
    text = record.getMessage()
    assert "merged 2 via polygons into 1 volume(s)" in text
    assert "smallest gap" not in text
    assert "a smaller distance would still fill in recesses" in text
    # the distance the unchecked message proposed (a third of the gap) keeps
    # both polygons but closes the slot, which is a large area growth
    assert len(_merge_via_polygons(polys, 1.0 / 3)) == 2
    caplog.clear()
    ratio_again, _ = _check(caplog, polys, 1.0 / 3)
    assert ratio_again is not None
    assert ratio_again > VIA_MERGE_AREA_WARN_RATIO


# ----- end to end ----------------------------------------------------------------

MESH = {"refined_mesh_size": 0.5, "max_mesh_size": 3.0}


def _u_sim(tmp_path) -> ElectrostaticSim:
    """Two U-shaped v1 vias 6 um apart between a bar of m1 and a bar of m2."""
    boxes = [
        ("m1", -1, -1, 15, 5),
        ("m2", -1, -1, 15, 5),
        ("m1", 25, 0, 29, 4),
    ]
    c = component(boxes, [("PLUS", 2, 2, "m1")])
    for x0 in (0.0, 10.0):
        xs, ys, _ = u_via(x0)
        c.add_polygon(list(zip(xs, ys, strict=True)), layer=(LAYERS["v1"][0], 0))
    sim = ElectrostaticSim()
    sim.set_output_dir(tmp_path / "es")
    sim.set_geometry(c)
    sim.set_stack(stack())
    sim.set_airbox(margin_x=5, margin_y=5, z_above=5, z_below=0)
    sim.add_terminal("A", net="PLUS")
    sim.add_terminal("B", region=(25, 0, 29, 4), layer="m1")
    return sim


def test_meshing_u_vias_warns_once(tmp_path, caplog) -> None:
    sim = _u_sim(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        sim.mesh(**MESH, merge_via_distance=2.0)
    texts = [r.getMessage() for r in _warnings(caplog) if "Via layer" in r.getMessage()]
    assert len(texts) == 1
    assert "Via layer 'v1'" in texts[0]
    assert "kept 2 via polygons" in texts[0]
    assert "factor of 1.60" in texts[0]


def test_meshing_u_vias_without_merging_does_not_warn(tmp_path, caplog) -> None:
    sim = _u_sim(tmp_path)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        sim.mesh(**MESH, merge_via_distance=0)
    assert [r for r in _warnings(caplog) if "Via layer" in r.getMessage()] == []
