"""Tests for the pure symmetry-plane helpers in ``mesh/symmetry.py``."""

from __future__ import annotations

import klayout.db as kdb
import pytest

from gsim.palace.mesh.symmetry import (
    asymmetric_layers,
    check_plane_inside_domain,
    clamp_extent,
    clamp_xy_bounds,
    classify_interval,
    clip_polygon,
    mirror_region,
)
from gsim.palace.models.symmetry import SymmetryPlaneConfig


def _plane(axis="y", position=0.0, keep="positive", kind="pmc"):
    """Build a plane for the tests."""
    return SymmetryPlaneConfig(axis=axis, position=position, keep=keep, kind=kind)


def _rect(x0, y0, x1, y1):
    """Rectangle in um as a KLayout polygon (1 nm database unit)."""
    return kdb.Polygon(kdb.Box(*(round(v * 1000) for v in (x0, y0, x1, y1))))


@pytest.mark.parametrize(
    ("keep", "lo", "hi", "expected"),
    [
        ("positive", 1.0, 3.0, "kept"),
        ("positive", -3.0, -1.0, "removed"),
        ("positive", -1.0, 1.0, "straddles"),
        ("positive", 0.0, 2.0, "touches"),
        ("positive", -2.0, 0.0, "removed"),
        ("positive", 0.0, 0.0, "in_plane"),
        ("negative", -3.0, -1.0, "kept"),
        ("negative", 1.0, 3.0, "removed"),
        ("negative", -2.0, 0.0, "touches"),
        ("negative", 0.0, 2.0, "removed"),
        ("negative", -1.0, 1.0, "straddles"),
    ],
)
def test_classify_interval(keep, lo, hi, expected):
    """Intervals are classified relative to the plane and the kept side."""
    assert classify_interval(lo, hi, _plane(keep=keep)) == expected


def test_classify_interval_uses_plane_position():
    """The classification follows the plane position."""
    assert classify_interval(1.0, 3.0, _plane(position=2.0)) == "straddles"
    assert classify_interval(2.0, 3.0, _plane(position=2.0)) == "touches"


@pytest.mark.parametrize(
    ("keep", "expected"),
    [("positive", (2.0, 10.0)), ("negative", (-10.0, 2.0))],
)
def test_clamp_extent(keep, expected):
    """The removed-side bound moves to the plane."""
    assert clamp_extent(-10.0, 10.0, _plane(position=2.0, keep=keep)) == expected


@pytest.mark.parametrize(
    ("axis", "keep", "expected"),
    [
        ("y", "positive", (-4.0, 1.0, 6.0, 8.0)),
        ("y", "negative", (-4.0, -2.0, 6.0, 1.0)),
        ("x", "positive", (1.0, -2.0, 6.0, 8.0)),
        ("x", "negative", (-4.0, -2.0, 1.0, 8.0)),
    ],
)
def test_clamp_xy_bounds(axis, keep, expected):
    """Only the bound on the plane axis is cut."""
    plane = _plane(axis=axis, position=1.0, keep=keep)

    assert clamp_xy_bounds((-4.0, -2.0, 6.0, 8.0), plane) == expected


def test_plane_inside_domain_passes():
    """A plane strictly inside the domain is accepted."""
    check_plane_inside_domain(-5.0, 5.0, _plane())


@pytest.mark.parametrize("position", [-5.0, 5.0, 7.0, -9.0])
def test_plane_outside_or_on_wall_raises(position):
    """A plane on or beyond the domain wall is rejected."""
    with pytest.raises(ValueError, match="outside the simulation domain"):
        check_plane_inside_domain(-5.0, 5.0, _plane(position=position))


def test_clip_polygon_with_hole_across_plane():
    """A polygon with a hole is clipped without losing the hole."""
    outer = _rect(-10, -10, 10, 10)
    region = kdb.Region(outer) - kdb.Region(_rect(-2, -2, 2, 2))
    (polygon,) = list(region.each())

    (clipped,) = clip_polygon(polygon, _plane())

    assert clipped.bbox() == kdb.Box(-10000, 0, 10000, 10000)
    assert clipped.holes() == 0  # the hole is cut open at the plane
    assert clipped.area() == pytest.approx(20000 * 10000 - 4000 * 2000)


def test_clip_polygon_keeps_hole_on_kept_side():
    """A hole fully on the kept side stays a hole."""
    region = kdb.Region(_rect(-10, -10, 10, 10)) - kdb.Region(_rect(-2, 3, 2, 6))
    (polygon,) = list(region.each())

    (clipped,) = clip_polygon(polygon, _plane())

    assert clipped.holes() == 1


def test_clip_polygon_negative_side_and_removed():
    """``keep`` selects the side, and a polygon on the other side vanishes."""
    poly = _rect(0, 1, 5, 4)

    assert clip_polygon(poly, _plane(keep="negative")) == []
    (kept,) = clip_polygon(poly, _plane(keep="positive"))
    assert kept.bbox() == poly.bbox()


def test_clip_polygon_x_axis_and_offset():
    """Clipping works along x at a non-zero position."""
    (clipped,) = clip_polygon(_rect(0, 0, 10, 4), _plane(axis="x", position=3.0))

    assert clipped.bbox() == kdb.Box(3000, 0, 10000, 4000)


def test_mirror_region_y_and_x():
    """Mirroring maps a rectangle onto its image about the plane."""
    region = kdb.Region(_rect(0, 1, 4, 3))

    assert mirror_region(region, _plane(position=0.5)).bbox() == kdb.Box(
        0, -2000, 4000, 0
    )
    assert mirror_region(region, _plane(axis="x", position=1.0)).bbox() == kdb.Box(
        -2000, 1000, 2000, 3000
    )


def _gsgsg(offset=0.0):
    """Ground-signal-ground-signal-ground strips along x, symmetric about y=0."""
    strips = [(-35, -25), (-15, -5), (-2, 2), (5, 15), (25, 35)]
    return kdb.Region(
        [
            _rect(0, a + (offset if a > 0 else 0), 100, b + (offset if a > 0 else 0))
            for a, b in strips
        ]
    )


def _gssg():
    """Ground-signal-signal-ground strips symmetric about y=0."""
    strips = [(-30, -20), (-10, -3), (3, 10), (20, 30)]
    return kdb.Region([_rect(0, a, 100, b) for a, b in strips])


def test_mirror_check_passes_on_symmetric_layouts():
    """GSSG and GSGSG layouts are symmetric about y=0."""
    assert asymmetric_layers({"gssg": _gssg(), "gsgsg": _gsgsg()}, _plane()) == []


def test_mirror_check_flags_shifted_strip():
    """Shifting one half of the layout breaks the symmetry."""
    layers = {"ok": _gssg(), "bad": _gsgsg(offset=1.0)}

    assert asymmetric_layers(layers, _plane()) == ["bad"]


def test_mirror_check_skips_precut_layout():
    """A layout with content on the kept side only is not checked."""
    layers = {"half": kdb.Region([_rect(0, 3, 100, 10), _rect(0, 20, 100, 30)])}

    assert asymmetric_layers(layers, _plane()) == []


def test_mirror_check_tolerates_one_database_unit():
    """A one nm difference is below the check's tolerance."""
    region = kdb.Region(_rect(0, -10, 100, 10)) + kdb.Region(_rect(0, 10, 100, 10.001))

    assert asymmetric_layers({"m": region}, _plane()) == []
