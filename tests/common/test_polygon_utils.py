"""Check polygon conversion, simplification, and layer inspection."""

from __future__ import annotations

import logging

import gdsfactory as gf
import klayout.db as kdb
import matplotlib.pyplot as plt
import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box

from gsim.common import polygon_utils


@pytest.mark.parametrize("with_hole", [False, True])
def test_polygon_round_trip_preserves_units_and_holes(with_hole):
    holes = [[(0, 0), (1, 0), (1, 1), (0, 1)]] if with_hole else []
    original = Polygon([(-2, -1), (3, -1), (3, 2), (-2, 2)], holes=holes)

    converted = polygon_utils.shapely_to_klayout(original)

    assert isinstance(converted, kdb.Polygon)
    assert converted.holes() == len(holes)
    assert gf.kcl.to_um(converted.bbox().width()) == pytest.approx(5)
    restored = polygon_utils.klayout_to_shapely(converted)
    assert restored.equals(original)
    assert restored.area == pytest.approx(14 if with_hole else 15)


@pytest.mark.parametrize(
    "shape",
    [
        Polygon(),
        Polygon([(0, 0), (2, 2), (0, 2), (2, 0)]),
        LineString([(0, 0), (1, 1)]),
        MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]),
    ],
    ids=["empty", "self-intersection", "line", "multipolygon"],
)
def test_conversion_rejects_unusable_shapes(shape):
    assert polygon_utils.shapely_to_klayout(shape) is None


def test_conversion_reports_unexpected_failures(monkeypatch, caplog):
    def fail_conversion(*_args):
        raise RuntimeError("conversion unavailable")

    monkeypatch.setattr(polygon_utils.kdb, "Point", fail_conversion)

    assert polygon_utils.shapely_to_klayout(box(0, 0, 1, 1)) is None
    assert "conversion unavailable" in caplog.text


def test_simplification_reduces_vertices_and_preserves_hole():
    original = (
        Point(0, 0)
        .buffer(10, quad_segs=64)
        .difference(Point(0, 0).buffer(3, quad_segs=32))
    )
    polygon = polygon_utils.shapely_to_klayout(original)
    assert polygon is not None

    simplified = polygon_utils.simplify_polygon(polygon, tolerance=0.05)
    restored = polygon_utils.klayout_to_shapely(simplified)

    assert simplified.num_points_hull() < polygon.num_points_hull()
    assert simplified.holes() == 1
    assert restored.is_valid
    assert restored.area == pytest.approx(original.area, rel=0.02)
    assert not restored.covers(Point(0, 0))


@pytest.mark.parametrize("failure", ["conversion", "invalid-result"])
def test_simplification_returns_independent_copy_on_failure(monkeypatch, failure):
    original = polygon_utils.shapely_to_klayout(box(0, 0, 2, 1))
    assert original is not None
    if failure == "conversion":

        def fail_conversion(_polygon):
            raise ValueError("cannot convert polygon")

        monkeypatch.setattr(polygon_utils, "klayout_to_shapely", fail_conversion)
    else:
        monkeypatch.setattr(polygon_utils, "shapely_to_klayout", lambda _shape: None)

    restored = polygon_utils.simplify_polygon(original)

    assert restored is not original
    assert restored == original
    restored.move(1000, 0)
    assert original.bbox().left == 0


@pytest.mark.parametrize(
    ("point_count", "scale"), [(4, 0.1), (32, 0.5), (128, 1), (1024, 1.5)]
)
def test_adaptive_tolerance_scales_with_polygon_complexity(point_count, scale):
    angles = np.linspace(0, 2 * np.pi, point_count, endpoint=False)
    polygon = polygon_utils.shapely_to_klayout(
        Polygon(np.column_stack((100 * np.cos(angles), 100 * np.sin(angles))))
    )
    assert polygon is not None
    assert polygon.num_points_hull() == point_count
    assert polygon_utils._adaptive_tolerance(polygon, relative=0.01) == pytest.approx(
        200 * 0.01 * scale
    )


@pytest.mark.parametrize(("relative", "expected"), [(0, 0.001), (100, 1)])
def test_adaptive_tolerance_limits(relative, expected):
    polygon = polygon_utils.shapely_to_klayout(box(0, 0, 10, 10))
    assert polygon_utils._adaptive_tolerance(polygon, relative) == pytest.approx(
        expected
    )


def test_decimation_keeps_simple_shapes_and_reduces_complex_shapes(caplog):
    rectangle = polygon_utils.shapely_to_klayout(box(0, 0, 2, 1))
    circle = polygon_utils.shapely_to_klayout(Point(0, 0).buffer(10, quad_segs=128))
    assert rectangle is not None
    assert circle is not None
    original_point_count = circle.num_points_hull()
    with caplog.at_level(logging.DEBUG, logger=polygon_utils.__name__):
        reduced = polygon_utils.decimate([rectangle, circle], verbose=True)

    assert reduced[0] is rectangle
    assert reduced[1].num_points_hull() < circle.num_points_hull()
    assert circle.num_points_hull() == original_point_count
    assert polygon_utils.klayout_to_shapely(reduced[1]).area == pytest.approx(
        np.pi * 100, rel=0.01
    )
    assert "Polygon 1:" in caplog.text
    assert "Decimation:" in caplog.text


def test_decimation_handles_empty_input(caplog):
    with caplog.at_level(logging.DEBUG, logger=polygon_utils.__name__):
        assert polygon_utils.decimate([], verbose=True) == []
    assert "0 -> 0 pts (0.0% removed)" in caplog.text


def test_inspect_layers_rejects_empty_component(tmp_path):
    with pytest.raises(ValueError, match="Component has no polygons"):
        polygon_utils.inspect_layers(gf.Component(), tmp_path / "empty.png")
    assert not (tmp_path / "empty.png").exists()


@pytest.mark.parametrize("layer_count", [1, 2])
def test_inspect_layers_writes_nonempty_image_and_closes_figure(tmp_path, layer_count):
    component = gf.Component()
    for layer_number in range(1, layer_count + 1):
        component.add_polygon([(0, 0), (2, 0), (2, 1), (0, 1)], layer=(layer_number, 0))
    output = tmp_path / "layers.png"
    figures_before = plt.get_fignums()

    polygon_utils.inspect_layers(component, output, dpi=30)

    image = plt.imread(output)
    assert image.shape[0] > 100
    assert image.shape[1] > 100
    assert np.std(image[..., :3]) > 0
    assert plt.get_fignums() == figures_before
