"""Check supported legacy layer and polygon representations."""

from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString, Point

from gsim.common.cross_section import _layer_shapely_polys, _line_intervals
from gsim.common.stack._layer_utils import get_gds_layer_tuple


@pytest.mark.parametrize(
    ("layer", "expected"),
    [
        ((3, 2), (3, 2)),
        (3, (3, 0)),
        ("3/2", (3, 2)),
        ("named", None),
        (SimpleNamespace(layer=3, datatype=2), (3, 2)),
        (SimpleNamespace(layer=SimpleNamespace(layer=3, datatype=2)), (3, 2)),
        (SimpleNamespace(layer=SimpleNamespace(layer=3)), (3, 0)),
        (SimpleNamespace(value=(3, 2)), (3, 2)),
        (SimpleNamespace(value=3), (3, 0)),
        (3.0, (3, 0)),
        (object(), None),
    ],
)
def test_layer_protocols_resolve_to_gds_tuples(layer, expected):
    assert get_gds_layer_tuple(SimpleNamespace(layer=layer)) == expected


@pytest.mark.parametrize("representation", ["array", "tuple", "invalid", "empty"])
def test_polygon_protocols_repair_self_intersection(representation):
    points = [(0, 0), (2, 2), (0, 2), (2, 0)]
    polygon = np.array(points) if representation == "array" else tuple(points)
    raw = {} if representation == "empty" else {(1, 0): [object(), [1], polygon]}
    if representation == "invalid":
        raw = []
    component = SimpleNamespace(get_polygons=lambda **_kw: raw)
    result = _layer_shapely_polys(component, (1, 0), 0.001)
    assert len(result) == (1 if representation in {"array", "tuple"} else 0)
    if result:
        assert result[0].is_valid
        assert result[0].area == pytest.approx(1)


def test_geometry_collection_intervals_merge_touching_segments():
    collection = GeometryCollection(
        [
            LineString([(0, 0), (2, 0)]),
            Point(4, 0),
            LineString([(1, 0), (3, 0)]),
            LineString([(5, 0), (6, 0)]),
        ]
    )
    assert _line_intervals(collection) == [(0, 3), (5, 6)]
    assert _line_intervals(GeometryCollection()) == []
