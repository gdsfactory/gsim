"""Detect physical widths and spacing without refining overlap or corner artifacts."""

from uuid import uuid4

import gdsfactory as gf
import klayout.db as kdb
import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.auto_size import (
    auto_refined_mesh_size,
    min_conductor_feature_size,
    min_conductor_gap,
)

_SHAPES = {
    "trace": [(0, 0), (100, 0), (100, 2), (0, 2)],
    "neck": [
        (0, 0),
        (40, 0),
        (40, 49),
        (60, 49),
        (60, 0),
        (100, 0),
        (100, 100),
        (60, 100),
        (60, 51),
        (40, 51),
        (40, 100),
        (0, 100),
    ],
    "notch": [
        (0, 0),
        (100, 0),
        (100, 100),
        (51, 100),
        (51, 20),
        (49, 20),
        (49, 100),
        (0, 100),
    ],
}


def _layout(dbu, polygons, angle=0):
    gf.gpdk.PDK.activate()
    cell_layout = gf.kf.KCLayout(f"mesh_feature_{uuid4().hex}")
    cell_layout.layout.dbu = dbu
    component = gf.Component(kcl=cell_layout)
    metal_index = cell_layout.layout.layer(201, 0)
    transform = kdb.ICplxTrans(1, angle, False, 0, 0)
    for points in polygons:
        polygon = kdb.DPolygon([kdb.DPoint(*point) for point in points]).to_itype(dbu)
        component.kdb_cell.shapes(metal_index).insert(polygon.transformed(transform))
    stack = LayerStack(
        layers={
            "metal": Layer(
                name="metal",
                gds_layer=(201, 0),
                zmin=0,
                zmax=0.5,
                thickness=0.5,
                material="copper",
                layer_type="conductor",
            )
        }
    )
    return component, stack


@pytest.mark.parametrize("dbu", [0.001, 0.01, 0.0005])
@pytest.mark.parametrize("angle", [0, 37, 90])
@pytest.mark.parametrize("shape", ["trace", "neck", "notch"])
def test_narrow_feature_inside_connected_polygon(dbu, angle, shape):
    component, stack = _layout(dbu, [_SHAPES[shape]], angle)
    feature = min_conductor_feature_size(component, stack)
    assert feature == pytest.approx(2.0, abs=3 * dbu)
    assert auto_refined_mesh_size(component, stack, 10.0) == pytest.approx(
        1.0, abs=2 * dbu
    )
    if shape == "notch":
        assert min_conductor_gap(component, stack) == pytest.approx(2.0, abs=3 * dbu)


def test_hole_in_one_conductor_is_a_gap():
    component, stack = _layout(0.001, [[(0, 0), (100, 0), (100, 100), (0, 100)]])
    shapes = component.kdb_cell.shapes(component.kcl.layout.layer(201, 0))
    shapes.clear()
    polygon = kdb.Polygon(kdb.Box(0, 0, 100000, 100000))
    polygon.insert_hole(kdb.Box(49000, 10000, 51000, 90000))
    shapes.insert(polygon)
    assert min_conductor_gap(component, stack) == pytest.approx(2.0)
    assert min_conductor_feature_size(component, stack) == pytest.approx(2.0)


@pytest.mark.parametrize("split", [50, 51])
def test_input_tiles_do_not_introduce_small_features(split):
    component, stack = _layout(
        0.001,
        [
            [(0, 0), (split, 0), (split, 100), (0, 100)],
            [(49, 0), (100, 0), (100, 100), (49, 100)],
        ],
    )
    assert min_conductor_gap(component, stack) is None
    assert min_conductor_feature_size(component, stack) == pytest.approx(100.0)


def test_acute_corner_does_not_force_near_zero_size():
    component, stack = _layout(0.001, [[(0, 0), (100, 0), (50, 10)]])
    assert min_conductor_feature_size(component, stack) == pytest.approx(10.0)
    assert auto_refined_mesh_size(component, stack, 10.0) == pytest.approx(5.0)


def test_gap_search_limit_is_preserved():
    component, stack = _layout(
        0.001,
        [
            [(0, 0), (100, 0), (100, 20), (0, 20)],
            [(0, 35), (100, 35), (100, 55), (0, 55)],
        ],
    )
    assert min_conductor_gap(component, stack, max_gap_um=10) is None
    assert min_conductor_gap(component, stack, max_gap_um=16) == pytest.approx(15.0)
