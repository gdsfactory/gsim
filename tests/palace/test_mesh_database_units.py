"""Physical geometry and sizing must not depend on layout database units."""

import math
from uuid import uuid4

import gdsfactory as gf
import klayout.db as kdb
import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.auto_size import (
    auto_refined_mesh_size,
    min_conductor_feature_size,
)
from gsim.palace.mesh.geometry import (
    _parse_klayout_polygon,
    extract_geometry,
    extract_pec_polygons,
)
from gsim.palace.models.symmetry import SymmetryPlaneConfig


@pytest.fixture(params=[0.001, 0.01, 0.0005])
def layout_with_units(request):
    gf.gpdk.PDK.activate()
    cell_layout = gf.kf.KCLayout(f"mesh_units_{uuid4().hex}")
    cell_layout.layout.dbu = request.param
    component = gf.Component(kcl=cell_layout)
    metal_index = cell_layout.layout.layer(201, 0)

    def box(xmin, ymin, xmax, ymax):
        return kdb.DBox(xmin, ymin, xmax, ymax).to_itype(request.param)

    component.kdb_cell.shapes(metal_index).insert(box(0, -1, 100, 1))
    ring = kdb.Polygon(box(0, 20, 100, 120))
    ring.insert_hole(box(40, 50, 60, 80))
    component.kdb_cell.shapes(metal_index).insert(ring)
    stack = LayerStack(
        layers={
            "metal": Layer(
                name="metal",
                gds_layer=(201, 0),
                zmin=0.0,
                zmax=0.5,
                thickness=0.5,
                material="copper",
                layer_type="conductor",
            )
        }
    )
    return component, stack


def test_custom_layout_extraction_preserves_physical_bounds_and_holes(
    layout_with_units,
):
    component, stack = layout_with_units
    geometry = extract_geometry(component, stack)
    assert geometry.bbox == pytest.approx((0, -1, 100, 120))
    assert len(geometry.polygons) == 2
    holes = [hole for _, _, _, holes in geometry.polygons for hole in holes]
    assert len(holes) == 1
    xs, ys = holes[0]
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx((40, 50, 60, 80))


def test_pec_extraction_matches_geometry_extraction(layout_with_units):
    component, stack = layout_with_units
    polygons = extract_pec_polygons(component, (201, 0))
    assert polygons == [
        (xs, ys, holes)
        for _, xs, ys, holes in extract_geometry(component, stack).polygons
    ]


def test_sizing_uses_physical_trace_width(layout_with_units):
    component, stack = layout_with_units
    assert min_conductor_feature_size(component, stack) == pytest.approx(2.0)
    assert auto_refined_mesh_size(component, stack, preset_size=10.0) == pytest.approx(
        1.0
    )


def test_offset_symmetry_plane_uses_layout_units(layout_with_units):
    component, stack = layout_with_units
    plane = SymmetryPlaneConfig(axis="x", position=50.0)
    geometry = extract_geometry(component, stack, symmetry_plane=plane)
    assert geometry.bbox == pytest.approx((0, -1, 100, 120))
    for _, xs, _, _ in geometry.polygons:
        assert min(xs) == pytest.approx(50.0)
        assert max(xs) == pytest.approx(100.0)
    assert extract_pec_polygons(component, (201, 0), symmetry_plane=plane) == [
        (xs, ys, holes) for _, xs, ys, holes in geometry.polygons
    ]


@pytest.mark.parametrize("dbu", [0.001, 0.01, 0.0005])
def test_polygon_unit_conversion_without_layout_lookup(dbu):
    polygon = kdb.DPolygon(kdb.DBox(0, 0, 100, 100))
    polygon.insert_hole(kdb.DBox(40, 50, 60, 80))
    parsed = _parse_klayout_polygon(polygon.to_itype(dbu), dbu)
    assert parsed is not None
    xs, ys, holes = parsed
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx((0, 0, 100, 100))
    hole_x, hole_y = holes[0]
    assert (min(hole_x), min(hole_y), max(hole_x), max(hole_y)) == pytest.approx(
        (40, 50, 60, 80)
    )


@pytest.mark.parametrize("verify_symmetry", [False, True])
def test_symmetry_plane_rejects_silent_layout_grid_snapping(
    layout_with_units, verify_symmetry
):
    component, stack = layout_with_units
    dbu = component.kcl.layout.dbu
    plane = SymmetryPlaneConfig(
        axis="x", position=50 + dbu / 2, verify_symmetry=verify_symmetry
    )
    with pytest.raises(ValueError, match=r"layout.*grid"):
        extract_geometry(component, stack, symmetry_plane=plane)
    with pytest.raises(ValueError, match=r"layout.*grid"):
        extract_pec_polygons(component, (201, 0), symmetry_plane=plane)


def test_subnanometer_symmetry_plane_uses_fine_layout_grid(layout_with_units):
    component, stack = layout_with_units
    dbu = component.kcl.layout.dbu
    plane = SymmetryPlaneConfig(axis="x", position=50 + dbu, verify_symmetry=False)
    geometry = extract_geometry(component, stack, symmetry_plane=plane)
    assert min(min(xs) for _, xs, _, _ in geometry.polygons) == pytest.approx(
        plane.position
    )


def test_decimation_does_not_depend_on_global_layout_units(
    layout_with_units, monkeypatch
):
    component, stack = layout_with_units
    circle = kdb.DPolygon(
        [
            kdb.DPoint(
                200 + math.cos(index * math.tau / 256), math.sin(index * math.tau / 256)
            )
            for index in range(256)
        ]
    )
    component.kdb_cell.shapes(component.kcl.layout.layer(201, 0)).insert(
        circle.to_itype(component.kcl.layout.dbu)
    )
    reference = extract_geometry(component, stack, decimate_tolerance=1e-6)
    monkeypatch.setattr(gf.kcl.layout, "dbu", 0.1)
    changed_global = extract_geometry(component, stack, decimate_tolerance=1e-6)
    assert changed_global.polygons == reference.polygons
    assert changed_global.bbox == reference.bbox
