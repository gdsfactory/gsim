"""Physical geometry and sizing must not depend on layout database units."""

from uuid import uuid4

import gdsfactory as gf
import klayout.db as kdb
import pytest

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.auto_size import (
    auto_refined_mesh_size,
    min_conductor_feature_size,
)
from gsim.palace.mesh.geometry import extract_geometry, extract_pec_polygons
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


def test_extracted_bounds_and_holes_are_in_microns(layout_with_units):
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
