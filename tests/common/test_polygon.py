"""Preserve physical footprints when KLayout contours touch at corners."""

from __future__ import annotations

from types import SimpleNamespace

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer
from shapely.geometry import MultiPolygon, Polygon, box

from gsim.common.pdk import resolve_passive_pcell
from gsim.common.polygon import fuse_polygons, layer_level_region


@pytest.fixture
def corner_contact_component() -> gf.Component:
    component = gf.Component()
    component.add_polygon([(0, 0), (15, 0), (15, 10), (0, 10)], layer=(1, 0))
    # The tip touches the upper boundary, joining otherwise separate interiors.
    component.add_polygon([(0, 2), (8, 10), (10, 6), (10, 2)], layer=(2, 0))
    component.add_polygon([(10, 0), (14, 0), (14, 10), (10, 10)], layer=(2, 0))
    # An isolated island with a hole also forces GEOS to union multiple contours.
    component.add_polygon(
        [(14.25, 4), (14.75, 4), (14.75, 6), (14.25, 6)], layer=(2, 0)
    )
    component.add_port(
        name="o1", center=(15, 5), width=0.5, orientation=0, layer=(1, 0)
    )
    return component


@pytest.fixture(params=["derived", "background"])
def core_level(request) -> LayerLevel:
    level = LayerLevel(
        layer=LogicalLayer(layer=(1, 0)) - LogicalLayer(layer=(2, 0)),
        derived_layer=LogicalLayer(layer=(10, 0)),
        material="Si",
        thickness=0.22,
        zmin=0,
    )
    if request.param == "background":
        level.layer = LogicalLayer(layer=(3, 0))
        # Keep coverage on gdsfactory versions preceding these fields as well.
        object.__setattr__(level, "background", True)
        object.__setattr__(level, "background_exclude_layers", [(2, 0)])
    return level


def assert_corner_contact_footprint(geometry) -> None:
    expected = MultiPolygon(
        [
            box(0, 0, 10, 2),
            Polygon([(0, 2), (8, 10), (0, 10)]),
            Polygon([(8, 10), (10, 6), (10, 10)]),
            box(14, 0, 15, 10).difference(box(14.25, 4, 14.75, 6)),
        ]
    )
    assert geometry.is_valid
    assert geometry.equals(expected)
    assert geometry.area == pytest.approx(65)
    assert geometry.bounds == pytest.approx((0, 0, 15, 10))
    assert sum(len(part.interiors) for part in geometry.geoms) == 1


@pytest.mark.parametrize("simplify_tol", [0.0, 1e-4])
def test_fuse_corner_contacts_preserves_footprint(
    corner_contact_component, core_level, simplify_tol
):
    geometry = fuse_polygons(
        corner_contact_component, core_level, round_tol=6, simplify_tol=simplify_tol
    )

    assert_corner_contact_footprint(geometry)


def test_fuse_corner_contacts_preserves_source_region(
    corner_contact_component, core_level
):
    region = layer_level_region(corner_contact_component, core_level)
    original = region.dup()
    source_layer = SimpleNamespace(get_shapes=lambda _component: region)

    geometry = fuse_polygons(corner_contact_component, source_layer, simplify_tol=0)

    assert_corner_contact_footprint(geometry)
    assert region.min_coherence is False
    assert (region ^ original).is_empty()


def test_resolve_corner_contacts_preserves_physical_layer(
    corner_contact_component, core_level
):
    # Use the drawn port layer; a background footprint ignores its source shapes.
    if getattr(core_level, "background", False):
        core_level.layer = LogicalLayer(layer=(1, 0))
    pdk = gf.Pdk(
        name="corner_contact_test",
        layer_stack=LayerStack(layers={"core": core_level}),
    )

    resolved = resolve_passive_pcell(corner_contact_component, pdk=pdk)

    assert_corner_contact_footprint(resolved.layers["core"].geometry)
    assert resolved.layers["core"].material == "Si"
    assert resolved.layers["core"].z_bounds == (0, 0.22)
    assert resolved.ports["o1"].layer_key == "core"
    assert resolved.ports["o1"].center == (15, 5, 0.11)
