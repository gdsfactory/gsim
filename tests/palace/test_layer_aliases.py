"""Preserve geometry when named internal layers share a physical GDS layer."""

import uuid

import gdsfactory as gf
import pytest
from kfactory import KCLayout, kdb

from gsim.common.stack import Layer, LayerStack
from gsim.palace.mesh.geometry import extract_geometry, extract_pec_polygons


@pytest.fixture
def aliased_component():
    """Keep duplicate layer names isolated from the session's PDK layout."""
    gf.gpdk.PDK.activate()
    layout = KCLayout(f"palace_aliases_{uuid.uuid4().hex}")
    component = gf.Component(kcl=layout)
    indices = [
        layout.layout.insert_layer(kdb.LayerInfo(18000, 0, name))
        for name in ("generic_name", "pdk_name")
    ]
    component.kdb_cell.shapes(indices[1]).insert(kdb.Box(0, 0, 1000, 1000))
    return component, indices


@pytest.mark.parametrize("populate_first_alias", [False, True])
def test_all_named_aliases_reach_geometry_and_pec_extraction(
    aliased_component, populate_first_alias
):
    component, indices = aliased_component
    if populate_first_alias:
        component.kdb_cell.shapes(indices[0]).insert(kdb.Box(2000, 0, 3000, 1000))
    stack = LayerStack.from_layer_list(
        [
            Layer(
                name="metal",
                gds_layer=(18000, 0),
                zmin=0,
                zmax=1,
                thickness=1,
                material="aluminum",
                layer_type="conductor",
            )
        ]
    )
    geometry = extract_geometry(component, stack)
    pec_polygons = extract_pec_polygons(component, (18000, 0))
    expected_bounds = {(0.0, 1.0)}
    if populate_first_alias:
        expected_bounds.add((2.0, 3.0))
    assert len(geometry.polygons) == len(pec_polygons) == len(expected_bounds)
    assert {layer for layer, *_ in geometry.polygons} == {18000}
    assert {(min(x), max(x)) for _, x, *_ in geometry.polygons} == expected_bounds
    assert {(min(x), max(x)) for x, *_ in pec_polygons} == expected_bounds
