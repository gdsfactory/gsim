"""``restore_generic_pdk`` heals the shared layout that tests leave behind.

Regression for test-order failures in ``tests/meep/test_physical_layers.py``:
after IHP was activated, a raw ``(2, 0)`` registered in a freed slot and the
generic PDK came back, the layout held two ``(2, 0)`` layers and
``LAYER.SLAB150`` wrote to a different one than a ``(2, 0)`` lookup read.
"""

from __future__ import annotations

import gdsfactory as gf
import kfactory as kf
import pytest
from kfactory import kdb

from tests._helpers import restore_generic_pdk

BOX = kdb.Box(0, 0, 1000, 1000)  # 1 um x 1 um in dbu


def _tuples() -> list[tuple[int, int]]:
    layout = kf.kcl.layout
    infos = (layout.get_info(index) for index in layout.layer_indexes())
    return [(info.layer, info.datatype) for info in infos]


def _assert_clean() -> None:
    layout = kf.kcl.layout
    tuples = _tuples()
    for layer in gf.gpdk.LAYER:  # ty: ignore[not-iterable]
        info = layout.get_info(int(layer))
        assert (info.layer, info.datatype, info.name) == (
            layer.layer,
            layer.datatype,
            layer.name,
        )
        assert tuples.count((layer.layer, layer.datatype)) == 1


def _area(component: gf.Component, layer: tuple[int, int]) -> int:
    """Area (dbu^2) the component holds on whatever layer ``layer`` resolves to."""
    index = component.kcl.layer(*layer)
    return kdb.Region(component.kdb_cell.begin_shapes_rec(index)).area()


@pytest.fixture
def locked_cell() -> gf.Component:
    """A cached gdsfactory cell. kfactory locks it, which blocks gdsfactory's own
    ``Layout.move_layer`` / ``delete_layer`` when it restores the layer table."""
    gf.gpdk.PDK.activate()
    cell = gf.components.straight(length=7.0)
    assert cell.kdb_cell.is_locked()
    return cell


@pytest.mark.usefixtures("locked_cell")
def test_a_duplicated_slab_layer_is_merged_into_the_enum_slot() -> None:
    restore_generic_pdk()
    layout = kf.kcl.layout
    slot = int(gf.gpdk.LAYER.SLAB150)
    # the state gdsfactory leaves: the enum slot unnamed, a named twin elsewhere
    layout.set_info(slot, kdb.LayerInfo(2, 0))
    twin = layout.insert_layer(kdb.LayerInfo(2, 0, "SLAB150"))
    component = gf.Component()
    component.shapes(slot).insert(BOX)
    component.shapes(twin).insert(BOX.moved(5000, 0))
    assert _tuples().count((2, 0)) == 2

    restore_generic_pdk()

    _assert_clean()
    assert layout.find_layer(2, 0) == slot
    assert _area(component, (2, 0)) == 2_000_000  # both boxes, on the one layer


@pytest.mark.usefixtures("locked_cell")
def test_a_raw_layer_in_a_generic_slot_is_moved_out_with_its_shapes() -> None:
    restore_generic_pdk()
    layout = kf.kcl.layout
    slot = int(gf.gpdk.LAYER.SHALLOW_ETCH)
    layout.set_info(slot, kdb.LayerInfo(250, 0))
    component = gf.Component()
    component.shapes(slot).insert(BOX)

    restore_generic_pdk()

    _assert_clean()
    assert layout.find_layer(250, 0) not in {None, slot}
    assert _area(component, (250, 0)) == 1_000_000
    assert _area(component, (2, 6)) == 0
    drawn = gf.Component()
    drawn.add_polygon([(0, 0), (2, 0), (2, 1), (0, 1)], layer=gf.gpdk.LAYER.SLAB150)
    drawn.add_polygon(
        [(4, 0), (6, 0), (6, 1), (4, 1)], layer=gf.gpdk.LAYER.SHALLOW_ETCH
    )
    assert _area(drawn, (2, 0)) == 2_000_000
    assert _area(drawn, (2, 6)) == 2_000_000


@pytest.mark.usefixtures("locked_cell")
def test_a_raw_layer_taken_under_ihp_does_not_split_the_slab_layer() -> None:
    ihp = pytest.importorskip("ihp")
    restore_generic_pdk()
    ihp.PDK.activate()
    gf.Pdk(name="restore_generic_pdk_probe").activate()  # no layer enum: no pruning
    kf.kcl.layer(2, 0)  # lowest free slot while the generic slots are pruned
    restore_generic_pdk()

    _assert_clean()
    assert gf.get_active_pdk().name == gf.gpdk.PDK.name
    component = gf.Component()
    component.add_polygon([(0, 0), (2, 0), (2, 1), (0, 1)], layer=gf.gpdk.LAYER.SLAB150)
    assert _area(component, (2, 0)) == 2_000_000
