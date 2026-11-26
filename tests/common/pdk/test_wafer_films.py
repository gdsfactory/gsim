"""Levels declared on the wafer layer are blanket films on any PDK."""

from __future__ import annotations

from types import SimpleNamespace

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerLevel, LayerStack, LogicalLayer

from gsim.common.materials import GSIM_MATERIAL_CARDS
from gsim.common.pdk import LayerResolutionError, resolve_passive_pcell
from gsim.common.pdk.wafer import wafer_layer_tuple, with_wafer_films

STACK_ORDER = ["substrate", "box", "core", "clad"]


@pytest.fixture(autouse=True)
def gpdk_active():
    """Use the generic PDK and restore the previous process state."""
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    gf.gpdk.PDK.activate(force=True)
    yield gf.gpdk.PDK
    if previous is not None:
        previous.activate(force=True)


def _level(layer, material="Si", **kwargs):
    return LayerLevel(layer=layer, thickness=1, zmin=0, material=material, **kwargs)


@pytest.mark.parametrize("name", ["coupler", "mmi1x2"])
def test_gpdk_component_resolves_the_whole_stack(gpdk_active, name) -> None:
    resolved = resolve_passive_pcell(name, pdk=gpdk_active)

    assert list(resolved.layers) == STACK_ORDER
    assert {key: layer.material for key, layer in resolved.layers.items()} == {
        "substrate": "Si",
        "box": "SiO2",
        "core": "Si",
        "clad": "SiO2",
    }
    assert resolved.materials["Si"].refractive_index == pytest.approx(3.478, abs=1e-3)
    assert resolved.materials["SiO2"].refractive_index == pytest.approx(1.444, abs=1e-3)
    # Films cover the footprint without any polygon on the wafer layer.
    core = resolved.layers["core"].geometry.bounds
    for key in ("substrate", "box", "clad"):
        assert resolved.layers[key].geometry.bounds == pytest.approx(core, abs=0.01)
    assert resolved.layers["substrate"].zmin == pytest.approx(-13)
    assert resolved.layers["clad"].zmax == pytest.approx(3)
    assert not resolved.component.get_polygons(by="tuple").get((999, 0))


def test_ports_stay_on_the_component_faces(gpdk_active) -> None:
    resolved = resolve_passive_pcell("coupler", pdk=gpdk_active)
    xmin, _ymin, _zmin = resolved.bounds[0]
    xmax, _ymax, _zmax = resolved.bounds[1]
    for port in resolved.ports.values():
        assert port.center[0] in (pytest.approx(xmin), pytest.approx(xmax))


def test_resolution_does_not_modify_the_pdk_stack(gpdk_active) -> None:
    original = gpdk_active.layer_stack.model_dump()
    resolved = resolve_passive_pcell("mmi1x2", pdk=gpdk_active)
    assert gpdk_active.layer_stack.model_dump() == original
    assert not gpdk_active.layer_stack.layers["box"].background
    assert resolved.layer_stack is not gpdk_active.layer_stack
    assert resolved.layer_stack.layers["box"].background


def test_wafer_layer_comes_from_the_pdk_layer_map(gpdk_active) -> None:
    assert wafer_layer_tuple(gpdk_active) == tuple(gf.gpdk.LAYER.WAFER) == (999, 0)
    custom = SimpleNamespace(layers=SimpleNamespace(WAFER=(5, 1)))
    assert wafer_layer_tuple(custom) == (5, 1)
    assert wafer_layer_tuple(SimpleNamespace(layers=SimpleNamespace())) == (999, 0)


def test_only_plain_wafer_levels_become_blanket_films() -> None:
    patterned = LogicalLayer(layer=(1, 0)) - LogicalLayer(layer=(2, 0))
    sized = LogicalLayer(layer=(999, 0), sizings_xoffsets=(100,))
    stack = LayerStack(
        layers={
            "bulk": _level((999, 0)),
            "drawn": _level((1, 0)),
            "mask": _level(patterned, derived_layer=LogicalLayer(layer=(3, 0))),
            "sized": _level(sized),
            "already": _level((999, 0), background=True),
        }
    )
    marked = with_wafer_films(stack, SimpleNamespace(layers=SimpleNamespace()))
    assert {key: level.background for key, level in marked.layers.items()} == {
        "bulk": True,
        "drawn": False,
        "mask": False,
        "sized": False,
        "already": True,
    }
    assert not stack.layers["bulk"].background


def test_stack_without_wafer_levels_is_returned_unchanged() -> None:
    stack = LayerStack(layers={"core": _level((1, 0))})
    assert with_wafer_films(stack, SimpleNamespace(layers=None)) is stack


def test_levels_off_the_wafer_layer_keep_polygon_semantics(gpdk_active) -> None:
    """A custom PDK without wafer levels still resolves only drawn layers."""
    pdk = gf.Pdk(
        name="no_wafer_levels",
        layers=gpdk_active.layers,
        cross_sections=gpdk_active.cross_sections,
        cells={"straight": gpdk_active.cells["straight"]},
        layer_stack=LayerStack(
            layers={
                "core": _level(gf.gpdk.LAYER.WG, material="Si"),
                "oxide": _level((99, 0), material="SiO2"),
            }
        ),
    )
    resolved = resolve_passive_pcell("straight", pdk=pdk)
    assert list(resolved.layers) == ["core"]


def test_ambiguous_material_names_fail_with_a_layer_error(gpdk_active) -> None:
    pdk = gf.Pdk(
        name="ambiguous_materials",
        layers=gpdk_active.layers,
        cross_sections=gpdk_active.cross_sections,
        cells={"straight": gpdk_active.cells["straight"]},
        layer_stack=LayerStack(
            layers={"core": _level(gf.gpdk.LAYER.WG, material="CORE")}
        ),
    )
    module = SimpleNamespace(
        PDK=pdk,
        MATERIAL_CARDS={
            "Core": GSIM_MATERIAL_CARDS["Si"].model_copy(update={"name": "Core"}),
            "core": GSIM_MATERIAL_CARDS["SiO2"].model_copy(update={"name": "core"}),
        },
    )
    with pytest.raises(LayerResolutionError, match=r"\['Core', 'core'\]"):
        resolve_passive_pcell("straight", pdk=module)
