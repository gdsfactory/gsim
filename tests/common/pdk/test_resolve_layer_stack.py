"""Tests for the PDK-authoritative stack resolver (gsim#230)."""

from __future__ import annotations

from collections.abc import Iterator

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim.common.pdk import (
    LayerResolutionError,
    classify_layer_role,
    resolve_layer_stack,
    resolve_material_properties,
)


def _make_pdk(*, core_material: str = "Si") -> gf.Pdk:
    layer_stack = GfLayerStack(
        layers={
            "metal1": LayerLevel(
                layer=(1, 0),
                thickness=0.5,
                zmin=0.0,
                material="aluminum",
                info={"mesh_order": 7, "layer_type": "conductor"},
            ),
            "via1": LayerLevel(
                layer=(2, 0),
                thickness=0.5,
                zmin=0.5,
                material="tungsten",
                info={"mesh_order": 8},
            ),
            "core": LayerLevel(
                layer=(3, 0),
                thickness=0.22,
                zmin=0.0,
                material=core_material,
            ),
            "clad": LayerLevel(
                layer=(4, 0),
                thickness=2.0,
                zmin=-2.0,
                material="SiO2",
                info={"layer_type": "dielectric"},
            ),
        }
    )
    return gf.Pdk(name="stack_test", layer_stack=layer_stack)


def _component_on(*layers: tuple[int, int]) -> gf.Component:
    component = gf.Component()
    for layer in layers:
        component.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=layer)
    return component


def test_honors_info_mesh_order_and_layer_type() -> None:
    resolved = resolve_layer_stack(pdk=_make_pdk())
    # mesh_order comes from info, not the LayerLevel default (3).
    assert resolved.layers["metal1"].mesh_order == 7
    assert resolved.layers["via1"].mesh_order == 8
    # explicit info['layer_type'] is authoritative.
    assert resolved.layers["metal1"].layer_type == "conductor"
    assert resolved.layers["clad"].layer_type == "dielectric"


def test_role_is_property_based_without_info() -> None:
    resolved = resolve_layer_stack(pdk=_make_pdk())
    assert resolved.layers["via1"].layer_type == "conductor"  # tungsten
    assert resolved.layers["core"].layer_type == "dielectric"  # silicon, no cond.


def test_role_overrides_declare_interconnect() -> None:
    resolved = resolve_layer_stack(
        pdk=_make_pdk(), role_overrides={"via1": "via", "metal1": "conductor"}
    )
    assert resolved.layers["via1"].layer_type == "via"


def test_role_override_unknown_layer_raises() -> None:
    with pytest.raises(LayerResolutionError, match="unknown layer"):
        resolve_layer_stack(pdk=_make_pdk(), role_overrides={"nope": "via"})


def test_no_synthetic_dielectrics() -> None:
    resolved = resolve_layer_stack(pdk=_make_pdk())
    assert resolved.dielectrics == ()


def test_unknown_material_raises() -> None:
    with pytest.raises(LayerResolutionError, match="No material properties"):
        resolve_layer_stack(pdk=_make_pdk(core_material="MissingOnPurpose"))


def test_drawn_extent_excludes_undrawn_layers() -> None:
    component = _component_on((1, 0))  # only metal1, z 0.0..0.5
    resolved = resolve_layer_stack(component, pdk=_make_pdk())
    assert resolved.layers["metal1"].drawn is True
    assert resolved.layers["clad"].drawn is False
    assert resolved.drawn_z_range == (0.0, 0.5)
    assert resolved.undrawn_z_range == (-2.0, 1.0)


def test_material_tag_normalization() -> None:
    # IHP-style lowercase/chemistry tokens normalize to registry entries.
    assert resolve_material_properties("si") is not None
    assert resolve_material_properties("sio2") is not None
    assert resolve_material_properties("sin") is not None


def test_classify_layer_role_from_properties() -> None:
    assert classify_layer_role(resolve_material_properties("aluminum")) == "conductor"
    assert classify_layer_role(resolve_material_properties("SiO2")) == "dielectric"


# ---------------------------------------------------------------------------
# IHP integration
# ---------------------------------------------------------------------------

ihp = pytest.importorskip("ihp")


@pytest.fixture
def _ihp_active() -> Iterator[None]:
    import gdsfactory as gf

    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    ihp.PDK.activate()
    try:
        yield
    finally:
        if previous is not None:
            previous.activate()


@pytest.mark.usefixtures("_ihp_active")
def test_ihp_preserves_pdk_layers_and_applies_its_overlay() -> None:
    resolved = resolve_layer_stack()

    assert resolved.pdk_name == "IHP"
    assert len(resolved.layers) == 31

    # PDK passivation is preserved as a distinct dielectric region.
    passivation_layer = resolved.layers["passiv"]  # codespell:ignore passiv
    assert passivation_layer.material == "sin"
    assert passivation_layer.layer_type == "dielectric"
    assert (passivation_layer.zmin, passivation_layer.zmax) == pytest.approx(
        (14.03, 15.93)
    )

    # Conductors keep their PDK z-extent and material.
    metal1 = resolved.layers["metal1"]
    assert metal1.layer_type == "conductor"
    assert metal1.material == "aluminum"
    assert (metal1.zmin, metal1.zmax) == pytest.approx((0.84, 1.26))

    # The reviewed IHP overlay supplies the implicit BEOL region and marks the
    # tungsten interconnects as vias (explicit, not inferred in the core).
    assert resolved.overlay == "IHP"
    assert {d.name for d in resolved.dielectrics} == {"ild", "passivation"}
    assert resolved.layers["via1"].layer_type == "via"
    assert resolved.layers["cont"].layer_type == "via"

    # Every material token resolves (no silent "unknown" fallback).
    assert {"aluminum", "tungsten", "TiN", "poly_si", "si", "sin", "sio2"} <= set(
        resolved.materials
    )

    # Real mesh orders come from info.
    assert metal1.mesh_order == 14
    assert resolved.layers["via1"].mesh_order == 15


@pytest.mark.usefixtures("_ihp_active")
def test_overlay_can_be_disabled_and_overridden() -> None:
    # No overlay: nothing is synthesized, vias stay property-classified.
    bare = resolve_layer_stack(use_pdk_overlay=False)
    assert bare.overlay is None
    assert bare.dielectrics == ()
    assert bare.layers["via1"].layer_type == "conductor"

    # Explicit input wins over the overlay (same region name -> override).
    overridden = resolve_layer_stack(
        dielectrics=[{"name": "ild", "material": "sin", "zmin": -1.0, "zmax": 5.0}],
        role_overrides={"via1": "conductor"},
    )
    by_name = {d.name: d for d in overridden.dielectrics}
    assert by_name["ild"].material == "sin"
    assert (by_name["ild"].zmin, by_name["ild"].zmax) == (-1.0, 5.0)
    assert by_name["passivation"].material == "sin"  # overlay still applied
    assert overridden.layers["via1"].layer_type == "conductor"
