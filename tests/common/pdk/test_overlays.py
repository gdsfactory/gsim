"""Tests for the reviewed UBC and Cornerstone stack overlays."""

from __future__ import annotations

import gdsfactory as gf
import pytest
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim.common.pdk import resolve_layer_stack


def _pdk(name: str, layers: dict[str, LayerLevel]) -> gf.Pdk:
    return gf.Pdk(name=name, layer_stack=GfLayerStack(layers=layers))


def _ubc_pdk() -> gf.Pdk:
    return _pdk(
        "ubcpdk",
        {
            "substrate": LayerLevel(
                layer=(999, 0),
                thickness=10.0,
                zmin=-13.0,
                material="si",
                info={"mesh_order": 99},
            ),
            "box": LayerLevel(
                layer=(999, 0),
                thickness=3.0,
                zmin=-3.0,
                material="sio2",
                info={"mesh_order": 99},
            ),
            "clad": LayerLevel(
                layer=(999, 0),
                thickness=1.8,
                zmin=0.0,
                material="sio2",
                info={"mesh_order": 100},
            ),
            "core": LayerLevel(
                layer=(1, 0),
                thickness=0.22,
                zmin=0.0,
                material="si",
                info={"mesh_order": 1},
                sidewall_angle=10.0,
            ),
            "heater": LayerLevel(
                layer=(11, 0),
                thickness=0.75,
                zmin=1.1,
                material="TiN",
                info={"mesh_order": 1},
            ),
            "metal2": LayerLevel(
                layer=(12, 0),
                thickness=0.7,
                zmin=1.8,
                material="Aluminum",
                info={"mesh_order": 2},
            ),
        },
    )


def _cspdk_pdk() -> gf.Pdk:
    return _pdk(
        "cspdk.si220",
        {
            "core": LayerLevel(
                layer=(3, 0),
                thickness=0.22,
                zmin=0.0,
                material="Si",
                info={"mesh_order": 1},
            ),
            "slab": LayerLevel(
                layer=(5, 0),
                thickness=0.1,
                zmin=0.0,
                material="Si",
                info={"mesh_order": 1},
            ),
            "heater": LayerLevel(
                layer=(39, 0),
                thickness=0.15,
                zmin=1.1,
                material="TiN",
                info={"mesh_order": 1},
            ),
            "metal": LayerLevel(
                layer=(41, 0),
                thickness=0.22,
                zmin=1.32,
                material="Al",
                info={"mesh_order": 2},
            ),
        },
    )


def test_ubc_overlay_preserves_authoritative_regions_and_roles() -> None:
    resolved = resolve_layer_stack(pdk=_ubc_pdk())

    assert resolved.overlay == "UBC"
    by_name = {dielectric.name: dielectric for dielectric in resolved.dielectrics}
    assert set(by_name) == {"substrate", "box", "clad"}
    assert by_name["substrate"].material == "si"
    assert (by_name["substrate"].zmin, by_name["substrate"].zmax) == (-13.0, -3.0)
    assert (by_name["box"].zmin, by_name["box"].zmax) == (-3.0, 0.0)
    assert (by_name["clad"].zmin, by_name["clad"].zmax) == (0.0, 1.8)
    assert resolved.layers["substrate"].layer_type == "substrate"
    assert resolved.layers["heater"].layer_type == "conductor"
    assert resolved.layers["metal2"].layer_type == "conductor"
    assert {"si", "sio2", "TiN", "Aluminum"} <= set(resolved.materials)


def test_cspdk_overlay_supplies_implicit_box_and_cladding() -> None:
    resolved = resolve_layer_stack(pdk=_cspdk_pdk())

    assert resolved.overlay == "CSPDK"
    by_name = {dielectric.name: dielectric for dielectric in resolved.dielectrics}
    assert set(by_name) == {"box", "clad"}
    assert (by_name["box"].zmin, by_name["box"].zmax) == (-2.0, 0.0)
    assert (by_name["clad"].zmin, by_name["clad"].zmax) == (0.0, 2.0)
    assert by_name["box"].material == "SiO2"
    assert resolved.layers["heater"].layer_type == "conductor"
    assert resolved.layers["metal"].layer_type == "conductor"
    assert {"Si", "TiN", "Al", "SiO2"} <= set(resolved.materials)


def test_overlays_are_not_applied_to_other_pdks() -> None:
    resolved = resolve_layer_stack(
        pdk=_pdk(
            "some_other_pdk",
            {"core": LayerLevel(layer=(1, 0), thickness=0.22, zmin=0.0, material="Si")},
        )
    )

    assert resolved.overlay is None
    assert resolved.dielectrics == ()


def test_cspdk_overlay_can_be_disabled_or_overridden() -> None:
    bare = resolve_layer_stack(pdk=_cspdk_pdk(), use_pdk_overlay=False)
    assert bare.overlay is None
    assert bare.dielectrics == ()

    overridden = resolve_layer_stack(
        pdk=_cspdk_pdk(),
        dielectrics=[{"name": "clad", "material": "Si3N4", "zmin": 0.0, "zmax": 1.0}],
        role_overrides={"metal": "via"},
    )
    by_name = {dielectric.name: dielectric for dielectric in overridden.dielectrics}
    assert by_name["clad"].material == "Si3N4"
    assert by_name["box"].zmin == -2.0
    assert overridden.layers["metal"].layer_type == "via"


@pytest.mark.parametrize("pdk_name", ["cspdk.si220", "CSPDK.SI220", "cornerstone"])
def test_cspdk_name_variants(pdk_name: str) -> None:
    pdk = _cspdk_pdk()
    pdk.name = pdk_name
    resolved = resolve_layer_stack(pdk=pdk)
    assert resolved.overlay == "CSPDK"
