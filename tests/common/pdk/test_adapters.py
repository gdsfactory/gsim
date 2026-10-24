"""Tests for the canonical PDK stack adapters."""

from __future__ import annotations

import gdsfactory as gf
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack

from gsim.common.pdk import build_layer_stack, resolve_layer_stack
from gsim.palace.stack import build_palace_layer_stack


def _make_pdk() -> gf.Pdk:
    return gf.Pdk(
        name="adapter_test",
        layer_stack=GfLayerStack(
            layers={
                "metal": LayerLevel(
                    layer=(1, 0),
                    thickness=0.5,
                    zmin=0.5,
                    material="aluminum",
                    info={"mesh_order": 7, "layer_type": "conductor"},
                ),
                "core": LayerLevel(
                    layer=(2, 0),
                    thickness=0.22,
                    zmin=0.0,
                    material="Si",
                    info={"mesh_order": 1},
                    sidewall_angle=10.0,
                ),
                "box": LayerLevel(
                    layer=(999, 0),
                    thickness=2.0,
                    zmin=-2.0,
                    material="SiO2",
                    info={"mesh_order": 99},
                ),
            }
        ),
    )


def test_adapter_preserves_layers_materials_and_dielectrics() -> None:
    resolved = resolve_layer_stack(
        pdk=_make_pdk(),
        dielectrics=[
            {"name": "box_region", "material": "SiO2", "zmin": -2.0, "zmax": 0.0}
        ],
    )

    stack = build_layer_stack(resolved, backend="meep")

    assert stack.pdk_name == "adapter_test"
    assert set(stack.layers) == {"metal", "core", "box"}
    assert stack.layers["core"].sidewall_angle == 10.0
    assert stack.layers["metal"].layer_type == "conductor"
    assert stack.layers["metal"].mesh_order == 7
    assert stack.layers["box"].gds_layer == (999, 0)
    assert stack.dielectrics == [
        {"name": "box_region", "zmin": -2.0, "zmax": 0.0, "material": "SiO2"}
    ]
    # The dielectric material must be in the material table even when it is not
    # a separate PDK layer.
    assert "SiO2" in stack.materials


def test_adapter_records_auditable_provenance() -> None:
    resolved = resolve_layer_stack(
        pdk=_make_pdk(),
        dielectrics=[
            {"name": "box_region", "material": "SiO2", "zmin": -2.0, "zmax": 0.0}
        ],
        wavelength_um=1.55,
    )

    stack = build_layer_stack(resolved, backend="meep")

    assert stack.simulation["pdk_authoritative"] is True
    assert stack.simulation["backend"] == "meep"
    assert stack.simulation["wavelength_um"] == 1.55
    assert stack.simulation["stack_hash"] == resolved.stack_hash
    assert stack.simulation["pdk_provenance"]["pdk_name"] == "adapter_test"
    assert stack.simulation["pdk_provenance"]["layers"]["core"]["mesh_order"] == 1


def test_palace_adapter_delegates_to_the_shared_adapter() -> None:
    resolved = resolve_layer_stack(pdk=_make_pdk())

    stack = build_palace_layer_stack(resolved)

    assert stack.simulation["backend"] == "palace"
    assert stack.simulation["stack_hash"] == resolved.stack_hash
    assert set(stack.layers) == set(resolved.layers)


def test_stack_hash_is_deterministic_and_sensitive_to_content() -> None:
    first = resolve_layer_stack(pdk=_make_pdk())
    second = resolve_layer_stack(pdk=_make_pdk())
    changed = resolve_layer_stack(
        pdk=_make_pdk(),
        dielectrics=[
            {"name": "box_region", "material": "SiO2", "zmin": -2.0, "zmax": 0.0}
        ],
    )

    assert first.stack_hash == second.stack_hash
    assert first.stack_hash != changed.stack_hash
