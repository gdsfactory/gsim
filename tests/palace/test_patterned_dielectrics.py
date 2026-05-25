"""Tests for patterned dielectric mesh support and stack dielectric toggles."""

from __future__ import annotations

from types import SimpleNamespace

from gdsfactory.technology import LayerLevel

from gsim.common.stack.extractor import Layer, LayerStack, extract_layer_stack
from gsim.palace.mesh.geometry import build_entities


def _fake_gf_stack():
    """Build a minimal gdsfactory-like stack for extractor tests."""
    return SimpleNamespace(
        layers={
            "core": LayerLevel(layer=(1, 0), thickness=0.22, zmin=0.0, material="si"),
            "metal1": LayerLevel(
                layer=(2, 0), thickness=1.0, zmin=1.0, material="aluminum"
            ),
        }
    )


def test_extract_layer_stack_can_disable_synthetic_dielectrics():
    """Synthetic oxide/passivation can be disabled explicitly."""
    stack = extract_layer_stack(
        _fake_gf_stack(),
        pdk_name="test-pdk",
        add_oxide_dielectric=False,
        add_passivation_dielectric=False,
    )

    names = [d["name"] for d in stack.dielectrics]
    assert names == []
    assert stack.simulation["add_oxide_dielectric"] is False
    assert stack.simulation["add_passivation_dielectric"] is False


def test_extract_layer_stack_defaults_keep_synthetic_dielectrics():
    """Defaults preserve synthetic oxide/passive dielectric regions."""
    stack = extract_layer_stack(_fake_gf_stack(), pdk_name="test-pdk")

    names = [d["name"] for d in stack.dielectrics]
    assert "oxide" in names
    assert "passive" in names
    assert "air_box" not in names


def test_build_entities_prioritizes_patterned_dielectrics_over_background_boxes():
    """Patterned dielectric volumes must be cut before blanket boxes."""
    stack_layers = {
        "via1": Layer(
            name="via1",
            gds_layer=(10, 0),
            zmin=0.0,
            zmax=1.0,
            thickness=1.0,
            material="tungsten",
            layer_type="via",
        )
    }
    stack = LayerStack(layers=stack_layers)

    metal_tags = {
        "via1": {
            "volumes": [41],
            "surfaces_xy": [],
            "surfaces_z": [],
        }
    }
    dielectric_tags = {"SiO2": [31], "airbox": [32]}
    patterned_dielectric_tags = {"core": [21]}
    port_tags = {"P1": [11]}
    port_info = [{"portnumber": 1}]

    entities = build_entities(
        metal_tags=metal_tags,
        dielectric_tags=dielectric_tags,
        patterned_dielectric_tags=patterned_dielectric_tags,
        port_tags=port_tags,
        port_info=port_info,
        stack=stack,
    )

    orders = {entity.name: entity.mesh_order for entity in entities}
    assert orders["via1"] == 1
    assert orders["P1"] == -1
    assert orders["core"] == 2
    assert orders["SiO2"] == 3
    assert orders["air"] == 4
