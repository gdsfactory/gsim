"""The authoritative stack uses PDK RF cards for process-specific materials."""

import pytest

from gsim.common.pdk import resolve_layer_stack


def test_ihp_process_material_cards_take_precedence() -> None:
    ihp = pytest.importorskip("ihp")
    resolved = resolve_layer_stack(pdk=ihp.PDK)
    expected = {
        "cont": 2_390_000.0,
        "metal1": 21_640_000.0,
        "topmetal1": 27_800_000.0,
        "topmetal2": 30_300_000.0,
        "topvia2": 3_143_000.0,
    }
    for layer, conductivity in expected.items():
        assert resolved.layers[layer].material_properties[
            "conductivity"
        ] == pytest.approx(conductivity)
    assert resolved.materials["alcu_sg13g2_topmetal2"]["conductivity"] == pytest.approx(
        30_300_000.0
    )
