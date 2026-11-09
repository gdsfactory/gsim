"""Keep process-specific RF properties when extracting a PDK stack."""

import pytest
import yaml
from pdk_schema import ScalarValue

from gsim.common.materials.registry import GSIM_MATERIAL_CARDS
from gsim.common.stack import extract_from_pdk
from gsim.common.stack.materials import get_material_properties
from gsim.common.stack.rf_materials import constant_rf_properties


@pytest.fixture
def ihp_pdk():
    """Use the released PDK's real cards and fabrication stack."""
    return pytest.importorskip("ihp")


@pytest.mark.parametrize("module", [False, True])
def test_ihp_stack_retains_process_conductivity_and_dielectric_constants(
    ihp_pdk, module, tmp_path
):
    bulk_oxide = get_material_properties("sio2")
    assert bulk_oxide is not None
    original_oxide = bulk_oxide.model_dump()
    path = tmp_path / "stack.yml"
    stack = extract_from_pdk(
        ihp_pdk if module else ihp_pdk.PDK,
        include_substrate=True,
        output_path=path,
    )
    for layer, conductivity in {
        "metal1": 21_640_000.0,
        "topmetal1": 27_800_000.0,
        "topmetal2": 30_300_000.0,
        "topvia1": 2_191_000.0,
        "topvia2": 3_143_000.0,
        "mim": 500_000.0,
        "vmim": 2_191_000.0,
    }.items():
        material = stack.layers[layer].material
        assert stack.materials[material]["conductivity"] == conductivity
    assert stack.materials["sio2"]["permittivity"] == 4.1
    assert stack.materials["silicon"]["permittivity"] == 11.9
    assert stack.materials["silicon"]["conductivity"] == 2.0
    assert yaml.full_load(path.read_text())["materials"] == stack.materials
    assert bulk_oxide.model_dump() == original_oxide


def test_rf_loss_is_retained_alongside_conductivity(ihp_pdk):
    card = ihp_pdk.MATERIAL_CARDS["silicon"].model_copy(deep=True)
    card.rf.permittivity.eps_imag = ScalarValue(value=0.119, unit="")
    properties = constant_rf_properties(card)
    assert properties is not None
    assert properties["loss_tangent"] == pytest.approx(0.01)
    assert properties["conductivity"] == 2.0


def test_optical_card_keeps_existing_stack_fallback():
    assert constant_rf_properties(GSIM_MATERIAL_CARDS["Si"]) is None


def test_rf_reader_rejects_wrong_units_instead_of_using_bulk_defaults(ihp_pdk):
    card = ihp_pdk.MATERIAL_CARDS["tungsten_sg13g2_topvia1"].model_copy(deep=True)
    card.rf.conductivity = ScalarValue(value=1.0, unit="")
    with pytest.raises(ValueError, match="constant scalar"):
        constant_rf_properties(card)


def test_rf_reader_rejects_dispersive_models_instead_of_dropping_them(ihp_pdk):
    card = ihp_pdk.MATERIAL_CARDS["silicon"].model_copy(deep=True)
    optical = GSIM_MATERIAL_CARDS["Si"].optical
    assert optical is not None
    card.rf.permittivity = optical.permittivity
    with pytest.raises(ValueError, match="constant permittivity"):
        constant_rf_properties(card)
