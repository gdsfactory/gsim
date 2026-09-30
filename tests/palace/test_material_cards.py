"""PDK RF cards must survive extraction, YAML, and Palace config generation."""

import json
from types import SimpleNamespace

import pytest
from gdsfactory.technology import LayerLevel
from gdsfactory.technology import LayerStack as GfLayerStack
from pdk_schema import MaterialCard, Permittivity, Provenance, Regime, ScalarValue

from gsim.common.materials.rf import resolve_rf_material_card
from gsim.common.stack import extract_from_pdk, load_stack_yaml
from gsim.palace import DrivenSim
from gsim.palace.materials import resolve_palace_materials_at_frequency
from gsim.palace.mesh.config_generator import generate_palace_config
from gsim.palace.models import DrivenConfig


def _card(name, *, sigma=None, eps=None, eps_imag=None):
    model = (
        None
        if eps is None
        else Permittivity(
            eps_real=ScalarValue(value=eps, unit=""),
            eps_imag=None if eps_imag is None else ScalarValue(value=eps_imag, unit=""),
            conductivity=None
            if sigma is None
            else ScalarValue(value=sigma, unit="S/m"),
            validity=None,
            variation=None,
        )
    )
    return MaterialCard(
        name=name,
        optical=None,
        info={},
        rf=Regime(
            temperature_ref=None,
            provenance=Provenance(
                source="user",
                label="test",
                maturity=None,
                citations=[],
                comment=None,
                url=None,
                data_url=None,
                info={},
            ),
            permittivity=model,
            conductivity=ScalarValue(value=sigma, unit="S/m")
            if model is None
            else None,
            permeability=None,
            perturbations=[],
            info={},
        ),
    )


def _pdk(eps=5.0):
    return SimpleNamespace(
        name="test-pdk",
        material_cards={
            "metal": _card("metal", sigma=21.64e6),
            "via": _card("via", sigma=1.66e6),
            # Deliberately overlaps the legacy database; PDK must win.
            "sio2": _card("sio2", eps=eps),
            "silicon": _card("silicon", eps=11.9, sigma=2),
        },
        layer_stack=GfLayerStack(
            layers={
                "metal1": LayerLevel(
                    layer=(8, 0), zmin=1, thickness=0.42, material="metal"
                ),
                "via1": LayerLevel(
                    layer=(19, 0), zmin=1.42, thickness=0.54, material="via"
                ),
                "oxide": LayerLevel(
                    layer=(999, 0), zmin=0, thickness=2, material="sio2"
                ),
            }
        ),
    )


def test_pdk_card_precedence_and_isolation():
    first = extract_from_pdk(_pdk(5.0), include_substrate=True)
    second = extract_from_pdk(_pdk(6.0))
    assert first.materials["sio2"]["permittivity"] == 5
    assert second.materials["sio2"]["permittivity"] == 6
    assert first.materials["silicon"]["conductivity"] == 2
    resolved = resolve_palace_materials_at_frequency(first.materials, 50e9)
    assert resolved["sio2"]["permittivity"] == 5
    assert resolved["metal"]["conductivity"] == 21.64e6


def test_dual_regime_card_keeps_its_optical_model():
    from scipy.constants import c

    from gsim.common.materials.sio2_malitson import SIO2_MALITSON

    pdk = _pdk()
    pdk.material_cards["sio2"].optical = SIO2_MALITSON.optical
    stack = extract_from_pdk(pdk)
    optical = resolve_palace_materials_at_frequency(stack.materials, c / 1.55e-6)
    rf = resolve_palace_materials_at_frequency(stack.materials, 50e9)
    assert optical["sio2"]["permittivity"] == pytest.approx(2.0852, abs=0.001)
    assert rf["sio2"]["permittivity"] == 5


def test_module_cards_and_yaml_preserve_provenance(tmp_path):
    pdk = _pdk()
    pdk.material_cards["metal"].info = {
        "composition": "Ti/TiN/AlCu/Ti/TiN",
        "composition_source_url": "https://doi.org/10.1109/TCPMT.2022.3172502",
    }
    # Real PDKs mix RF cards with legacy models containing validity tuples.
    pdk.layer_stack.layers["nitride"] = LayerLevel(
        layer=(998, 0), zmin=2, thickness=0.4, material="sin"
    )
    module = SimpleNamespace(PDK=pdk, LAYER_STACK=pdk.layer_stack)
    stack = extract_from_pdk(module)
    path = tmp_path / "stack.yaml"
    stack.to_yaml(path)
    restored = load_stack_yaml(path)
    assert restored.materials == stack.materials
    assert restored.materials["metal"]["material_card"]["info"] == (
        pdk.material_cards["metal"].info
    )
    assert restored.materials["sin"]["dispersion_models"]
    assert (
        resolve_palace_materials_at_frequency(restored.materials, 50e9)["sio2"][
            "permittivity"
        ]
        == 5
    )


@pytest.mark.parametrize("prebuilt", [True, False])
def test_explicit_material_override_wins(prebuilt, monkeypatch):
    sim = DrivenSim()
    stack = extract_from_pdk(_pdk())
    if prebuilt:
        sim.set_stack(stack)
    else:
        monkeypatch.setattr("gdsfactory.get_active_pdk", _pdk)
        sim.set_stack()
    sim.set_material("sio2", permittivity=8.0)
    resolved = resolve_palace_materials_at_frequency(
        sim._resolve_stack().materials, 50e9
    )
    assert resolved["sio2"]["permittivity"] == 8
    assert "material_card" not in resolved["sio2"]


@pytest.mark.parametrize("simulation_type", ["driven", "electrostatic", "eigenmode"])
def test_palace_config_uses_cards_for_domains_and_surfaces(tmp_path, simulation_type):
    stack = extract_from_pdk(_pdk(), include_substrate=True)
    path = generate_palace_config(
        groups={
            "volumes": {
                "via1": {"phys_group": 10, "is_via": True},
                "sio2": {"phys_group": 11},
                "silicon": {"phys_group": 12},
            },
            "conductor_surfaces": {"metal1_xy": {"phys_group": 20}},
            "pec_surfaces": {},
            "port_surfaces": {},
            "boundary_surfaces": {},
        },
        ports=[],
        port_info=[],
        stack=stack,
        output_path=tmp_path,
        model_name="palace",
        fmax=50e9,
        simulation_type=simulation_type,
        driven_config=DrivenConfig(fmin=1e9, fmax=50e9)
        if simulation_type == "driven"
        else None,
        absorbing_boundary=False,
    )
    config = json.loads(path.read_text())
    materials = {m["Attributes"][0]: m for m in config["Domains"]["Materials"]}
    assert materials[10]["Conductivity"] == 1.66e6
    assert materials[11]["Permittivity"] == 5
    assert materials[12]["Permittivity"] == 11.9
    assert materials[12]["Conductivity"] == 2
    if simulation_type != "electrostatic":
        assert config["Boundaries"]["Conductivity"][0]["Conductivity"] == 21.64e6


def test_complex_permittivity_loss_conversion():
    props = resolve_rf_material_card(_card("dielectric", eps=4, eps_imag=0.08))
    assert props["loss_tangent"] == pytest.approx(0.02)


def test_combined_loss_channels_are_not_silently_dropped():
    with pytest.raises(ValueError, match="simultaneous RF conductivity"):
        resolve_rf_material_card(_card("lossy", eps=4, eps_imag=0.08, sigma=2))


@pytest.mark.parametrize("sigma", [-1.0, float("nan"), float("inf")])
def test_invalid_conductivity_fails_loudly(sigma):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        resolve_rf_material_card(_card("bad", sigma=sigma))


def test_unsupported_rf_model_fails_instead_of_using_legacy_database():
    from gsim.common.materials.sio2_malitson import SIO2_MALITSON

    pdk = _pdk()
    assert SIO2_MALITSON.optical is not None
    pdk.material_cards["sio2"].rf.permittivity = SIO2_MALITSON.optical.permittivity
    with pytest.raises(TypeError, match="unsupported RF model"):
        extract_from_pdk(pdk)


def test_diagonal_permittivity_and_conductivity():
    card = _card("anisotropic", eps=4)
    card.rf.permittivity.eps_real = [ScalarValue(value=v, unit="") for v in [4, 5, 6]]
    card.rf.permittivity.conductivity = [
        ScalarValue(value=v, unit="S/m") for v in [1, 2, 3]
    ]
    props = resolve_rf_material_card(card)
    assert props["permittivity"] == [4, 5, 6]
    assert props["conductivity"] == [1, 2, 3]


def test_ihp_cards_reach_palace_stack(tmp_path):
    ihp = pytest.importorskip("ihp")
    if not hasattr(ihp.PDK, "material_cards"):
        pytest.skip("requires IHP release with RF material cards")
    stack = extract_from_pdk(ihp, include_substrate=True)
    path = tmp_path / "ihp-stack.yaml"
    stack.to_yaml(path)
    restored = load_stack_yaml(path)
    assert restored.materials == stack.materials
    stack = restored
    resolved = resolve_palace_materials_at_frequency(stack.materials, 50e9)
    expected = {"metal1": 21.64e6, "via1": 1.66e6, "topmetal2": 30.3e6, "mim": 0.5e6}
    for layer, sigma in expected.items():
        name = stack.layers[layer].material
        assert resolved[name]["conductivity"] == sigma
        assert resolved[name]["material_card"] == ihp.PDK.material_cards[
            name
        ].model_dump(mode="json")
    assert resolved["sio2"]["permittivity"] == 4.1
    assert resolved["silicon"]["conductivity"] == 2
