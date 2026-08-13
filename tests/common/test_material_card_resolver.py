from math import sqrt

import pytest
from pdk_schema import Index, MaterialCard, Sellmeier, TabulatedValue

from gsim.common.materials import (
    GSIM_MATERIAL_CARDS,
    MaterialCardNotFoundError,
    MaterialCardResolver,
)


def _evaluate_sellmeier(model: Sellmeier, wavelength_um: float) -> float:
    wavelength_squared = wavelength_um**2
    refractive_index_squared = (
        1.0
        + model.offset
        + sum(
            term.b * wavelength_squared / (wavelength_squared - term.c_um**2)
            for term in model.terms
        )
    )
    return sqrt(refractive_index_squared)


def test_gsim_material_card_names_match_registry_keys() -> None:
    assert set(GSIM_MATERIAL_CARDS) == {
        "Si",
        "Si-Salzberg",
        "Si-Li-293K",
        "SiO2",
        "SiO2-Malitson",
    }
    assert all(
        registry_name == card.name
        for registry_name, card in GSIM_MATERIAL_CARDS.items()
    )


def test_silicon_fallback_uses_salzberg_model() -> None:
    fallback = GSIM_MATERIAL_CARDS["Si"]
    named = GSIM_MATERIAL_CARDS["Si-Salzberg"]

    assert fallback.optical is not None
    assert named.optical is not None
    assert fallback.optical == named.optical
    assert isinstance(fallback.optical.permittivity, Sellmeier)
    assert _evaluate_sellmeier(fallback.optical.permittivity, 1.55) == pytest.approx(
        3.477723756, abs=1e-9
    )


def test_li_293k_card_contains_rii_table() -> None:
    card = GSIM_MATERIAL_CARDS["Si-Li-293K"]

    assert card.optical is not None
    assert isinstance(card.optical.permittivity, Index)
    assert isinstance(card.optical.permittivity.n, TabulatedValue)
    wavelength_values = card.optical.permittivity.n.data.coords["wavelength"].values
    telecom_index = wavelength_values.index(1.55)
    assert len(wavelength_values) == 35
    assert card.optical.permittivity.n.data.values[telecom_index] == 3.4757


def test_silicon_dioxide_fallback_uses_malitson_model() -> None:
    fallback = GSIM_MATERIAL_CARDS["SiO2"]
    named = GSIM_MATERIAL_CARDS["SiO2-Malitson"]

    assert fallback.optical is not None
    assert named.optical is not None
    assert fallback.optical == named.optical
    assert isinstance(fallback.optical.permittivity, Sellmeier)
    assert _evaluate_sellmeier(fallback.optical.permittivity, 1.55) == pytest.approx(
        1.444023622, abs=1e-9
    )


def test_resolver_combines_project_and_gsim_cards() -> None:
    project_silicon = GSIM_MATERIAL_CARDS["Si"].model_copy(deep=True)
    resolver = MaterialCardResolver(project_material_cards={"Si": project_silicon})

    assert resolver.resolve("Si") is project_silicon
    assert resolver.resolve("SiO2") is GSIM_MATERIAL_CARDS["SiO2"]


def test_project_card_takes_precedence_over_gsim_fallback() -> None:
    project_silicon_dioxide = MaterialCard(
        name="SiO2",
        optical=None,
        rf=None,
        info={"source": "project"},
    )
    resolver = MaterialCardResolver(
        project_material_cards={"SiO2": project_silicon_dioxide}
    )

    assert resolver.resolve("SiO2") is project_silicon_dioxide


def test_resolver_uses_exact_case_sensitive_names() -> None:
    resolver = MaterialCardResolver()

    with pytest.raises(MaterialCardNotFoundError, match="'sio2'"):
        resolver.resolve("sio2")
