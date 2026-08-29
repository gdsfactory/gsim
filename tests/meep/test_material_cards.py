"""Tests for MaterialCard compatibility validation and MEEP conversion."""

from __future__ import annotations

import math

import pytest
from pdk_schema import (
    Cauchy,
    CauchyTerm,
    Debye,
    DebyeTerm,
    Drude,
    DrudeTerm,
    Index,
    Lorentz,
    LorentzTerm,
    Permittivity,
    Pole,
    PoleResidue,
    ScalarValue,
    SellmeierPoleSquared,
    SellmeierSquaredTerm,
)
from scipy.constants import c as C0  # noqa: N812

from gsim.common.materials import SI_LI_293K, SI_SALZBERG, SIO2_MALITSON
from gsim.common.materials._helpers import material_card
from gsim.meep.material_cards import (
    MeepMaterialCompatibilityError,
    material_data_from_card,
    validate_meep_material_card,
)


def _index_card(name: str = "constant", *, k: float | None = None):
    """Return a scalar index-authored card for tests."""
    return material_card(
        name=name,
        temperature_ref=None,
        permittivity=Index(
            validity=None,
            variation=None,
            conductivity=None,
            n=ScalarValue(unit="", value=2.0),
            k=None if k is None else ScalarValue(unit="", value=k),
        ),
    )


def test_builtin_si_and_sio2_cards_are_meep_compatible() -> None:
    validate_meep_material_card(SI_SALZBERG, (1.5, 1.6))
    validate_meep_material_card(SIO2_MALITSON, (1.5, 1.6))


def test_scalar_index_card_becomes_nondispersive_epsilon() -> None:
    material = material_data_from_card(_index_card())

    assert material.epsilon_diag == [4.0, 4.0, 4.0]
    assert material.epsilon_susceptibilities is None


def test_scalar_permittivity_card_stays_nondispersive() -> None:
    card = material_card(
        name="dielectric",
        temperature_ref=None,
        permittivity=Permittivity(
            validity=None,
            variation=None,
            conductivity=None,
            eps_real=ScalarValue(unit="", value=4.0),
            eps_imag=None,
        ),
    )

    material = material_data_from_card(card)

    assert material.epsilon_diag == [4.0, 4.0, 4.0]


def test_sellmeier_maps_exactly_to_lorentz_terms() -> None:
    material = material_data_from_card(SIO2_MALITSON, (1.5, 1.6))

    terms = material.epsilon_susceptibilities or []
    assert material.epsilon_diag == [1.0, 1.0, 1.0]
    assert [term.frequency for term in terms] == pytest.approx(
        [1 / 0.0684043, 1 / 0.1162414, 1 / 9.896161]
    )
    assert [term.sigma for term in terms] == pytest.approx(
        [0.6961663, 0.4079426, 0.8974794]
    )
    assert {term.gamma for term in terms} == {0.0}


def test_squared_sellmeier_poles_are_not_squared_twice() -> None:
    card = material_card(
        name="squared",
        temperature_ref=None,
        permittivity=SellmeierPoleSquared(
            validity=None,
            variation=None,
            conductivity=None,
            terms=(SellmeierSquaredTerm(b=0.5, c_um2=0.04),),
            offset=0.25,
        ),
    )

    material = material_data_from_card(card)

    assert material.epsilon_diag == [1.25, 1.25, 1.25]
    term = (material.epsilon_susceptibilities or [])[0]
    assert term.frequency == pytest.approx(5.0)
    assert term.sigma == 0.5


def test_lorentz_and_drude_terms_convert_si_frequencies() -> None:
    lorentz_card = material_card(
        name="lorentz",
        temperature_ref=None,
        permittivity=Lorentz(
            validity=None,
            variation=None,
            eps_inf=2.0,
            terms=(LorentzTerm(delta_eps=1.2, omega_0=2e15, gamma=1e13),),
        ),
    )
    drude_card = material_card(
        name="drude",
        temperature_ref=None,
        permittivity=Drude(
            validity=None,
            variation=None,
            eps_inf=3.0,
            terms=(DrudeTerm(omega_p=3e15, gamma=2e13),),
        ),
    )

    lorentz = material_data_from_card(lorentz_card)
    drude = material_data_from_card(drude_card)

    conversion = 1e-6 / (2 * math.pi * C0)
    lorentz_term = (lorentz.epsilon_susceptibilities or [])[0]
    drude_term = (drude.epsilon_susceptibilities or [])[0]
    assert lorentz.epsilon_diag == [2.0, 2.0, 2.0]
    assert lorentz_term.kind == "lorentzian"
    assert lorentz_term.frequency == pytest.approx(2e15 * conversion)
    assert lorentz_term.gamma == pytest.approx(1e13 * conversion)
    assert lorentz_term.sigma == 1.2
    assert drude.epsilon_diag == [3.0, 3.0, 3.0]
    assert drude_term.kind == "drude"
    assert drude_term.frequency == pytest.approx(3e15 * conversion)
    assert drude_term.gamma == pytest.approx(2e13 * conversion)
    assert drude_term.sigma == 1.0


@pytest.mark.parametrize(
    "model",
    [
        Cauchy(
            validity=None,
            variation=None,
            conductivity=None,
            a=1.5,
            terms=(CauchyTerm(b=0.01, power=-2),),
        ),
        Debye(
            validity=None,
            variation=None,
            eps_inf=2.0,
            terms=(DebyeTerm(delta_eps=1.0, tau=1e-12),),
        ),
        PoleResidue(
            validity=None,
            variation=None,
            eps_inf=2.0,
            poles=(Pole(a=(1.0, 2.0), c=(3.0, 4.0)),),
        ),
    ],
)
def test_non_native_closed_form_models_are_rejected(model) -> None:
    card = material_card(name="unsupported", temperature_ref=None, permittivity=model)

    with pytest.raises(MeepMaterialCompatibilityError, match="unsupported optical"):
        validate_meep_material_card(card)


def test_tabulated_index_requires_causal_fit() -> None:
    with pytest.raises(MeepMaterialCompatibilityError, match="causal Lorentz/Drude"):
        validate_meep_material_card(SI_LI_293K, (1.5, 1.6))


def test_constant_extinction_requires_causal_fit() -> None:
    with pytest.raises(MeepMaterialCompatibilityError, match="extinction coefficient"):
        validate_meep_material_card(_index_card(k=0.01))


def test_model_conductivity_is_rejected_explicitly() -> None:
    card = material_card(
        name="conductive",
        temperature_ref=None,
        permittivity=Index(
            validity=None,
            variation=None,
            conductivity=ScalarValue(unit="S/m", value=1.0),
            n=ScalarValue(unit="", value=2.0),
            k=None,
        ),
    )

    with pytest.raises(MeepMaterialCompatibilityError, match="model conductivity"):
        validate_meep_material_card(card)


def test_source_band_must_fit_material_validity() -> None:
    with pytest.raises(MeepMaterialCompatibilityError, match="simulation band"):
        validate_meep_material_card(SI_SALZBERG, (1.2, 1.3))
