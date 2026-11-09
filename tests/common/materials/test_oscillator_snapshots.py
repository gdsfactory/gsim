"""Schema oscillator cards evaluated against independent optical references."""

from math import pi, sqrt

import pytest
from pdk_schema import (
    Band,
    Drude,
    DrudeTerm,
    Lorentz,
    LorentzTerm,
    MaterialCard,
    Validity,
)

from gsim.common.materials import (
    GSIM_MATERIAL_CARDS,
    MaterialModelError,
    WavelengthOutOfRangeError,
    resolve_material_snapshot,
)

OMEGA = 2 * pi * 299_792_458 / 1e-6


def oscillator_card(model) -> MaterialCard:
    """Round-trip a synthetic model through the public MaterialCard schema."""
    card = GSIM_MATERIAL_CARDS["SiO2"].model_dump()
    card["name"] = "Oscillator"
    card["optical"]["permittivity"] = model.model_dump()
    return MaterialCard.model_validate(card)


@pytest.mark.parametrize("kind", ["lorentz", "drude"])
def test_schema_oscillators_recover_known_complex_index(kind) -> None:
    # Both independent constructions give epsilon=3.99+0.4j=(2+0.1j)^2.
    if kind == "lorentz":
        model = Lorentz(
            validity=None,
            variation=None,
            eps_inf=3.99,
            terms=(LorentzTerm(delta_eps=0.08, omega_0=OMEGA, gamma=0.2 * OMEGA),),
        )
    else:
        model = Drude(
            validity=None,
            variation=None,
            eps_inf=4.49,
            terms=(DrudeTerm(omega_p=sqrt(0.82) * OMEGA, gamma=0.8 * OMEGA),),
        )
    snapshot = resolve_material_snapshot(
        "Oscillator", 1, {"Oscillator": oscillator_card(model)}
    )
    assert snapshot.source == "project"
    assert snapshot.refractive_index == pytest.approx(2)
    assert snapshot.extinction_coefficient == pytest.approx(0.1)


def test_weak_lorentz_absorption_is_not_rounded_to_zero() -> None:
    model = Lorentz(
        validity=None,
        variation=None,
        eps_inf=4,
        terms=(LorentzTerm(delta_eps=1e-25, omega_0=OMEGA, gamma=OMEGA),),
    )
    snapshot = resolve_material_snapshot(
        "Oscillator", 1, {"Oscillator": oscillator_card(model)}
    )
    assert snapshot.refractive_index == pytest.approx(2)
    assert snapshot.extinction_coefficient == pytest.approx(2.5e-26, rel=1e-14, abs=0)


def test_oscillator_respects_schema_wavelength_validity() -> None:
    model = Lorentz(
        validity=Validity(
            at=None,
            over={"wavelength": Band(min=0.8, max=1.2, unit="um", label=None)},
            on_out_of_range="raise",
        ),
        variation=None,
        eps_inf=2,
        terms=(LorentzTerm(delta_eps=1, omega_0=OMEGA, gamma=OMEGA),),
    )
    with pytest.raises(WavelengthOutOfRangeError, match="valid"):
        resolve_material_snapshot(
            "Oscillator", 2, {"Oscillator": oscillator_card(model)}
        )


def test_undamped_resonance_has_a_material_diagnostic() -> None:
    model = Lorentz(
        validity=None,
        variation=None,
        eps_inf=2,
        terms=(LorentzTerm(delta_eps=1, omega_0=OMEGA, gamma=0),),
    )
    with pytest.raises(
        MaterialModelError, match=r"lorentz material 'Oscillator'.*1 um"
    ):
        resolve_material_snapshot(
            "Oscillator", 1, {"Oscillator": oscillator_card(model)}
        )
