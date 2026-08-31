"""Tests for the MaterialCard-to-GDSFactory FDTD material adapter."""

from __future__ import annotations

import pytest
from pdk_schema import (
    Drude,
    DrudeTerm,
    Index,
    Lorentz,
    LorentzTerm,
    Permittivity,
    ScalarValue,
)
from pydantic import ValidationError
from scipy.constants import electron_volt, hbar

from gsim.common.materials import resolve_material_snapshot
from gsim.common.materials._helpers import material_card, wavelength_validity
from gsim.fdtd.material_config import MaterialConfig, material_config_from_snapshot
from gsim.fdtd.models import FDTDConfigError


def _angular_frequency(energy_ev: float) -> float:
    return energy_ev * electron_volt / hbar


def test_scalar_index_card_stays_nondispersive() -> None:
    card = material_card(
        name="constant",
        temperature_ref=None,
        permittivity=Index(
            validity=None,
            variation=None,
            conductivity=None,
            n=ScalarValue(unit="", value=2.0),
            k=None,
        ),
    )
    snapshot = resolve_material_snapshot("constant", 1.55, {"constant": card})

    config = material_config_from_snapshot(snapshot, (1500.0, 1600.0))

    assert config.model_dump(exclude_none=True) == {"refractive_index": 2.0}


def test_sellmeier_card_maps_to_fdtd_coefficients() -> None:
    snapshot = resolve_material_snapshot("SiO2-Malitson", 1.55, {})

    config = material_config_from_snapshot(snapshot, (1500.0, 1600.0))

    assert config.dispersion is not None
    assert config.dispersion.wavelength_range_nm == (1500.0, 1600.0)
    sellmeier = config.dispersion.sellmeier
    assert sellmeier is not None
    assert sellmeier.b == [0.6961663, 0.4079426, 0.8974794]
    assert sellmeier.c_um2 == pytest.approx([0.0684043**2, 0.1162414**2, 9.896161**2])


def test_tabulated_index_card_maps_wavelengths_and_loss() -> None:
    snapshot = resolve_material_snapshot("Si-Li-293K", 1.55, {})

    config = material_config_from_snapshot(snapshot, (1500.0, 1600.0))

    assert config.dispersion is not None
    table = config.dispersion.table
    assert table is not None
    index = table.wavelength_nm.index(1550.0)
    assert table.n[index] == 3.4757
    assert set(table.k) == {0.0}


def test_drude_and_lorentz_cards_map_rad_per_second_to_ev() -> None:
    drude_card = material_card(
        name="metal",
        temperature_ref=None,
        permittivity=Drude(
            validity=wavelength_validity(0.4, 2.0),
            variation=None,
            eps_inf=1.0,
            terms=(
                DrudeTerm(
                    omega_p=_angular_frequency(9.03),
                    gamma=_angular_frequency(0.053),
                ),
            ),
        ),
    )
    lorentz_card = material_card(
        name="resonant",
        temperature_ref=None,
        permittivity=Lorentz(
            validity=wavelength_validity(0.4, 2.0),
            variation=None,
            eps_inf=2.0,
            terms=(
                LorentzTerm(
                    delta_eps=1.09,
                    omega_0=_angular_frequency(2.7),
                    gamma=_angular_frequency(1.2),
                ),
            ),
        ),
    )

    drude = material_config_from_snapshot(
        resolve_material_snapshot("metal", 1.55, {"metal": drude_card}),
        (1500.0, 1600.0),
    )
    lorentz = material_config_from_snapshot(
        resolve_material_snapshot("resonant", 1.55, {"resonant": lorentz_card}),
        (1500.0, 1600.0),
    )

    assert drude.dispersion is not None
    drude_lorentz = drude.dispersion.drude_lorentz
    assert drude_lorentz is not None
    assert drude_lorentz.drude is not None
    assert drude_lorentz.drude.plasma_energy_ev == pytest.approx(9.03)
    assert drude_lorentz.drude.damping_ev == pytest.approx(0.053)
    assert lorentz.dispersion is not None
    lorentz_terms = lorentz.dispersion.drude_lorentz
    assert lorentz_terms is not None
    assert lorentz_terms.lorentz is not None
    assert lorentz_terms.lorentz[0].resonance_ev == pytest.approx(2.7)
    assert lorentz_terms.lorentz[0].damping_ev == pytest.approx(1.2)


def test_scalar_permittivity_card_maps_to_refractive_index() -> None:
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
    snapshot = resolve_material_snapshot("dielectric", 1.55, {"dielectric": card})

    config = material_config_from_snapshot(snapshot, (1500.0, 1600.0))

    assert config.refractive_index == 2.0


def test_dispersive_card_requires_a_source_band() -> None:
    snapshot = resolve_material_snapshot("SiO2", 1.55, {})

    with pytest.raises(FDTDConfigError, match="nonzero frequency band"):
        material_config_from_snapshot(snapshot, None)


def test_fdtd_material_config_requires_exactly_one_shape() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        MaterialConfig()

    with pytest.raises(ValidationError, match="exactly one"):
        MaterialConfig(
            refractive_index=1.5,
            dispersion={
                "wavelength_range_nm": (1500, 1600),
                "sellmeier": {"b": [1.0], "c_um2": [0.01]},
            },
        )
