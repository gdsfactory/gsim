from __future__ import annotations

import pytest
from pdk_schema import Index, MaterialCard, TabulatedValue

from gsim.common.materials import (
    GSIM_MATERIAL_CARDS,
    SI_LI_293K,
    MaterialModelError,
    MaterialNotFoundError,
    WavelengthOutOfRangeError,
    resolve_material_snapshot,
)


def _li_index_at(wavelength_um: float, temperature_kelvin: float) -> float:
    assert SI_LI_293K.optical is not None
    model = SI_LI_293K.optical.permittivity
    assert isinstance(model, Index)
    assert isinstance(model.n, TabulatedValue)
    table = model.n.data
    wavelengths_um = table.coords["wavelength"].values
    temperatures_kelvin = table.coords["temperature"].values
    wavelength_index = wavelengths_um.index(wavelength_um)
    temperature_index = temperatures_kelvin.index(temperature_kelvin)
    flat_index = wavelength_index * len(temperatures_kelvin) + temperature_index
    return table.values[flat_index]


@pytest.mark.parametrize(
    ("material_name", "expected_index"),
    [
        ("Si-Salzberg", 3.477723756),
        ("Si-Li-293K", 3.475687046),
        ("SiN-Luke", 1.996279731714),
        ("SiO2-Malitson", 1.444023622),
    ],
)
def test_material_snapshots_at_telecom_wavelength(
    material_name: str,
    expected_index: float,
) -> None:
    snapshot = resolve_material_snapshot(material_name, 1.55, {})

    assert snapshot.refractive_index == pytest.approx(expected_index, abs=1e-9)
    assert snapshot.extinction_coefficient == 0
    assert snapshot.source == "gsim"


def test_tabulated_material_interpolates() -> None:
    snapshot = resolve_material_snapshot("Si-Li-293K", 1.525, {})

    assert snapshot.refractive_index == pytest.approx(3.477775832)


def test_li_card_supports_temperature_axis() -> None:
    assert SI_LI_293K.optical is not None
    model = SI_LI_293K.optical.permittivity
    assert isinstance(model, Index)
    assert isinstance(model.n, TabulatedValue)
    assert model.n.data.dims == ("wavelength", "temperature")
    temperature_coordinate = model.n.data.coords["temperature"]
    assert temperature_coordinate.unit == "K"
    assert temperature_coordinate.values[0] == 100.0
    assert temperature_coordinate.values[-1] == 750.0
    assert SI_LI_293K.optical.temperature_ref == 293.0

    assert model.validity is not None
    assert model.validity.over is not None
    temperature_band = model.validity.over["temperature"]
    assert temperature_band.min == 100.0
    assert temperature_band.max == 750.0
    assert _li_index_at(1.55, 100.0) == pytest.approx(3.4467109, abs=1e-7)
    assert _li_index_at(1.55, 293.0) == pytest.approx(3.4756870, abs=1e-7)
    assert _li_index_at(1.55, 750.0) == pytest.approx(3.5767183, abs=1e-7)


def test_li_card_cites_primary_paper() -> None:
    assert SI_LI_293K.optical is not None
    citations = SI_LI_293K.optical.provenance.citations

    assert [citation.doi for citation in citations] == ["10.1063/1.555624"]


def test_sin_fallback_alias_uses_luke_card() -> None:
    snapshot = resolve_material_snapshot("SiN", 1.55, {})

    assert snapshot.refractive_index == pytest.approx(1.996279731714)
    assert snapshot.source == "gsim"


def test_project_card_overrides_fallback_and_missing_card_uses_fallback() -> None:
    project_si = GSIM_MATERIAL_CARDS["Si-Li-293K"].model_copy(update={"name": "Si"})

    silicon = resolve_material_snapshot("Si", 1.55, {"Si": project_si})
    silica = resolve_material_snapshot("SiO2", 1.55, {"Si": project_si})

    assert silicon.source == "project"
    assert silicon.refractive_index == pytest.approx(3.475687046)
    assert silica.source == "gsim"
    assert silica.refractive_index == pytest.approx(1.444023622)


def test_invalid_project_card_does_not_silently_fallback() -> None:
    project_si = MaterialCard(name="Si", optical=None, rf=None, info={})

    with pytest.raises(MaterialModelError, match="no optical permittivity"):
        resolve_material_snapshot("Si", 1.55, {"Si": project_si})


def test_wavelength_validation_is_strict() -> None:
    with pytest.raises(WavelengthOutOfRangeError, match="valid from"):
        resolve_material_snapshot("Si-Salzberg", 1.0, {})
    with pytest.raises(WavelengthOutOfRangeError, match="finite positive"):
        resolve_material_snapshot("Si", 0, {})


def test_unknown_material_error_explains_how_to_supply_card() -> None:
    with pytest.raises(MaterialNotFoundError, match="Attach the card"):
        resolve_material_snapshot("GaAs", 1.55, {})
