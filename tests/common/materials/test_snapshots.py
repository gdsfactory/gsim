from __future__ import annotations

import pytest
from pdk_schema import Coord, Index, MaterialCard, TableData, TabulatedValue

import gsim.common.materials.snapshots as material_snapshots
from gsim.common.materials import (
    GSIM_MATERIAL_CARDS,
    SI_LI_293K,
    MaterialModelError,
    MaterialNotFoundError,
    WavelengthOutOfRangeError,
    resolve_material_snapshot,
)


def _one_dimensional_index_table() -> TabulatedValue:
    return TabulatedValue(
        unit="",
        data=TableData(
            dims=("wavelength",),
            coords={"wavelength": Coord(values=[1.5, 1.6], unit="um")},
            values=[2.0, 2.1],
            attrs={},
            interp="linear",
        ),
    )


def _temperature_index_table() -> TabulatedValue:
    return TabulatedValue(
        unit="",
        data=TableData(
            dims=("wavelength", "temperature"),
            coords={
                "wavelength": Coord(values=[1.5, 1.6], unit="um"),
                "temperature": Coord(values=[293.0, 303.0], unit="K"),
            },
            values=[2.0, 2.1, 2.2, 2.3],
            attrs={},
            interp="linear",
        ),
    )


def _with_table_updates(
    value: TabulatedValue,
    **updates: object,
) -> TabulatedValue:
    return value.model_copy(update={"data": value.data.model_copy(update=updates)})


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


def test_temperature_unit_conversion() -> None:
    assert material_snapshots._temperature_to_kelvin(293.0, "K") == 293.0
    assert material_snapshots._temperature_to_kelvin(20.0, "degC") == pytest.approx(
        293.15
    )

    with pytest.raises(MaterialModelError, match="Unsupported temperature unit"):
        material_snapshots._temperature_to_kelvin(68.0, "degF")


def test_wavelength_table_values_supports_one_dimensional_tables() -> None:
    wavelengths_um, refractive_indices = material_snapshots._wavelength_table_values(
        _one_dimensional_index_table(),
        temperature_ref_kelvin=None,
    )

    assert wavelengths_um == [1.5, 1.6]
    assert refractive_indices == [2.0, 2.1]


def test_wavelength_table_values_requires_wavelength_coordinates() -> None:
    value = _with_table_updates(_one_dimensional_index_table(), coords={})

    with pytest.raises(MaterialModelError, match="missing wavelength coordinates"):
        material_snapshots._wavelength_table_values(value, 293.0)


def test_wavelength_table_values_rejects_unsupported_dimensions() -> None:
    value = _with_table_updates(
        _temperature_index_table(),
        dims=("temperature", "wavelength"),
    )

    with pytest.raises(MaterialModelError, match="must have 'wavelength'"):
        material_snapshots._wavelength_table_values(value, 293.0)


def test_temperature_table_requires_reference_temperature() -> None:
    with pytest.raises(MaterialModelError, match="require a reference temperature"):
        material_snapshots._wavelength_table_values(
            _temperature_index_table(),
            temperature_ref_kelvin=None,
        )


def test_temperature_table_requires_temperature_coordinates() -> None:
    value = _with_table_updates(
        _temperature_index_table(),
        coords={"wavelength": Coord(values=[1.5, 1.6], unit="um")},
    )

    with pytest.raises(MaterialModelError, match="missing temperature coordinates"):
        material_snapshots._wavelength_table_values(value, 293.0)


def test_temperature_table_rejects_invalid_shape() -> None:
    value = _with_table_updates(
        _temperature_index_table(),
        values=[2.0, 2.1, 2.2],
    )

    with pytest.raises(MaterialModelError, match="invalid shape"):
        material_snapshots._wavelength_table_values(value, 293.0)


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


def test_temperature_table_resolves_extinction_at_reference_temperature() -> None:
    base_card = GSIM_MATERIAL_CARDS["Si-Li-293K"]
    assert base_card.optical is not None
    base_model = base_card.optical.permittivity
    assert isinstance(base_model, Index)
    extinction_table = _with_table_updates(
        _temperature_index_table(),
        values=[0.01, 0.02, 0.03, 0.04],
    )
    lossy_model = base_model.model_copy(update={"k": extinction_table})
    lossy_optical = base_card.optical.model_copy(update={"permittivity": lossy_model})
    lossy_card = base_card.model_copy(
        update={"name": "Lossy-Si", "optical": lossy_optical}
    )

    snapshot = resolve_material_snapshot("Lossy-Si", 1.55, {"Lossy-Si": lossy_card})

    assert snapshot.extinction_coefficient == pytest.approx(0.02)


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
