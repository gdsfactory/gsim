"""Tests for anisotropic lithium-niobate material cards."""

from typing import Any

import pytest
from pdk_schema import Index, MaterialCard, TabulatedValue

from gsim.common.materials import (
    GSIM_MATERIAL_CARDS,
    LINBO3_MGO5_GAYER,
    LINBO3_ZELMON,
    MaterialModelError,
    resolve_material_snapshot,
)


def _tensor_model(card: MaterialCard) -> Index:
    assert card.optical is not None
    model = card.optical.permittivity
    assert isinstance(model, Index)
    assert isinstance(model.n, list)
    assert len(model.n) == 3
    return model


def _axis_table(model: Index, axis_index: int) -> TabulatedValue:
    assert isinstance(model.n, list)
    axis_value: Any = model.n[axis_index]
    assert isinstance(axis_value, TabulatedValue)
    return axis_value


def _index_at_wavelength(
    table: TabulatedValue,
    wavelength_um: float,
    temperature_kelvin: float | None = None,
) -> float:
    wavelengths_um = table.data.coords["wavelength"].values
    wavelength_index = min(
        range(len(wavelengths_um)),
        key=lambda index: abs(wavelengths_um[index] - wavelength_um),
    )
    assert wavelengths_um[wavelength_index] == pytest.approx(wavelength_um, abs=1e-12)
    if tuple(table.data.dims) == ("wavelength",):
        return table.data.values[wavelength_index]

    assert tuple(table.data.dims) == ("wavelength", "temperature")
    assert temperature_kelvin is not None
    temperatures_kelvin = table.data.coords["temperature"].values
    temperature_index = min(
        range(len(temperatures_kelvin)),
        key=lambda index: abs(temperatures_kelvin[index] - temperature_kelvin),
    )
    assert temperatures_kelvin[temperature_index] == pytest.approx(
        temperature_kelvin, abs=1e-12
    )
    flat_index = wavelength_index * len(temperatures_kelvin) + temperature_index
    return table.data.values[flat_index]


@pytest.mark.parametrize(
    ("registry_name", "card"),
    [
        ("LiNbO3-Zelmon", LINBO3_ZELMON),
        ("LiNbO3-MgO5-Gayer", LINBO3_MGO5_GAYER),
    ],
)
def test_anisotropic_cards_are_registered(
    registry_name: str, card: MaterialCard
) -> None:
    assert GSIM_MATERIAL_CARDS[registry_name] is card
    assert card.name == registry_name


def test_default_ln_alias_uses_zelmon() -> None:
    default_ln = GSIM_MATERIAL_CARDS["LN"]

    assert default_ln.name == "LN"
    assert default_ln is not LINBO3_ZELMON
    assert default_ln.optical == LINBO3_ZELMON.optical
    assert default_ln.info == LINBO3_ZELMON.info


@pytest.mark.parametrize(
    ("card", "minimum_um", "maximum_um", "temperature_kelvin"),
    [
        (LINBO3_ZELMON, 0.4, 5.0, 294.15),
        (LINBO3_MGO5_GAYER, 0.5, 1.62, 297.65),
    ],
)
def test_anisotropic_tensor_shape_and_validity(
    card: MaterialCard,
    minimum_um: float,
    maximum_um: float,
    temperature_kelvin: float,
) -> None:
    model = _tensor_model(card)
    ordinary_x = _axis_table(model, 0)
    ordinary_y = _axis_table(model, 1)
    extraordinary_z = _axis_table(model, 2)

    assert ordinary_x.data.values == ordinary_y.data.values
    assert ordinary_x.data.values != extraordinary_z.data.values
    assert ordinary_x.data.attrs["polarization"] == "ordinary"
    assert extraordinary_z.data.attrs["polarization"] == "extraordinary"
    assert model.validity is not None
    assert model.validity.over is not None
    wavelength_band = model.validity.over["wavelength"]
    assert wavelength_band.min == minimum_um
    assert wavelength_band.max == maximum_um
    assert card.optical is not None
    assert card.optical.temperature_ref == temperature_kelvin
    assert card.info["principal_axis_order"] == [
        "ordinary",
        "ordinary",
        "extraordinary",
    ]


@pytest.mark.parametrize(
    ("card", "expected_ordinary", "expected_extraordinary"),
    [
        (LINBO3_ZELMON, 2.211111009, 2.137559649),
        (LINBO3_MGO5_GAYER, 2.208812689, 2.130570331),
    ],
)
def test_anisotropic_indices_at_telecom_wavelength(
    card: MaterialCard,
    expected_ordinary: float,
    expected_extraordinary: float,
) -> None:
    model = _tensor_model(card)
    assert card.optical is not None
    ordinary = _index_at_wavelength(
        _axis_table(model, 0), 1.55, card.optical.temperature_ref
    )
    extraordinary = _index_at_wavelength(
        _axis_table(model, 2), 1.55, card.optical.temperature_ref
    )

    assert ordinary == pytest.approx(expected_ordinary, abs=1e-9)
    assert extraordinary == pytest.approx(expected_extraordinary, abs=1e-9)
    assert ordinary > extraordinary


def test_gayer_card_supports_temperature() -> None:
    model = _tensor_model(LINBO3_MGO5_GAYER)
    ordinary = _axis_table(model, 0)
    extraordinary = _axis_table(model, 2)

    assert ordinary.data.dims == ("wavelength", "temperature")
    assert extraordinary.data.dims == ("wavelength", "temperature")
    temperatures_kelvin = ordinary.data.coords["temperature"]
    assert temperatures_kelvin.unit == "K"
    assert temperatures_kelvin.values[0] == pytest.approx(293.15)
    assert temperatures_kelvin.values[-1] == pytest.approx(373.15)
    assert 297.65 in temperatures_kelvin.values

    assert model.validity is not None
    assert model.validity.over is not None
    temperature_band = model.validity.over["temperature"]
    assert temperature_band.min == pytest.approx(293.15)
    assert temperature_band.max == pytest.approx(373.15)
    assert temperature_band.unit == "K"

    assert _index_at_wavelength(ordinary, 1.55, 293.15) == pytest.approx(
        2.208315324, abs=1e-9
    )
    assert _index_at_wavelength(extraordinary, 1.55, 293.15) == pytest.approx(
        2.129385680, abs=1e-9
    )
    assert _index_at_wavelength(ordinary, 1.55, 373.15) == pytest.approx(
        2.218265977, abs=1e-9
    )
    assert _index_at_wavelength(extraordinary, 1.55, 373.15) == pytest.approx(
        2.153013134, abs=1e-9
    )


def test_cards_include_structured_source_citations() -> None:
    assert LINBO3_ZELMON.optical is not None
    assert LINBO3_MGO5_GAYER.optical is not None
    zelmon_dois = {
        citation.doi for citation in LINBO3_ZELMON.optical.provenance.citations
    }
    gayer_dois = {
        citation.doi for citation in LINBO3_MGO5_GAYER.optical.provenance.citations
    }

    assert zelmon_dois == {"10.1364/JOSAB.14.003319"}
    assert "10.1007/s00340-008-2998-2" in gayer_dois
    assert "10.1007/s00340-010-4203-7" in gayer_dois
    assert LINBO3_MGO5_GAYER.optical.provenance.info == {
        "coefficients_include_published_errata": True,
        "errata_dois": [
            "10.1007/s00340-008-3316-8",
            "10.1007/s00340-010-4203-7",
        ],
    }


@pytest.mark.parametrize("material_name", ["LN", "LiNbO3-Zelmon", "LiNbO3-MgO5-Gayer"])
def test_scalar_snapshot_resolver_rejects_anisotropic_cards(
    material_name: str,
) -> None:
    with pytest.raises(MaterialModelError, match="Tensor refractive indices"):
        resolve_material_snapshot(material_name, 1.55, {})
