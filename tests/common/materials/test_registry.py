"""Tests for built-in material-card metadata."""

import pytest
from pdk_schema import Provenance

from gsim.common.materials import GSIM_MATERIAL_CARDS, resolve_material_snapshot


def _provenance(material_name: str) -> Provenance:
    """Return the required optical provenance for a built-in card."""
    optical_regime = GSIM_MATERIAL_CARDS[material_name].optical
    assert optical_regime is not None
    assert optical_regime.provenance is not None
    return optical_regime.provenance


@pytest.mark.parametrize(
    ("material_name", "primary_doi", "expected_dois"),
    [
        (
            "Si-Salzberg",
            "10.1364/JOSA.47.000244",
            {"10.1364/JOSA.47.000244"},
        ),
        ("Si-Li-293K", "10.1063/1.555624", {"10.1063/1.555624"}),
        ("SiN-Luke", "10.1364/OL.40.004823", {"10.1364/OL.40.004823"}),
        (
            "SiO2-Malitson",
            "10.1364/JOSA.55.001205",
            {"10.1364/JOSA.55.001205"},
        ),
        (
            "LiNbO3-Zelmon",
            "10.1364/JOSAB.14.003319",
            {"10.1364/JOSAB.14.003319"},
        ),
        (
            "LiNbO3-MgO5-Gayer",
            "10.1007/s00340-008-2998-2",
            {
                "10.1007/s00340-008-2998-2",
                "10.1007/s00340-008-3316-8",
                "10.1007/s00340-010-4203-7",
            },
        ),
    ],
)
def test_builtin_model_cards_link_primary_papers(
    material_name: str,
    primary_doi: str,
    expected_dois: set[str],
) -> None:
    """Each named model should carry DOI links to its primary paper and errata."""
    provenance = _provenance(material_name)

    assert provenance.source == "literature"
    assert provenance.url == f"https://doi.org/{primary_doi}"
    assert {citation.doi for citation in provenance.citations} == expected_dois
    for citation in provenance.citations:
        assert citation.doi is not None
        assert citation.url == f"https://doi.org/{citation.doi}"


@pytest.mark.parametrize(
    ("alias", "named_model"),
    [
        ("Si", "Si-Salzberg"),
        ("SiN", "SiN-Luke"),
        ("SiO2", "SiO2-Malitson"),
        ("LN", "LiNbO3-Zelmon"),
    ],
)
def test_default_aliases_share_model_provenance(alias: str, named_model: str) -> None:
    """Default aliases should retain the named model's paper links."""
    assert _provenance(alias) == _provenance(named_model)


@pytest.mark.parametrize("name", ["Air", "air"])
@pytest.mark.parametrize("wavelength", [0.1, 1.55, 20])
def test_ideal_air_has_unit_index_and_no_absorption(name, wavelength) -> None:
    """The ideal air model has relative permittivity one throughout its band."""
    snapshot = resolve_material_snapshot(name, wavelength, {})

    assert snapshot.refractive_index == 1
    assert snapshot.extinction_coefficient == 0


def test_native_material_aliases_and_module_fallback() -> None:
    from types import SimpleNamespace

    from gsim.common.materials import (
        canonical_material_name,
        get_project_material_cards,
    )

    card = GSIM_MATERIAL_CARDS["Air"]
    pdk = SimpleNamespace(
        material_cards={"Air": card}, material_aliases={"air": "Air", "legacy": "air"}
    )
    assert get_project_material_cards(pdk)["legacy"] is card
    assert canonical_material_name("legacy", pdk) == "Air"
    module = SimpleNamespace(
        PDK=SimpleNamespace(material_cards={}), MATERIAL_CARDS={"Air": card}
    )
    assert get_project_material_cards(module) == {"Air": card}


def test_distinct_project_material_case_is_preserved() -> None:
    from types import SimpleNamespace

    from gsim.common.materials import canonical_material_name, get_material_card

    air = GSIM_MATERIAL_CARDS["Air"]
    other = air.model_copy(update={"name": "air", "info": {"distinct": True}})
    pdk = SimpleNamespace(material_cards={"Air": air, "air": other})
    assert canonical_material_name("air", pdk) == "air"
    assert get_material_card("air", pdk.material_cards) is other
    assert get_material_card("air", {"Air": air}) is air


def test_alias_can_use_a_builtin_fallback(monkeypatch) -> None:
    from types import SimpleNamespace

    import gdsfactory as gf

    from gsim.common.materials import find_material_card, get_project_material_cards

    pdk = SimpleNamespace(material_cards={}, material_aliases={"legacy_si": "Si"})
    monkeypatch.setattr(gf, "get_active_pdk", lambda: pdk)
    assert get_project_material_cards() == {}
    card, source = find_material_card("legacy_si")
    assert card is GSIM_MATERIAL_CARDS["Si"]
    assert source == "gsim"


def test_exact_project_card_overrides_conflicting_alias() -> None:
    from types import SimpleNamespace

    from gsim.common.materials import (
        canonical_material_name,
        get_project_material_cards,
    )

    air = GSIM_MATERIAL_CARDS["Air"]
    custom = air.model_copy(update={"name": "air", "info": {"distinct": True}})
    pdk = SimpleNamespace(
        material_cards={"Air": air, "air": custom},
        material_aliases={"air": "Air", "ambient": "air"},
    )
    assert canonical_material_name("air", pdk) == "air"
    assert canonical_material_name("ambient", pdk) == "air"
    cards = get_project_material_cards(pdk)
    assert cards["air"] is custom
    assert cards["ambient"] is custom
