"""Material names that differ from a card only by letter case resolve to it."""

from types import SimpleNamespace

import pytest

from gsim.common.materials import (
    GSIM_MATERIAL_CARDS,
    AmbiguousMaterialNameError,
    MaterialResolutionError,
    canonical_material_name,
    find_material_card,
    get_material_card,
    registry,
    resolve_material_snapshot,
)

EMPTY_PDK = SimpleNamespace(material_cards={}, material_aliases={})


@pytest.mark.parametrize(
    ("declared", "canonical", "index"),
    [
        ("si", "Si", 3.477723756),
        ("sio2", "SiO2", 1.444023622),
        ("sin", "SiN", 1.996279731714),
        ("SIO2", "SiO2", 1.444023622),
        ("sIo2", "SiO2", 1.444023622),
    ],
)
def test_gpdk_style_names_use_the_builtin_cards(declared, canonical, index) -> None:
    assert canonical_material_name(declared, EMPTY_PDK) == canonical
    snapshot = resolve_material_snapshot(declared, 1.55, {})
    assert snapshot.refractive_index == pytest.approx(index, abs=1e-9)
    assert snapshot.source == "gsim"
    assert find_material_card(declared, {})[0] is GSIM_MATERIAL_CARDS[canonical]


def test_exact_names_are_unchanged() -> None:
    for name in GSIM_MATERIAL_CARDS:
        # The lowercase built-in air entry keeps its older mapping to ``Air``.
        expected = "Air" if name == "air" else name
        assert canonical_material_name(name, EMPTY_PDK) == expected
    assert canonical_material_name("Si-Salzberg", EMPTY_PDK) == "Si-Salzberg"


def test_air_spellings_are_one_card() -> None:
    """The built-in ``Air`` and ``air`` entries carry the same card."""
    assert canonical_material_name("AIR", EMPTY_PDK) == "Air"
    assert get_material_card("AIR", {}).name == "Air"
    assert canonical_material_name("air", EMPTY_PDK) == "Air"


def test_unknown_names_still_fail() -> None:
    assert canonical_material_name("GaAs", EMPTY_PDK) == "GaAs"
    with pytest.raises(MaterialResolutionError, match="No MaterialCard found"):
        resolve_material_snapshot("gaas", 1.55, {})


def test_exact_builtin_name_beats_a_project_card_differing_in_case() -> None:
    project = GSIM_MATERIAL_CARDS["SiN"].model_copy(update={"name": "SI"})
    pdk = SimpleNamespace(material_cards={"SI": project})
    assert canonical_material_name("Si", pdk) == "Si"
    assert canonical_material_name("si", pdk) == "SI"
    assert find_material_card("si", {"SI": project}) == (project, "project")
    assert find_material_card("Si", {"SI": project})[1] == "gsim"


def test_project_card_is_found_ignoring_case() -> None:
    project = GSIM_MATERIAL_CARDS["SiN"].model_copy(update={"name": "Core"})
    pdk = SimpleNamespace(material_cards={"Core": project})
    assert canonical_material_name("core", pdk) == "Core"
    assert find_material_card("CORE", {"Core": project}) == (project, "project")


def test_pdk_alias_is_found_ignoring_case() -> None:
    pdk = SimpleNamespace(material_cards={}, material_aliases={"Legacy_Si": "Si"})
    assert canonical_material_name("legacy_si", pdk) == "Si"
    assert canonical_material_name("LEGACY_SI", pdk) == "Si"


def test_project_cards_differing_only_by_case_are_ambiguous() -> None:
    first = GSIM_MATERIAL_CARDS["Si"].model_copy(update={"name": "Core"})
    second = GSIM_MATERIAL_CARDS["SiO2"].model_copy(update={"name": "core"})
    pdk = SimpleNamespace(material_cards={"Core": first, "core": second})
    # Exact names keep working.
    assert canonical_material_name("core", pdk) == "core"
    assert canonical_material_name("Core", pdk) == "Core"
    with pytest.raises(AmbiguousMaterialNameError, match=r"\['Core', 'core'\]"):
        canonical_material_name("CORE", pdk)
    with pytest.raises(AmbiguousMaterialNameError) as caught:
        find_material_card("CORE", {"Core": first, "core": second})
    assert "Core" in str(caught.value)
    assert "core" in str(caught.value)
    with pytest.raises(MaterialResolutionError, match="differ only by letter case"):
        resolve_material_snapshot("CORE", 1.55, {"Core": first, "core": second})


def test_builtin_cards_differing_only_by_case_are_ambiguous(monkeypatch) -> None:
    other = GSIM_MATERIAL_CARDS["SiN"].model_copy(update={"name": "Sio2"})
    monkeypatch.setitem(registry.GSIM_MATERIAL_CARDS, "Sio2", other)
    assert canonical_material_name("Sio2", EMPTY_PDK) == "Sio2"
    with pytest.raises(AmbiguousMaterialNameError, match=r"SiO2.*Sio2"):
        canonical_material_name("sio2", EMPTY_PDK)
