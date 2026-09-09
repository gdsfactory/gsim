"""Tests for built-in material-card metadata."""

import pytest
from pdk_schema import Provenance

from gsim.common.materials import GSIM_MATERIAL_CARDS


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
