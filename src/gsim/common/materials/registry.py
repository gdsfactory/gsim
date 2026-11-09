"""Project-first material-card lookup."""

from collections.abc import Mapping
from typing import Any, Literal

import gdsfactory as gf
from pdk_schema import MaterialCard

from gsim.common.materials.air import AIR
from gsim.common.materials.lithium_niobate import (
    LINBO3_MGO5_GAYER,
    LINBO3_ZELMON,
)
from gsim.common.materials.si_li_293k import SI_LI_293K
from gsim.common.materials.si_salzberg import SI_SALZBERG
from gsim.common.materials.sin_luke import SIN_LUKE
from gsim.common.materials.sio2_malitson import SIO2_MALITSON

MaterialSource = Literal["project", "gsim"]


def _project_materials(pdk: Any | None) -> tuple[Any, Mapping[str, MaterialCard]]:
    """Find native PDK cards, retaining the legacy module fallback."""
    module = gf.get_active_pdk() if pdk is None else pdk
    pdk_object = getattr(module, "PDK", module)
    cards = getattr(pdk_object, "material_cards", None)
    return module, cards or getattr(module, "MATERIAL_CARDS", None) or {}


def canonical_material_name(material_name: str, pdk: Any | None = None) -> str:
    """Resolve declared PDK aliases and the built-in air alias, without case folding.

    An exact project card takes precedence over aliases. Distinct
    project names such as ``Air`` and ``air`` therefore remain distinct.
    """
    module, cards = _project_materials(pdk)
    aliases = get_project_material_aliases(module)
    seen = set()
    while material_name not in cards and material_name in aliases:
        if material_name in seen:
            raise ValueError(f"Material alias cycle at {material_name!r}")
        seen.add(material_name)
        material_name = aliases[material_name]
    if material_name == "air" and material_name not in cards:
        return "Air"
    return material_name


def get_project_material_aliases(pdk: Any | None = None) -> Mapping[str, str]:
    """Return explicit compatibility names declared by a PDK or its module."""
    module = gf.get_active_pdk() if pdk is None else pdk
    pdk_object = getattr(module, "PDK", module)
    return (
        getattr(pdk_object, "material_aliases", None)
        or getattr(module, "MATERIAL_ALIASES", None)
        or {}
    )


GSIM_MATERIAL_CARDS: dict[str, MaterialCard] = {
    "Air": AIR,
    "air": AIR.model_copy(update={"name": "air"}),
    "Si": SI_SALZBERG.model_copy(update={"name": "Si"}),
    "Si-Salzberg": SI_SALZBERG,
    "Si-Li-293K": SI_LI_293K,
    "SiN": SIN_LUKE.model_copy(update={"name": "SiN"}),
    "SiN-Luke": SIN_LUKE,
    "SiO2": SIO2_MALITSON.model_copy(update={"name": "SiO2"}),
    "SiO2-Malitson": SIO2_MALITSON,
    "LN": LINBO3_ZELMON.model_copy(update={"name": "LN"}),
    "LiNbO3-Zelmon": LINBO3_ZELMON,
    "LiNbO3-MgO5-Gayer": LINBO3_MGO5_GAYER,
}


def get_project_material_cards(
    pdk: Any | None = None,
    *,
    include_aliases: bool = True,
) -> Mapping[str, MaterialCard]:
    """Return project cards, optionally including expanded compatibility names."""
    module, cards = _project_materials(pdk)
    if not include_aliases:
        return cards
    aliases = get_project_material_aliases(module)
    expanded = dict(cards)
    for alias in aliases:
        target = canonical_material_name(alias, module)
        # Missing project targets can be supplied by a built-in fallback or a
        # later explicit override. Validate them when the material is used.
        if target in cards:
            expanded[alias] = cards[target]
    return expanded


def find_material_card(
    material_name: str,
    project_material_cards: Mapping[str, MaterialCard] | None = None,
) -> tuple[MaterialCard, MaterialSource]:
    """Find a project card first, then a built-in fallback card."""
    cards = (
        get_project_material_cards()
        if project_material_cards is None
        else project_material_cards
    )
    if project_material_cards is None:
        material_name = canonical_material_name(material_name)
    if material_name in cards:
        return cards[material_name], "project"
    if material_name == "air" and "Air" in cards:
        return cards["Air"], "project"
    if material_name in GSIM_MATERIAL_CARDS:
        return GSIM_MATERIAL_CARDS[material_name], "gsim"
    available = sorted(set(cards) | set(GSIM_MATERIAL_CARDS))
    raise KeyError(
        f"No MaterialCard found for {material_name!r}. Available cards: {available}. "
        "Attach the card to the active PDK as 'material_cards' or add it to "
        "GSIM_MATERIAL_CARDS."
    )


def get_material_card(
    material_name: str,
    project_material_cards: Mapping[str, MaterialCard] | None = None,
) -> MaterialCard:
    """Return a project card when present, otherwise a built-in card."""
    return find_material_card(material_name, project_material_cards)[0]


__all__ = [
    "GSIM_MATERIAL_CARDS",
    "MaterialSource",
    "canonical_material_name",
    "find_material_card",
    "get_material_card",
    "get_project_material_aliases",
    "get_project_material_cards",
]
