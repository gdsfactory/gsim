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


class AmbiguousMaterialNameError(ValueError):
    """A material name matches several cards that differ only by letter case."""


def _follow_aliases(
    material_name: str,
    cards: Mapping[str, MaterialCard],
    aliases: Mapping[str, str],
) -> str:
    """Follow declared aliases until an exact project card or a plain name."""
    seen = set()
    while material_name not in cards and material_name in aliases:
        if material_name in seen:
            raise ValueError(f"Material alias cycle at {material_name!r}")
        seen.add(material_name)
        material_name = aliases[material_name]
    return material_name


def _same_card(first: MaterialCard, second: MaterialCard) -> bool:
    """Return whether two cards differ at most in their name."""
    try:
        first_data = first.model_dump(exclude={"name"})
        second_data = second.model_dump(exclude={"name"})
    except (AttributeError, TypeError, ValueError):
        return False  # Not comparable, so treat the cards as distinct.
    return first_data == second_data


def _pick_unique(
    material_name: str, matches: Mapping[str, MaterialCard | None]
) -> str | None:
    """Pick the single case-insensitive match, or raise when it is ambiguous.

    *matches* maps each candidate name to its card (``None`` if unresolved).
    Several candidates are accepted only when they carry the same card, for
    example the built-in ``Air`` and ``air`` entries.
    """
    if len(matches) <= 1:
        return next(iter(matches), None)
    first, *others = matches.values()
    if first is not None and all(
        card is not None and _same_card(first, card) for card in others
    ):
        return next(iter(matches))
    raise AmbiguousMaterialNameError(
        f"Material name {material_name!r} matches several cards that differ only "
        f"by letter case: {sorted(matches)}. Use the exact card name."
    )


def _case_insensitive_name(
    material_name: str,
    cards: Mapping[str, MaterialCard],
    aliases: Mapping[str, str],
) -> str | None:
    """Return the project or built-in name equal to *material_name* ignoring case.

    Project cards and PDK aliases are searched before the built-in cards.
    """
    folded = material_name.casefold()
    project: dict[str, MaterialCard | None] = {}
    for key in (*cards, *aliases):
        if key.casefold() == folded:
            target = _follow_aliases(key, cards, aliases)
            project[target] = cards.get(target)
    match = _pick_unique(material_name, project)
    if match is not None:
        return match
    builtin = {
        name: card
        for name, card in GSIM_MATERIAL_CARDS.items()
        if name.casefold() == folded
    }
    return _pick_unique(material_name, builtin)


def canonical_material_name(material_name: str, pdk: Any | None = None) -> str:
    """Resolve declared PDK aliases, the built-in air alias and letter case.

    An exact project card takes precedence over aliases, and any exact name
    takes precedence over a case-insensitive one. Distinct project names such
    as ``Air`` and ``air`` therefore remain distinct. A name with no exact
    match is resolved to the unique project card, PDK alias or built-in card
    that equals it ignoring letter case (``si`` -> ``Si``). Several such
    matches that carry different cards raise ``AmbiguousMaterialNameError``.
    """
    module, cards = _project_materials(pdk)
    aliases = get_project_material_aliases(module)
    material_name = _follow_aliases(material_name, cards, aliases)
    if material_name == "air" and material_name not in cards:
        return "Air"
    if material_name in cards or material_name in GSIM_MATERIAL_CARDS:
        return material_name
    match = _case_insensitive_name(material_name, cards, aliases)
    return material_name if match is None else match


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
    """Find a project card first, then a built-in fallback card.

    An exact name wins. Otherwise the unique card whose name equals it
    ignoring letter case is used, so ``si`` finds ``Si``. Several cards that
    differ only by case raise ``AmbiguousMaterialNameError``.
    """
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
    match = _case_insensitive_name(material_name, cards, {})
    if match is not None:
        if match in cards:
            return cards[match], "project"
        return GSIM_MATERIAL_CARDS[match], "gsim"
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
    "AmbiguousMaterialNameError",
    "MaterialSource",
    "canonical_material_name",
    "find_material_card",
    "get_material_card",
    "get_project_material_aliases",
    "get_project_material_cards",
]
