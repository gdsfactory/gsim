"""Project-first material-card lookup."""

from collections.abc import Mapping
from typing import Any, Literal

import gdsfactory as gf
from pdk_schema import MaterialCard

from gsim.common.materials.si_li_293k import SI_LI_293K
from gsim.common.materials.si_salzberg import SI_SALZBERG
from gsim.common.materials.sin_luke import SIN_LUKE
from gsim.common.materials.sio2_malitson import SIO2_MALITSON
from gsim.common.materials.sio2_palik import (
    SIO2_PALIK,
    SIO2_PALIK_LOSSLESS,
)

MaterialSource = Literal["project", "gsim"]

GSIM_MATERIAL_CARDS: dict[str, MaterialCard] = {
    "Si": SI_SALZBERG.model_copy(update={"name": "Si"}),
    "si": SI_SALZBERG.model_copy(update={"name": "si"}),
    "Si-Salzberg": SI_SALZBERG,
    "Si-Li-293K": SI_LI_293K,
    "SiN": SIN_LUKE.model_copy(update={"name": "SiN"}),
    "sin": SIN_LUKE.model_copy(update={"name": "sin"}),
    "SiN-Luke": SIN_LUKE,
    "SiO2": SIO2_PALIK_LOSSLESS.model_copy(update={"name": "SiO2"}),
    "sio2": SIO2_PALIK_LOSSLESS.model_copy(update={"name": "sio2"}),
    "SiO2-Malitson": SIO2_MALITSON,
    "SiO2-Palik": SIO2_PALIK,
    "SiO2-Palik-Lossless": SIO2_PALIK_LOSSLESS,
}


def get_project_material_cards(pdk: Any | None = None) -> Mapping[str, MaterialCard]:
    """Return cards attached to a PDK object or PDK module."""
    pdk_or_module = gf.get_active_pdk() if pdk is None else pdk
    pdk_object = getattr(pdk_or_module, "PDK", pdk_or_module)
    cards = getattr(pdk_object, "material_cards", None)
    if cards is None:
        cards = getattr(pdk_or_module, "MATERIAL_CARDS", None)
    return cards or {}


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
    if material_name in cards:
        return cards[material_name], "project"
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
    "find_material_card",
    "get_material_card",
    "get_project_material_cards",
]
