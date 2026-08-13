"""Project-first material-card resolution."""

from collections.abc import Mapping

from pdk_schema import MaterialCard

from gsim.common.materials.cards import GSIM_MATERIAL_CARDS


class MaterialCardNotFoundError(LookupError):
    """Raised when neither the project nor gsim defines a material card."""


class MaterialCardResolver:
    """Resolve exact material names from a project, then gsim fallbacks."""

    def __init__(
        self,
        project_material_cards: Mapping[str, MaterialCard] | None = None,
        fallback_material_cards: Mapping[str, MaterialCard] | None = None,
    ) -> None:
        """Initialize project cards and optional fallback overrides."""
        self._project_material_cards = project_material_cards or {}
        self._fallback_material_cards = (
            GSIM_MATERIAL_CARDS
            if fallback_material_cards is None
            else fallback_material_cards
        )

    def resolve(self, material_name: str) -> MaterialCard:
        """Return a card using exact, case-sensitive project-first lookup."""
        if material_name in self._project_material_cards:
            return self._project_material_cards[material_name]
        if material_name in self._fallback_material_cards:
            return self._fallback_material_cards[material_name]

        available_names = sorted(
            self._project_material_cards.keys() | self._fallback_material_cards.keys()
        )
        available = ", ".join(available_names) or "none"
        raise MaterialCardNotFoundError(
            f"No MaterialCard named {material_name!r}. Available names: {available}."
        )


__all__ = ["MaterialCardNotFoundError", "MaterialCardResolver"]
