"""Helpers shared by the built-in material cards."""

from pdk_schema import (
    Band,
    Index,
    MaterialCard,
    Provenance,
    Regime,
    Sellmeier,
    Validity,
)


def wavelength_validity(minimum_um: float, maximum_um: float) -> Validity:
    """Return a strict wavelength validity range in micrometers."""
    return Validity(
        at=None,
        over={
            "wavelength": Band(
                min=minimum_um,
                max=maximum_um,
                unit="um",
                label=None,
            )
        },
        on_out_of_range="raise",
    )


def material_card(
    name: str,
    permittivity: Index | Sellmeier,
    temperature_ref: float | None,
    *,
    provenance: Provenance | None = None,
    optical_info: dict[str, object] | None = None,
    info: dict[str, object] | None = None,
) -> MaterialCard:
    """Build a compact optical material card."""
    resolved_provenance = provenance or Provenance(
        source="literature",
        label=name,
        maturity="empirical",
        citations=[],
        comment=None,
        url=None,
        data_url=None,
        info={},
    )
    return MaterialCard(
        name=name,
        optical=Regime(
            temperature_ref=temperature_ref,
            provenance=resolved_provenance,
            permittivity=permittivity,
            conductivity=None,
            permeability=None,
            perturbations=[],
            info=dict(optical_info or {}),
        ),
        rf=None,
        info=dict(info or {}),
    )
