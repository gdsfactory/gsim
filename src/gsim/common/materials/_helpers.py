"""Helpers shared by the built-in material cards."""

from typing import Literal

from pdk_schema import (
    Band,
    Citation,
    DispersionModel,
    MaterialCard,
    Provenance,
    Regime,
    Validity,
)

ProvenanceSource = Literal["foundry", "gdsfactory", "user", "literature"]


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
    permittivity: DispersionModel,
    temperature_ref: float | None,
    *,
    citations: tuple[Citation, ...] = (),
    provenance_comment: str | None = None,
    provenance_source: ProvenanceSource = "literature",
    provenance_url: str | None = None,
    provenance_info: dict[str, object] | None = None,
) -> MaterialCard:
    """Build a compact optical material card."""
    provenance = Provenance(
        source=provenance_source,
        label=name,
        maturity="empirical",
        citations=list(citations),
        comment=provenance_comment,
        url=provenance_url,
        data_url=None,
        info={} if provenance_info is None else provenance_info,
    )
    return MaterialCard(
        name=name,
        optical=Regime(
            temperature_ref=temperature_ref,
            provenance=provenance,
            permittivity=permittivity,
            conductivity=None,
            permeability=None,
            perturbations=[],
            info={},
        ),
        rf=None,
        info={},
    )
