"""Arosa and de la Fuente fused-silica model."""

from pdk_schema import Citation, Sellmeier, SellmeierTerm

from gsim.common.materials._helpers import material_card, wavelength_validity

AROSA_FIT_CITATION = Citation(
    role="fit",
    doi="10.1364/OL.395510",
    journal="Opt. Lett. 45, 4268-4271 (2020)",
    authors="Y. Arosa; R. de la Fuente",
    url="https://doi.org/10.1364/OL.395510",
)

SIO2_AROSA = material_card(
    name="SiO2-Arosa",
    temperature_ref=None,
    permittivity=Sellmeier(
        validity=wavelength_validity(0.26, 1.7),
        variation=None,
        conductivity=None,
        terms=(
            SellmeierTerm(b=0.9310, c_um=0.079),
            SellmeierTerm(b=0.1735, c_um=0.130),
            SellmeierTerm(b=2.1121, c_um=14.918),
        ),
        offset=0.0,
    ),
    citations=(AROSA_FIT_CITATION,),
    provenance_comment=(
        "Lossless three-term Sellmeier fit to measured fused-silica phase and "
        "group indices."
    ),
    provenance_url="https://doi.org/10.1364/OL.395510",
    provenance_info={"coefficient_source_table": 1},
)

__all__ = ["SIO2_AROSA"]
