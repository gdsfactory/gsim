"""Malitson fused-silica model."""

from pdk_schema import Citation, Provenance, Sellmeier, SellmeierTerm

from gsim.common.materials._helpers import material_card, wavelength_validity

SIO2_MALITSON = material_card(
    name="SiO2-Malitson",
    temperature_ref=293.0,
    permittivity=Sellmeier(
        validity=wavelength_validity(0.21, 6.7),
        variation=None,
        conductivity=None,
        terms=(
            SellmeierTerm(b=0.6961663, c_um=0.0684043),
            SellmeierTerm(b=0.4079426, c_um=0.1162414),
            SellmeierTerm(b=0.8974794, c_um=9.896161),
        ),
        offset=0.0,
    ),
    provenance=Provenance(
        source="literature",
        label="Malitson 1965 fused silica",
        maturity="empirical",
        citations=[
            Citation(
                role="fit",
                doi="10.1364/JOSA.55.001205",
                journal=(
                    "Journal of the Optical Society of America 55, 1205-1209 (1965)"
                ),
                authors="I. H. Malitson",
                url="https://doi.org/10.1364/JOSA.55.001205",
            )
        ],
        comment="Optical-quality fused-silica refractive index measured at 20 °C.",
        url="https://doi.org/10.1364/JOSA.55.001205",
        data_url=None,
        info={},
    ),
    optical_info={"source_model": "three-term Sellmeier"},
    info={"composition": "optical-quality fused silica"},
)

__all__ = ["SIO2_MALITSON"]
