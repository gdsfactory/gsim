"""Luke et al. silicon-nitride model."""

from pdk_schema import Citation, Provenance, Sellmeier, SellmeierTerm

from gsim.common.materials._helpers import material_card, wavelength_validity

SIN_LUKE = material_card(
    name="SiN-Luke",
    temperature_ref=None,
    permittivity=Sellmeier(
        validity=wavelength_validity(0.310, 5.504),
        variation=None,
        conductivity=None,
        terms=(
            SellmeierTerm(b=3.0249, c_um=0.1353406),
            SellmeierTerm(b=40314.0, c_um=1239.842),
        ),
        offset=0.0,
    ),
    provenance=Provenance(
        source="literature",
        label="Luke et al. 2015 silicon nitride",
        maturity="empirical",
        citations=[
            Citation(
                role="fit",
                doi="10.1364/OL.40.004823",
                journal="Optics Letters 40, 4823-4826 (2015)",
                authors=(
                    "K. Luke, Y. Okawachi, M. R. E. Lamont, A. L. Gaeta, and M. Lipson"
                ),
                url="https://doi.org/10.1364/OL.40.004823",
            )
        ],
        comment=(
            "Silicon-nitride Sellmeier equation derived from film measurements "
            "from the ultraviolet to the infrared."
        ),
        url="https://doi.org/10.1364/OL.40.004823",
        data_url=None,
        info={},
    ),
    optical_info={"source_model": "two-term Sellmeier"},
    info={"composition": "Si3N4 film"},
)

__all__ = ["SIN_LUKE"]
