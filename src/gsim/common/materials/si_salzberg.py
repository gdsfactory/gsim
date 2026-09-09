"""Salzberg and Villa crystalline silicon model."""

from pdk_schema import Citation, Provenance, Sellmeier, SellmeierTerm

from gsim.common.materials._helpers import material_card, wavelength_validity

SI_SALZBERG = material_card(
    name="Si-Salzberg",
    temperature_ref=299.15,
    permittivity=Sellmeier(
        validity=wavelength_validity(1.357, 11.04),
        variation=None,
        conductivity=None,
        terms=(
            SellmeierTerm(b=10.6684293, c_um=0.301516485),
            SellmeierTerm(b=0.0030434748, c_um=1.13475115),
            SellmeierTerm(b=1.54133408, c_um=1104.0),
        ),
        offset=0.0,
    ),
    provenance=Provenance(
        source="literature",
        label="Salzberg and Villa 1957 crystalline silicon",
        maturity="empirical",
        citations=[
            Citation(
                role="fit",
                doi="10.1364/JOSA.47.000244",
                journal="Journal of the Optical Society of America 47, 244-246 (1957)",
                authors="C. D. Salzberg and J. J. Villa",
                url="https://doi.org/10.1364/JOSA.47.000244",
            )
        ],
        comment="Single-crystal silicon refractive index measured at 26 °C.",
        url="https://doi.org/10.1364/JOSA.47.000244",
        data_url=None,
        info={},
    ),
    optical_info={"source_model": "three-term Sellmeier"},
    info={"composition": "single-crystal silicon"},
)

__all__ = ["SI_SALZBERG"]
