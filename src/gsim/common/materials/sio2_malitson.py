"""Malitson fused-silica model."""

from pdk_schema import Citation, Lorentz, LorentzTerm, Sellmeier, SellmeierTerm

from gsim.common.materials._helpers import material_card, wavelength_validity

MALITSON_FIT_CITATION = Citation(
    role="fit",
    doi="10.1364/JOSA.55.001205",
    journal="J. Opt. Soc. Am. 55, 1205-1209 (1965)",
    authors="I. H. Malitson",
    url="https://doi.org/10.1364/JOSA.55.001205",
)

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
    citations=(MALITSON_FIT_CITATION,),
    provenance_comment=(
        "Three-term lossless Sellmeier fit for optical-quality fused silica."
    ),
    provenance_url="https://doi.org/10.1364/JOSA.55.001205",
    provenance_info={"coefficient_source_page": 1205},
)

SIO2_MALITSON_2POLE = material_card(
    name="SiO2-Malitson-2Pole",
    temperature_ref=293.0,
    permittivity=Lorentz(
        validity=wavelength_validity(0.4, 2.0),
        variation=None,
        eps_inf=1.2648942846816222,
        terms=(
            LorentzTerm(
                delta_eps=0.8392240453423187,
                omega_0=1.8433977875628328e16,
                gamma=0.0,
            ),
            LorentzTerm(
                delta_eps=0.9045823938713766,
                omega_0=1.8964034770858616e14,
                gamma=0.0,
            ),
        ),
    ),
    citations=(MALITSON_FIT_CITATION,),
    provenance_comment=(
        "Lossless two-pole reduction of the Malitson model for time-domain "
        "simulation from 400 to 2000 nm."
    ),
    provenance_source="gdsfactory",
    provenance_url="https://doi.org/10.1364/JOSA.55.001205",
    provenance_info={
        "derived_from": "SiO2-Malitson",
        "fit_band_um": [0.4, 2.0],
        "fit_grid": "801 geometrically spaced wavelengths",
        "fit_objective": "least squares in real relative permittivity",
        "fit_constraints": (
            "eps_inf >= 1; positive strengths; UV pole below 0.4 um; "
            "IR pole above 2.0 um"
        ),
        "fit_max_abs_delta_n": 5.418048441008239e-7,
        "fit_rms_delta_n": 1.1324175621670019e-7,
        "pole_wavelengths_um": [0.1021836729987205, 9.932757401412259],
        "fit_coefficient_origin": "derived in gsim; not published",
        "reference_coefficient_source_page": 1205,
    },
)

__all__ = ["SIO2_MALITSON", "SIO2_MALITSON_2POLE"]
