"""Material cards supplied by gsim as project fallbacks."""

from pdk_schema import (
    Band,
    Citation,
    Coord,
    Index,
    MaterialCard,
    Provenance,
    Regime,
    Sellmeier,
    SellmeierTerm,
    TableData,
    TabulatedValue,
    Validity,
)

_RII_COMMIT = "6f3b772c3339d68a21538cb2562d2acb36731302"
_RII_BLOB_ROOT = (
    "https://github.com/polyanskiy/refractiveindex.info-database/blob/"
    f"{_RII_COMMIT}/database/data/main"
)

_LI_293K_WAVELENGTHS_UM = [
    1.20,
    1.22,
    1.24,
    1.26,
    1.28,
    1.30,
    1.32,
    1.34,
    1.36,
    1.38,
    1.40,
    1.45,
    1.50,
    1.55,
    1.60,
    1.65,
    1.70,
    1.80,
    1.90,
    2.00,
    2.25,
    2.50,
    2.75,
    3.00,
    4.00,
    5.00,
    6.00,
    7.00,
    8.00,
    9.00,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
]
_LI_293K_REFRACTIVE_INDICES = [
    3.5167,
    3.5133,
    3.5102,
    3.5072,
    3.5043,
    3.5016,
    3.4990,
    3.4965,
    3.4941,
    3.4918,
    3.4896,
    3.4845,
    3.4799,
    3.4757,
    3.4719,
    3.4684,
    3.4653,
    3.4597,
    3.4550,
    3.4510,
    3.4431,
    3.4375,
    3.4334,
    3.4302,
    3.4229,
    3.4195,
    3.4177,
    3.4165,
    3.4158,
    3.4153,
    3.4150,
    3.4147,
    3.4145,
    3.4144,
    3.4142,
]


def _validity(minimum_um: float, maximum_um: float) -> Validity:
    """Return strict wavelength validity metadata in micrometers."""
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


def _silicon_salzberg_card(name: str) -> MaterialCard:
    """Return the RII Salzberg crystalline-silicon card."""
    optical = Regime(
        temperature_ref=299.15,
        provenance=Provenance(
            source="literature",
            label="RII main/Si/nk/Salzberg",
            maturity="empirical",
            citations=[
                Citation(
                    role="data",
                    doi="10.1364/JOSA.47.000244",
                    journal="J. Opt. Soc. Am. 47, 244-246 (1957)",
                    authors="C. D. Salzberg and J. J. Villa",
                    url=None,
                ),
                Citation(
                    role="fit",
                    doi="10.1364/AO.23.004477",
                    journal="Appl. Opt. 23, 4477-4485 (1984)",
                    authors="B. Tatian",
                    url=None,
                ),
            ],
            comment="Crystalline silicon at 26 degC; purity is unknown.",
            url=("https://refractiveindex.info/?shelf=main&book=Si&page=Salzberg"),
            data_url=f"{_RII_BLOB_ROOT}/Si/nk/Salzberg.yml",
            info={"license": "CC0-1.0", "rii_commit": _RII_COMMIT},
        ),
        permittivity=Sellmeier(
            validity=_validity(1.357, 11.04),
            variation=None,
            conductivity=None,
            terms=(
                SellmeierTerm(b=10.6684293, c_um=0.301516485),
                SellmeierTerm(b=0.0030434748, c_um=1.13475115),
                SellmeierTerm(b=1.54133408, c_um=1104.0),
            ),
            offset=0.0,
        ),
        conductivity=None,
        permeability=None,
        perturbations=[],
        info={"material_form": "crystalline silicon", "purity": "unknown"},
    )
    return MaterialCard(name=name, optical=optical, rf=None, info={})


def _silicon_li_293k_card() -> MaterialCard:
    """Return the RII Li crystalline-silicon index table at 293 K."""
    refractive_index = TabulatedValue(
        unit="",
        data=TableData(
            dims=("wavelength",),
            coords={
                "wavelength": Coord(
                    values=_LI_293K_WAVELENGTHS_UM,
                    unit="um",
                )
            },
            values=_LI_293K_REFRACTIVE_INDICES,
            attrs={
                "interpolation_authoring_choice": "linear",
                "license": "CC0-1.0",
                "rii_commit": _RII_COMMIT,
            },
            interp="linear",
        ),
    )
    optical = Regime(
        temperature_ref=293.0,
        provenance=Provenance(
            source="literature",
            label="RII main/Si/nk/Li-293K",
            maturity="empirical",
            citations=[
                Citation(
                    role="data",
                    doi="10.1063/1.555624",
                    journal="J. Phys. Chem. Ref. Data 9, 561-658 (1980)",
                    authors="H. H. Li",
                    url=None,
                )
            ],
            comment=(
                "Crystalline silicon at 293 K. RII supplies samples without "
                "an interpolation rule; this card selects linear interpolation."
            ),
            url=("https://refractiveindex.info/?shelf=main&book=Si&page=Li-293K"),
            data_url=f"{_RII_BLOB_ROOT}/Si/nk/Li-293K.yml",
            info={
                "license": "CC0-1.0",
                "rii_commit": _RII_COMMIT,
                "rii_reference_year_note": (
                    "The RII YAML prints 1993; the DOI identifies the 1980 paper."
                ),
            },
        ),
        permittivity=Index(
            validity=_validity(1.2, 14.0),
            variation=None,
            conductivity=None,
            n=refractive_index,
            k=None,
        ),
        conductivity=None,
        permeability=None,
        perturbations=[],
        info={"material_form": "crystalline silicon"},
    )
    return MaterialCard(name="Si-Li-293K", optical=optical, rf=None, info={})


def _silicon_dioxide_malitson_card(name: str) -> MaterialCard:
    """Return the RII Malitson fused-silica card."""
    optical = Regime(
        temperature_ref=293.0,
        provenance=Provenance(
            source="literature",
            label="RII main/SiO2/nk/Malitson",
            maturity="empirical",
            citations=[
                Citation(
                    role="fit",
                    doi="10.1364/JOSA.55.001205",
                    journal="J. Opt. Soc. Am. 55, 1205-1208 (1965)",
                    authors="I. H. Malitson",
                    url=None,
                ),
                Citation(
                    role="data",
                    doi="10.1016/S0022-3093(97)00438-9",
                    journal="J. Non-Cryst. Solids 223, 158-163 (1998)",
                    authors="C. Z. Tan",
                    url=None,
                ),
            ],
            comment=(
                "Fused silica at 20 degC. This card reproduces the current "
                "RII formula coefficients without air-to-vacuum adjustment."
            ),
            url=("https://refractiveindex.info/?shelf=main&book=SiO2&page=Malitson"),
            data_url=f"{_RII_BLOB_ROOT}/SiO2/nk/Malitson.yml",
            info={
                "license": "CC0-1.0",
                "rii_commit": _RII_COMMIT,
                "index_normalization": "current RII formula as published",
            },
        ),
        permittivity=Sellmeier(
            validity=_validity(0.21, 6.7),
            variation=None,
            conductivity=None,
            terms=(
                SellmeierTerm(b=0.6961663, c_um=0.0684043),
                SellmeierTerm(b=0.4079426, c_um=0.1162414),
                SellmeierTerm(b=0.8974794, c_um=9.896161),
            ),
            offset=0.0,
        ),
        conductivity=None,
        permeability=None,
        perturbations=[],
        info={"material_form": "fused silica"},
    )
    return MaterialCard(name=name, optical=optical, rf=None, info={})


GSIM_MATERIAL_CARDS: dict[str, MaterialCard] = {
    "Si": _silicon_salzberg_card("Si"),
    "Si-Salzberg": _silicon_salzberg_card("Si-Salzberg"),
    "Si-Li-293K": _silicon_li_293k_card(),
    "SiO2": _silicon_dioxide_malitson_card("SiO2"),
    "SiO2-Malitson": _silicon_dioxide_malitson_card("SiO2-Malitson"),
}

__all__ = ["GSIM_MATERIAL_CARDS"]
