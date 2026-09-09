"""Anisotropic lithium-niobate optical material cards."""

from collections.abc import Callable
from math import sqrt

from pdk_schema import (
    Band,
    Citation,
    Coord,
    Index,
    Provenance,
    TableData,
    TabulatedValue,
    Validity,
)

from gsim.common.materials._helpers import material_card, wavelength_validity

_SAMPLE_SPACING_UM = 0.001
_ZELMON_VALID_WAVELENGTH_UM = (0.4, 5.0)
_GAYER_VALID_WAVELENGTH_UM = (0.5, 1.62)
_GAYER_VALID_TEMPERATURE_C = (20.0, 100.0)
_GAYER_REFERENCE_TEMPERATURE_C = 24.5
_GAYER_TEMPERATURE_SPACING_C = 1.0

_ZELMON_ORDINARY_TERMS = (
    (2.6734, 0.01764),
    (1.2290, 0.05914),
    (12.614, 474.60),
)
_ZELMON_EXTRAORDINARY_TERMS = (
    (2.9804, 0.02047),
    (0.5981, 0.0666),
    (8.9543, 416.08),
)

_GAYER_ORDINARY_DISPERSION_COEFFICIENTS = (
    5.653,
    0.1185,
    0.2091,
    89.61,
    10.85,
    1.97e-2,
)
_GAYER_ORDINARY_TEMPERATURE_COEFFICIENTS = (
    7.941e-7,
    3.134e-8,
    -4.641e-9,
    -2.188e-6,
)
_GAYER_EXTRAORDINARY_DISPERSION_COEFFICIENTS = (
    5.756,
    0.0983,
    0.2020,
    189.32,
    12.52,
    1.32e-2,
)
_GAYER_EXTRAORDINARY_TEMPERATURE_COEFFICIENTS = (
    2.860e-6,
    4.700e-8,
    6.113e-8,
    1.516e-4,
)


def _sample_wavelengths(minimum_um: float, maximum_um: float) -> list[float]:
    """Return wavelengths at approximately 1 nm spacing, including both bounds."""
    interval_count = round((maximum_um - minimum_um) / _SAMPLE_SPACING_UM)
    wavelength_span = maximum_um - minimum_um
    return [
        minimum_um + wavelength_span * index / interval_count
        for index in range(interval_count + 1)
    ]


def _sample_gayer_temperatures() -> list[float]:
    """Return temperatures in Celsius, including Gayer's reference point."""
    minimum_c, maximum_c = _GAYER_VALID_TEMPERATURE_C
    interval_count = round((maximum_c - minimum_c) / _GAYER_TEMPERATURE_SPACING_C)
    temperatures_c = [
        minimum_c + _GAYER_TEMPERATURE_SPACING_C * index
        for index in range(interval_count + 1)
    ]
    temperatures_c.append(_GAYER_REFERENCE_TEMPERATURE_C)
    return sorted(set(temperatures_c))


def _zelmon_index(
    wavelength_um: float,
    terms: tuple[tuple[float, float], ...],
) -> float:
    """Evaluate the three-oscillator Zelmon Sellmeier equation."""
    wavelength_squared = wavelength_um**2
    index_squared = 1.0 + sum(
        strength * wavelength_squared / (wavelength_squared - pole_squared)
        for strength, pole_squared in terms
    )
    return sqrt(index_squared)


def _gayer_index(
    wavelength_um: float,
    temperature_c: float,
    dispersion_coefficients: tuple[float, float, float, float, float, float],
    temperature_coefficients: tuple[float, float, float, float],
) -> float:
    """Evaluate Gayer's temperature-dependent modified Sellmeier equation."""
    a1, a2, a3, a4, a5, a6 = dispersion_coefficients
    b1, b2, b3, b4 = temperature_coefficients
    temperature_factor = (temperature_c - _GAYER_REFERENCE_TEMPERATURE_C) * (
        temperature_c + 570.82
    )
    wavelength_squared = wavelength_um**2
    index_squared = (
        a1
        + b1 * temperature_factor
        + (a2 + b2 * temperature_factor)
        / (wavelength_squared - (a3 + b3 * temperature_factor) ** 2)
        + (a4 + b4 * temperature_factor) / (wavelength_squared - a5**2)
        - a6 * wavelength_squared
    )
    return sqrt(index_squared)


def _tabulated_index(
    wavelengths_um: list[float],
    index_function: Callable[[float], float],
    polarization: str,
) -> TabulatedValue:
    """Sample one principal refractive index into a schema table."""
    return TabulatedValue(
        unit="",
        data=TableData(
            dims=("wavelength",),
            coords={"wavelength": Coord(values=wavelengths_um, unit="um")},
            values=[index_function(wavelength_um) for wavelength_um in wavelengths_um],
            attrs={"polarization": polarization},
            interp="linear",
        ),
    )


def _anisotropic_index(
    valid_wavelength_um: tuple[float, float],
    ordinary_index: Callable[[float], float],
    extraordinary_index: Callable[[float], float],
) -> Index:
    """Build a diagonal [ordinary, ordinary, extraordinary] index tensor."""
    wavelengths_um = _sample_wavelengths(*valid_wavelength_um)
    ordinary = _tabulated_index(wavelengths_um, ordinary_index, "ordinary")
    extraordinary = _tabulated_index(
        wavelengths_um, extraordinary_index, "extraordinary"
    )
    return Index(
        validity=wavelength_validity(*valid_wavelength_um),
        variation=None,
        conductivity=None,
        n=[ordinary, ordinary.model_copy(deep=True), extraordinary],
        k=None,
    )


def _temperature_tabulated_index(
    wavelengths_um: list[float],
    temperatures_c: list[float],
    index_function: Callable[[float, float], float],
    polarization: str,
) -> TabulatedValue:
    """Sample one principal index over wavelength and temperature."""
    return TabulatedValue(
        unit="",
        data=TableData(
            dims=("wavelength", "temperature"),
            coords={
                "wavelength": Coord(values=wavelengths_um, unit="um"),
                "temperature": Coord(
                    values=[temperature_c + 273.15 for temperature_c in temperatures_c],
                    unit="K",
                ),
            },
            values=[
                index_function(wavelength_um, temperature_c)
                for wavelength_um in wavelengths_um
                for temperature_c in temperatures_c
            ],
            attrs={"polarization": polarization},
            interp="linear",
        ),
    )


def _gayer_anisotropic_index() -> Index:
    """Build Gayer's temperature-dependent principal-index tensor."""
    wavelengths_um = _sample_wavelengths(*_GAYER_VALID_WAVELENGTH_UM)
    temperatures_c = _sample_gayer_temperatures()
    ordinary = _temperature_tabulated_index(
        wavelengths_um,
        temperatures_c,
        lambda wavelength_um, temperature_c: _gayer_index(
            wavelength_um,
            temperature_c,
            _GAYER_ORDINARY_DISPERSION_COEFFICIENTS,
            _GAYER_ORDINARY_TEMPERATURE_COEFFICIENTS,
        ),
        "ordinary",
    )
    extraordinary = _temperature_tabulated_index(
        wavelengths_um,
        temperatures_c,
        lambda wavelength_um, temperature_c: _gayer_index(
            wavelength_um,
            temperature_c,
            _GAYER_EXTRAORDINARY_DISPERSION_COEFFICIENTS,
            _GAYER_EXTRAORDINARY_TEMPERATURE_COEFFICIENTS,
        ),
        "extraordinary",
    )
    minimum_temperature_c, maximum_temperature_c = _GAYER_VALID_TEMPERATURE_C
    return Index(
        validity=Validity(
            at=None,
            over={
                "wavelength": Band(
                    min=_GAYER_VALID_WAVELENGTH_UM[0],
                    max=_GAYER_VALID_WAVELENGTH_UM[1],
                    unit="um",
                    label=None,
                ),
                "temperature": Band(
                    min=minimum_temperature_c + 273.15,
                    max=maximum_temperature_c + 273.15,
                    unit="K",
                    label=None,
                ),
            },
            on_out_of_range="raise",
        ),
        variation=None,
        conductivity=None,
        n=[ordinary, ordinary.model_copy(deep=True), extraordinary],
        k=None,
    )


_UNIAXIAL_CARD_INFO = {
    "anisotropy": "uniaxial",
    "principal_axis_order": ["ordinary", "ordinary", "extraordinary"],
    "optical_axis": "z",
}

LINBO3_ZELMON = material_card(
    name="LiNbO3-Zelmon",
    temperature_ref=294.15,
    permittivity=_anisotropic_index(
        _ZELMON_VALID_WAVELENGTH_UM,
        lambda wavelength_um: _zelmon_index(wavelength_um, _ZELMON_ORDINARY_TERMS),
        lambda wavelength_um: _zelmon_index(wavelength_um, _ZELMON_EXTRAORDINARY_TERMS),
    ),
    provenance=Provenance(
        source="literature",
        label="Zelmon 1997 congruent LiNbO3",
        maturity="empirical",
        citations=[
            Citation(
                role="fit",
                doi="10.1364/JOSAB.14.003319",
                journal=(
                    "Journal of the Optical Society of America B 14, 3319-3322 (1997)"
                ),
                authors="D. E. Zelmon, D. L. Small, and D. Jundt",
                url="https://doi.org/10.1364/JOSAB.14.003319",
            )
        ],
        comment=(
            "Bulk, undoped congruent LiNbO3 measured at 21 °C. The tabulated "
            "principal-index tensor is [no, no, ne]."
        ),
        url="https://doi.org/10.1364/JOSAB.14.003319",
        data_url=None,
        info={},
    ),
    optical_info={"source_model": "three-oscillator Sellmeier"},
    info={**_UNIAXIAL_CARD_INFO, "composition": "congruent LiNbO3"},
)

LINBO3_MGO5_GAYER = material_card(
    name="LiNbO3-MgO5-Gayer",
    temperature_ref=297.65,
    permittivity=_gayer_anisotropic_index(),
    provenance=Provenance(
        source="literature",
        label="Gayer 2008 5 mol% MgO-doped congruent LiNbO3",
        maturity="empirical",
        citations=[
            Citation(
                role="fit",
                doi="10.1007/s00340-008-2998-2",
                journal="Applied Physics B 91, 343-348 (2008)",
                authors="O. Gayer, Z. Sacks, E. Galun, and A. Arie",
                url="https://doi.org/10.1007/s00340-008-2998-2",
            ),
            Citation(
                role="fit",
                doi="10.1007/s00340-008-3316-8",
                journal="Applied Physics B 94, 367 (2009)",
                authors="O. Gayer, Z. Sacks, E. Galun, and A. Arie",
                url="https://doi.org/10.1007/s00340-008-3316-8",
            ),
            Citation(
                role="fit",
                doi="10.1007/s00340-010-4203-7",
                journal="Applied Physics B 101, 481 (2010)",
                authors="O. Gayer, Z. Sacks, E. Galun, and A. Arie",
                url="https://doi.org/10.1007/s00340-010-4203-7",
            ),
        ],
        comment=(
            "Bulk 5 mol% MgO-doped congruent LiNbO3. The tensor is [no, no, "
            "ne], tabulated over wavelength and temperature, with validity "
            "restricted to the range shared by both axes. The coefficients "
            "include the corrections published in the 2009 and 2010 errata."
        ),
        url="https://doi.org/10.1007/s00340-008-2998-2",
        data_url=None,
        info={
            "coefficients_include_published_errata": True,
            "errata_dois": [
                "10.1007/s00340-008-3316-8",
                "10.1007/s00340-010-4203-7",
            ],
        },
    ),
    optical_info={
        "source_model": "temperature-dependent modified Sellmeier",
        "reference_temperature_c": _GAYER_REFERENCE_TEMPERATURE_C,
    },
    info={
        **_UNIAXIAL_CARD_INFO,
        "composition": "5 mol% MgO-doped congruent LiNbO3",
        "mgo_mol_percent": 5.0,
    },
)

__all__ = ["LINBO3_MGO5_GAYER", "LINBO3_ZELMON"]
