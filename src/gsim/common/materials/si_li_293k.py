"""Li temperature-dependent crystalline-silicon model."""

from math import exp, sqrt

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

from gsim.common.materials._helpers import material_card

_LI_REFERENCE_TEMPERATURE_K = 293.0
_LI_VALID_TEMPERATURE_K = (100.0, 750.0)
_LI_WAVELENGTHS_UM = (
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
)


def _relative_length_change(temperature_kelvin: float) -> float:
    """Return Li's piecewise silicon length change relative to 293 K."""
    if temperature_kelvin <= _LI_REFERENCE_TEMPERATURE_K:
        return (
            -2.1e-4
            - 4.149e-7 * temperature_kelvin
            - 4.620e-10 * temperature_kelvin**2
            + 1.482e-11 * temperature_kelvin**3
        )
    return (
        -7.1e-4
        + 1.887e-6 * temperature_kelvin
        + 1.934e-9 * temperature_kelvin**2
        - 4.544e-13 * temperature_kelvin**3
    )


def _li_index(wavelength_um: float, temperature_kelvin: float) -> float:
    """Evaluate Li's wavelength- and temperature-dependent silicon equation."""
    dielectric_constant = (
        11.4445
        + 2.7739e-4 * temperature_kelvin
        + 1.7050e-6 * temperature_kelvin**2
        - 8.1347e-10 * temperature_kelvin**3
    )
    dispersion_strength = (
        0.8948 + 4.3977e-4 * temperature_kelvin + 7.3835e-8 * temperature_kelvin**2
    )
    lattice_factor = exp(-3.0 * _relative_length_change(temperature_kelvin))
    return sqrt(
        dielectric_constant + lattice_factor * dispersion_strength / wavelength_um**2
    )


def _li_index_table() -> TabulatedValue:
    """Sample Li's silicon equation over wavelength and temperature."""
    temperatures_kelvin = [
        float(temperature_kelvin)
        for temperature_kelvin in range(
            int(_LI_VALID_TEMPERATURE_K[0]),
            int(_LI_VALID_TEMPERATURE_K[1]) + 1,
        )
    ]
    return TabulatedValue(
        unit="",
        data=TableData(
            dims=("wavelength", "temperature"),
            coords={
                "wavelength": Coord(values=list(_LI_WAVELENGTHS_UM), unit="um"),
                "temperature": Coord(values=temperatures_kelvin, unit="K"),
            },
            values=[
                _li_index(wavelength_um, temperature_kelvin)
                for wavelength_um in _LI_WAVELENGTHS_UM
                for temperature_kelvin in temperatures_kelvin
            ],
            attrs={},
            interp="linear",
        ),
    )


SI_LI_293K = material_card(
    name="Si-Li-293K",
    temperature_ref=_LI_REFERENCE_TEMPERATURE_K,
    permittivity=Index(
        validity=Validity(
            at=None,
            over={
                "wavelength": Band(min=1.2, max=14.0, unit="um", label=None),
                "temperature": Band(min=100.0, max=750.0, unit="K", label=None),
            },
            on_out_of_range="raise",
        ),
        variation=None,
        conductivity=None,
        n=_li_index_table(),
        k=None,
    ),
    provenance=Provenance(
        source="literature",
        label="Li 1980 crystalline silicon",
        maturity="empirical",
        citations=[
            Citation(
                role="fit",
                doi="10.1063/1.555624",
                journal=(
                    "Journal of Physical and Chemical Reference Data 9, 561-658 (1980)"
                ),
                authors="H. H. Li",
                url="https://doi.org/10.1063/1.555624",
            )
        ],
        comment=(
            "Crystalline silicon refractive index over 1.2-14 um and "
            "100-750 K, with 293 K as the reference temperature."
        ),
        url="https://doi.org/10.1063/1.555624",
        data_url=None,
        info={},
    ),
    optical_info={
        "source_model": "temperature-dependent dispersion equation",
        "reference_temperature_k": _LI_REFERENCE_TEMPERATURE_K,
    },
    info={"composition": "crystalline silicon"},
)

__all__ = ["SI_LI_293K"]
