"""Evaluate optical MaterialCards at a simulation wavelength."""

from __future__ import annotations

from cmath import sqrt as complex_sqrt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, pi, sqrt
from typing import Any

from pdk_schema import (
    Drude,
    Index,
    Lorentz,
    MaterialCard,
    Permittivity,
    PoleResidue,
    ScalarValue,
    Sellmeier,
    SellmeierPoleSquared,
    TabulatedValue,
)
from scipy.constants import c as C0  # noqa: N812

from gsim.common.materials.registry import MaterialSource, find_material_card


class MaterialResolutionError(ValueError):
    """Base error for strict material resolution."""


class MaterialNotFoundError(MaterialResolutionError):
    """Raised when neither the project nor gsim defines a material."""


class MaterialModelError(MaterialResolutionError):
    """Raised when a material card cannot produce a scalar optical index."""


class WavelengthOutOfRangeError(MaterialResolutionError):
    """Raised when the requested wavelength is outside a model's validity."""


@dataclass(frozen=True)
class MaterialSnapshot:
    """Scalar optical properties evaluated at one wavelength."""

    material_name: str
    wavelength_um: float
    refractive_index: float
    extinction_coefficient: float
    source: MaterialSource
    card: MaterialCard


def _wavelength_to_um(value: float, unit: str) -> float:
    """Convert a supported wavelength unit to micrometers."""
    scale = {"um": 1.0, "nm": 1e-3, "m": 1e6}.get(unit)
    if scale is None:
        raise MaterialModelError(f"Unsupported wavelength unit {unit!r}.")
    return float(value) * scale


def _validate_wavelength(model: Any, wavelength_um: float, material_name: str) -> None:
    """Require the wavelength to fall within the model's declared range."""
    validity = getattr(model, "validity", None)
    validity_ranges = {} if validity is None else (validity.over or {})
    wavelength_band = validity_ranges.get("wavelength")
    if wavelength_band is None:
        return
    minimum_um = _wavelength_to_um(wavelength_band.min, wavelength_band.unit)
    maximum_um = _wavelength_to_um(wavelength_band.max, wavelength_band.unit)
    if not minimum_um <= wavelength_um <= maximum_um:
        raise WavelengthOutOfRangeError(
            f"Material {material_name!r} is valid from {minimum_um:g} to "
            f"{maximum_um:g} um, not at {wavelength_um:g} um."
        )


def _interpolate(
    coordinates: Sequence[float],
    values: Sequence[float],
    wavelength_um: float,
    interpolation: str,
) -> float:
    """Evaluate a supported one-dimensional interpolation table."""
    if len(coordinates) != len(values) or not coordinates:
        raise MaterialModelError("Tabulated optical data has inconsistent lengths.")
    if any(right <= left for left, right in pairwise(coordinates)):
        raise MaterialModelError("Tabulated wavelengths must be strictly increasing.")
    if not coordinates[0] <= wavelength_um <= coordinates[-1]:
        raise WavelengthOutOfRangeError(
            f"Tabulated data does not include {wavelength_um:g} um."
        )
    if interpolation == "nearest":
        index = min(
            range(len(coordinates)),
            key=lambda i: abs(coordinates[i] - wavelength_um),
        )
        return float(values[index])
    if interpolation != "linear":
        raise MaterialModelError(
            f"Unsupported table interpolation {interpolation!r}; expected "
            "linear or nearest."
        )
    for index, right in enumerate(coordinates[1:], start=1):
        if wavelength_um <= right:
            left = coordinates[index - 1]
            fraction = (wavelength_um - left) / (right - left)
            return float(
                values[index - 1] + fraction * (values[index] - values[index - 1])
            )
    return float(values[-1])


def _evaluate_value(value: Any, wavelength_um: float) -> float:
    """Evaluate a scalar or one-dimensional tabulated optical value."""
    if isinstance(value, ScalarValue):
        if value.unit:
            raise MaterialModelError("Refractive index values must be dimensionless.")
        return float(value.value)
    if not isinstance(value, TabulatedValue):
        raise MaterialModelError(
            f"Unsupported optical value type {type(value).__name__}; expected "
            "scalar or table."
        )
    table = value.data
    if tuple(table.dims) != ("wavelength",):
        raise MaterialModelError("Optical tables must have one 'wavelength' dimension.")
    coordinate = table.coords.get("wavelength")
    if coordinate is None:
        raise MaterialModelError("Optical table is missing wavelength coordinates.")
    coordinates_um = [
        _wavelength_to_um(item, coordinate.unit) for item in coordinate.values
    ]
    if any(isinstance(item, list) for item in table.values):
        raise MaterialModelError(
            "Tensor optical values are not supported for passive FDTD."
        )
    return _interpolate(
        coordinates_um,
        table.values,
        wavelength_um,
        table.interp,
    )


def _evaluate_permittivity(
    card: MaterialCard, wavelength_um: float
) -> tuple[float, float]:
    """Evaluate one card as scalar refractive index and extinction."""
    if card.optical is None or card.optical.permittivity is None:
        raise MaterialModelError(
            f"Material {card.name!r} has no optical permittivity model."
        )
    model = card.optical.permittivity
    _validate_wavelength(model, wavelength_um, card.name)
    if isinstance(model, Sellmeier):
        wavelength_squared = wavelength_um**2
        index_squared = (
            1.0
            + model.offset
            + sum(
                term.b * wavelength_squared / (wavelength_squared - term.c_um**2)
                for term in model.terms
            )
        )
        if index_squared <= 0:
            raise MaterialModelError(
                f"Material {card.name!r} produced non-positive index squared."
            )
        return sqrt(index_squared), 0.0
    if isinstance(model, SellmeierPoleSquared):
        wavelength_squared = wavelength_um**2
        index_squared = (
            1.0
            + model.offset
            + sum(
                term.b * wavelength_squared / (wavelength_squared - term.c_um2)
                for term in model.terms
            )
        )
        if index_squared <= 0:
            raise MaterialModelError(
                f"Material {card.name!r} produced non-positive index squared."
            )
        return sqrt(index_squared), 0.0
    if isinstance(model, Index):
        if isinstance(model.n, list):
            raise MaterialModelError(
                "Tensor refractive indices are not supported for passive FDTD."
            )
        refractive_index = _evaluate_value(model.n, wavelength_um)
        if model.k is None:
            return refractive_index, 0.0
        if isinstance(model.k, list):
            raise MaterialModelError(
                "Tensor extinction coefficients are not supported for passive FDTD."
            )
        return refractive_index, _evaluate_value(model.k, wavelength_um)
    if isinstance(model, Permittivity):
        if isinstance(model.eps_real, list) or isinstance(model.eps_imag, list):
            raise MaterialModelError(
                "Tensor permittivities are not supported for passive FDTD."
            )
        eps_real = _evaluate_value(model.eps_real, wavelength_um)
        eps_imag = (
            0.0
            if model.eps_imag is None
            else _evaluate_value(model.eps_imag, wavelength_um)
        )
        return _complex_index(eps_real, eps_imag)
    angular_frequency = 2 * pi * C0 / (wavelength_um * 1e-6)
    if isinstance(model, Drude):
        permittivity = complex(model.eps_inf, 0.0)
        for term in model.terms:
            permittivity -= term.omega_p**2 / complex(
                angular_frequency**2,
                term.gamma * angular_frequency,
            )
        return _complex_index(permittivity.real, permittivity.imag)
    if isinstance(model, Lorentz):
        permittivity = complex(model.eps_inf, 0.0)
        for term in model.terms:
            permittivity += (
                term.delta_eps
                * term.omega_0**2
                / complex(
                    term.omega_0**2 - angular_frequency**2,
                    -term.gamma * angular_frequency,
                )
            )
        return _complex_index(permittivity.real, permittivity.imag)
    if isinstance(model, PoleResidue):
        permittivity = complex(model.eps_inf, 0.0)
        for pole in model.poles:
            pole_frequency = complex(*pole.a)
            residue = complex(*pole.c)
            permittivity -= residue / (
                1j * angular_frequency + pole_frequency
            ) + residue.conjugate() / (
                1j * angular_frequency + pole_frequency.conjugate()
            )
        return _complex_index(permittivity.real, permittivity.imag)
    raise MaterialModelError(
        f"Unsupported optical model {type(model).__name__} for material {card.name!r}."
    )


def _complex_index(eps_real: float, eps_imag: float) -> tuple[float, float]:
    """Return the passive square root of one complex relative permittivity."""
    refractive_index = complex_sqrt(complex(eps_real, eps_imag))
    return float(refractive_index.real), float(refractive_index.imag)


def resolve_material_snapshot(
    material_name: str,
    wavelength_um: float,
    project_material_cards: Mapping[str, MaterialCard] | None = None,
) -> MaterialSnapshot:
    """Resolve and evaluate a project-first optical material card."""
    if not isfinite(wavelength_um) or wavelength_um <= 0:
        raise WavelengthOutOfRangeError(
            f"Wavelength must be a finite positive value in um, got {wavelength_um!r}."
        )
    try:
        card, source = find_material_card(material_name, project_material_cards)
    except KeyError as error:
        raise MaterialNotFoundError(error.args[0]) from error
    if not isinstance(card, MaterialCard):
        raise MaterialModelError(
            f"Material {material_name!r} resolves to {type(card).__name__}, "
            "not MaterialCard."
        )
    refractive_index, extinction_coefficient = _evaluate_permittivity(
        card, wavelength_um
    )
    if not isfinite(refractive_index) or refractive_index <= 0:
        raise MaterialModelError(
            f"Material {material_name!r} produced invalid refractive index "
            f"{refractive_index}."
        )
    if not isfinite(extinction_coefficient) or extinction_coefficient < 0:
        raise MaterialModelError(
            f"Material {material_name!r} produced invalid extinction coefficient "
            f"{extinction_coefficient}."
        )
    return MaterialSnapshot(
        material_name=material_name,
        wavelength_um=float(wavelength_um),
        refractive_index=refractive_index,
        extinction_coefficient=extinction_coefficient,
        source=source,
        card=card,
    )


__all__ = [
    "MaterialModelError",
    "MaterialNotFoundError",
    "MaterialResolutionError",
    "MaterialSnapshot",
    "WavelengthOutOfRangeError",
    "resolve_material_snapshot",
]
