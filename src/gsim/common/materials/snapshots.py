"""Evaluate optical MaterialCards at a simulation wavelength."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, sqrt
from typing import Any

from pdk_schema import Index, MaterialCard, ScalarValue, Sellmeier, TabulatedValue

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


def _temperature_to_kelvin(value: float, unit: str) -> float:
    """Convert a supported absolute temperature unit to kelvin."""
    if unit == "K":
        return float(value)
    if unit == "degC":
        return float(value) + 273.15
    raise MaterialModelError(f"Unsupported temperature unit {unit!r}.")


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


def _wavelength_table_values(
    value: TabulatedValue,
    temperature_ref_kelvin: float | None,
) -> tuple[list[float], Sequence[float]]:
    """Return a wavelength curve, selecting a table's reference temperature."""
    table = value.data
    wavelength_coordinate = table.coords.get("wavelength")
    if wavelength_coordinate is None:
        raise MaterialModelError("Optical table is missing wavelength coordinates.")
    coordinates_um = [
        _wavelength_to_um(item, wavelength_coordinate.unit)
        for item in wavelength_coordinate.values
    ]
    if tuple(table.dims) == ("wavelength",):
        return coordinates_um, table.values
    if tuple(table.dims) != ("wavelength", "temperature"):
        raise MaterialModelError(
            "Optical tables must have 'wavelength' or "
            "('wavelength', 'temperature') dimensions."
        )
    if temperature_ref_kelvin is None:
        raise MaterialModelError(
            "Temperature-dependent optical tables require a reference temperature."
        )

    temperature_coordinate = table.coords.get("temperature")
    if temperature_coordinate is None:
        raise MaterialModelError("Optical table is missing temperature coordinates.")
    temperatures_kelvin = [
        _temperature_to_kelvin(item, temperature_coordinate.unit)
        for item in temperature_coordinate.values
    ]
    temperature_count = len(temperatures_kelvin)
    if len(table.values) != len(coordinates_um) * temperature_count:
        raise MaterialModelError(
            "Temperature-dependent optical table has invalid shape."
        )
    reference_values = [
        _interpolate(
            temperatures_kelvin,
            table.values[
                wavelength_index * temperature_count : (wavelength_index + 1)
                * temperature_count
            ],
            temperature_ref_kelvin,
            table.interp,
        )
        for wavelength_index in range(len(coordinates_um))
    ]
    return coordinates_um, reference_values


def _evaluate_value(
    value: Any,
    wavelength_um: float,
    temperature_ref_kelvin: float | None,
) -> float:
    """Evaluate a scalar or tabulated optical value at its reference temperature."""
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
    if any(isinstance(item, list) for item in table.values):
        raise MaterialModelError(
            "Tensor optical values are not supported for passive FDTD."
        )
    coordinates_um, wavelength_values = _wavelength_table_values(
        value, temperature_ref_kelvin
    )
    return _interpolate(
        coordinates_um,
        wavelength_values,
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
    temperature_ref_kelvin = card.optical.temperature_ref
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
    if isinstance(model, Index):
        if isinstance(model.n, list):
            raise MaterialModelError(
                "Tensor refractive indices are not supported for passive FDTD."
            )
        refractive_index = _evaluate_value(
            model.n, wavelength_um, temperature_ref_kelvin
        )
        if model.k is None:
            return refractive_index, 0.0
        if isinstance(model.k, list):
            raise MaterialModelError(
                "Tensor extinction coefficients are not supported for passive FDTD."
            )
        return refractive_index, _evaluate_value(
            model.k, wavelength_um, temperature_ref_kelvin
        )
    raise MaterialModelError(
        f"Unsupported optical model {type(model).__name__} for material {card.name!r}."
    )


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
