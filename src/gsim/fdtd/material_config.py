"""Translate optical MaterialCards into GDSFactory FDTD material configuration."""

from __future__ import annotations

from math import hypot, isclose, sqrt
from typing import Any

from pdk_schema import (
    Drude,
    Index,
    Lorentz,
    MaterialCard,
    Permittivity,
    ScalarValue,
    Sellmeier,
    SellmeierPoleSquared,
    TabulatedValue,
)
from scipy.constants import electron_volt, hbar

from gsim.common.materials import MaterialSnapshot
from gsim.fdtd.material_schema import (
    DispersionConfig,
    DrudeConfig,
    DrudeLorentzConfig,
    IndexTableConfig,
    LorentzConfig,
    MaterialConfig,
    SellmeierConfig,
)
from gsim.fdtd.models import FDTDConfigError


def _config_error(material_name: str, message: str) -> FDTDConfigError:
    """Return one error scoped to a named material."""
    return FDTDConfigError(f"Material {material_name!r} {message}")


def _wavelengths_nm(values: list[float], unit: str, material_name: str) -> list[float]:
    """Convert MaterialCard wavelength values to nanometers."""
    scales = {"nm": 1.0, "um": 1000.0, "m": 1e9}
    try:
        scale = scales[unit]
    except KeyError as error:
        raise _config_error(
            material_name, f"uses unsupported wavelength unit {unit!r}."
        ) from error
    return [float(value) * scale for value in values]


def _dimensionless_scalar(value: Any, material_name: str, field: str) -> float:
    """Extract one dimensionless scalar physical value."""
    if not isinstance(value, ScalarValue) or value.unit != "":
        raise _config_error(
            material_name, f"{field} must be a dimensionless scalar or table."
        )
    return float(value.value)


def _table_samples(
    value: TabulatedValue,
    material_name: str,
    field: str,
) -> tuple[list[float], list[float]]:
    """Extract one linear wavelength table in nanometers."""
    if value.unit != "":
        raise _config_error(material_name, f"{field} values must be dimensionless.")
    table = value.data
    if tuple(table.dims) != ("wavelength",):
        raise _config_error(
            material_name, f"{field} must have one 'wavelength' dimension."
        )
    coordinate = table.coords.get("wavelength")
    if coordinate is None:
        raise _config_error(material_name, f"{field} has no wavelength coordinate.")
    if table.interp != "linear":
        raise _config_error(
            material_name,
            f"{field} uses {table.interp!r} interpolation; GDSFactory FDTD requires "
            "linear table semantics.",
        )
    wavelengths = _wavelengths_nm(coordinate.values, coordinate.unit, material_name)
    values = [float(item) for item in table.values]
    if len(wavelengths) != len(values):
        raise _config_error(material_name, f"{field} has inconsistent table lengths.")
    return wavelengths, values


def _matching_grid(first: list[float], second: list[float], material_name: str) -> None:
    """Require two tables to share one physical wavelength grid."""
    if len(first) != len(second) or any(
        not isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)
        for left, right in zip(first, second, strict=False)
    ):
        raise _config_error(
            material_name, "n and k tables must use the same wavelength grid."
        )


def _require_fit_band(
    wavelength_range_nm: tuple[float, float] | None,
    material_name: str,
) -> tuple[float, float]:
    """Return a usable source band or reject a dispersive material."""
    if wavelength_range_nm is None:
        raise _config_error(
            material_name,
            "is dispersive, but the excitation does not provide the nonzero "
            "frequency band required by GDSFactory FDTD.",
        )
    return wavelength_range_nm


def _validate_model_range(
    model: Any,
    wavelength_range_nm: tuple[float, float],
    material_name: str,
) -> None:
    """Require the source band to lie within model validity."""
    validity = getattr(model, "validity", None)
    wavelength_band = (
        None if validity is None else (validity.over or {}).get("wavelength")
    )
    if wavelength_band is None:
        return
    limits_nm = _wavelengths_nm(
        [wavelength_band.min, wavelength_band.max],
        wavelength_band.unit,
        material_name,
    )
    lower, upper = wavelength_range_nm
    if lower < limits_nm[0] or upper > limits_nm[1]:
        raise _config_error(
            material_name,
            f"is valid from {limits_nm[0]:g} to {limits_nm[1]:g} nm, not over "
            f"the source band {lower:g} to {upper:g} nm.",
        )


def _validate_table_coverage(
    wavelengths_nm: list[float],
    wavelength_range_nm: tuple[float, float],
    material_name: str,
) -> None:
    """Require a material table to cover the complete source band."""
    lower, upper = wavelength_range_nm
    if min(wavelengths_nm) > lower or max(wavelengths_nm) < upper:
        raise _config_error(
            material_name,
            f"table does not cover the source band {lower:g} to {upper:g} nm.",
        )


def _index_config(
    model: Index,
    wavelength_range_nm: tuple[float, float] | None,
    material_name: str,
) -> MaterialConfig:
    """Translate an index-authored card model to GDSFactory FDTD."""
    if model.conductivity is not None:
        raise _config_error(
            material_name,
            "has Index conductivity, which GDSFactory FDTD cannot encode.",
        )
    n_table = (
        _table_samples(model.n, material_name, "n")
        if isinstance(model.n, TabulatedValue)
        else None
    )
    k_table = (
        _table_samples(model.k, material_name, "k")
        if isinstance(model.k, TabulatedValue)
        else None
    )
    if n_table is None and k_table is None:
        refractive_index = _dimensionless_scalar(model.n, material_name, "n")
        extinction = (
            0.0
            if model.k is None
            else _dimensionless_scalar(model.k, material_name, "k")
        )
        if extinction == 0:
            return MaterialConfig(refractive_index=refractive_index)
        wavelengths_nm = list(_require_fit_band(wavelength_range_nm, material_name))
        n_values = [refractive_index] * len(wavelengths_nm)
        k_values = [extinction] * len(wavelengths_nm)
    else:
        reference_table = n_table if n_table is not None else k_table
        if reference_table is None:
            raise AssertionError("table branch requires n or k samples")
        wavelengths_nm = list(reference_table[0])
        if n_table is not None and k_table is not None:
            _matching_grid(n_table[0], k_table[0], material_name)
        n_values = (
            n_table[1]
            if n_table is not None
            else [_dimensionless_scalar(model.n, material_name, "n")]
            * len(wavelengths_nm)
        )
        k_values = (
            k_table[1]
            if k_table is not None
            else [
                0.0
                if model.k is None
                else _dimensionless_scalar(model.k, material_name, "k")
            ]
            * len(wavelengths_nm)
        )
    fit_band = _require_fit_band(wavelength_range_nm, material_name)
    _validate_model_range(model, fit_band, material_name)
    _validate_table_coverage(wavelengths_nm, fit_band, material_name)
    samples = sorted(zip(wavelengths_nm, n_values, k_values, strict=True))
    return MaterialConfig(
        dispersion=DispersionConfig(
            table=IndexTableConfig(
                wavelength_nm=[sample[0] for sample in samples],
                n=[sample[1] for sample in samples],
                k=[sample[2] for sample in samples],
            )
        )
    )


def _complex_index(eps_real: float, eps_imag: float) -> tuple[float, float]:
    """Return the passive square root of complex relative permittivity."""
    magnitude = hypot(eps_real, eps_imag)
    refractive_index = sqrt(max(0.0, (magnitude + eps_real) / 2))
    extinction = sqrt(max(0.0, (magnitude - eps_real) / 2))
    return refractive_index, extinction if eps_imag >= 0 else -extinction


def _permittivity_config(
    model: Permittivity,
    wavelength_range_nm: tuple[float, float] | None,
    material_name: str,
) -> MaterialConfig:
    """Translate a permittivity-authored card model to GDSFactory FDTD."""
    if model.conductivity is not None:
        raise _config_error(
            material_name,
            "has Permittivity conductivity, which GDSFactory FDTD cannot encode.",
        )
    real_table = (
        _table_samples(model.eps_real, material_name, "eps_real")
        if isinstance(model.eps_real, TabulatedValue)
        else None
    )
    imag_table = (
        _table_samples(model.eps_imag, material_name, "eps_imag")
        if isinstance(model.eps_imag, TabulatedValue)
        else None
    )
    if real_table is None and imag_table is None:
        eps_real = _dimensionless_scalar(model.eps_real, material_name, "eps_real")
        eps_imag = (
            0.0
            if model.eps_imag is None
            else _dimensionless_scalar(model.eps_imag, material_name, "eps_imag")
        )
        refractive_index, extinction = _complex_index(eps_real, eps_imag)
        if extinction == 0 and refractive_index > 0:
            return MaterialConfig(refractive_index=refractive_index)
        wavelengths_nm = list(_require_fit_band(wavelength_range_nm, material_name))
        indices = [(refractive_index, extinction)] * len(wavelengths_nm)
    else:
        reference_table = real_table if real_table is not None else imag_table
        if reference_table is None:
            raise AssertionError("table branch requires real or imaginary samples")
        wavelengths_nm = list(reference_table[0])
        if real_table is not None and imag_table is not None:
            _matching_grid(real_table[0], imag_table[0], material_name)
        real_values = (
            real_table[1]
            if real_table is not None
            else [_dimensionless_scalar(model.eps_real, material_name, "eps_real")]
            * len(wavelengths_nm)
        )
        imag_values = (
            imag_table[1]
            if imag_table is not None
            else [
                0.0
                if model.eps_imag is None
                else _dimensionless_scalar(model.eps_imag, material_name, "eps_imag")
            ]
            * len(wavelengths_nm)
        )
        indices = [
            _complex_index(real, imag)
            for real, imag in zip(real_values, imag_values, strict=True)
        ]
    fit_band = _require_fit_band(wavelength_range_nm, material_name)
    _validate_model_range(model, fit_band, material_name)
    _validate_table_coverage(wavelengths_nm, fit_band, material_name)
    samples = sorted(
        (
            (wavelength, refractive_index, extinction)
            for wavelength, (refractive_index, extinction) in zip(
                wavelengths_nm, indices, strict=True
            )
        )
    )
    return MaterialConfig(
        dispersion=DispersionConfig(
            table=IndexTableConfig(
                wavelength_nm=[sample[0] for sample in samples],
                n=[sample[1] for sample in samples],
                k=[sample[2] for sample in samples],
            )
        )
    )


def _sellmeier_config(
    model: Sellmeier | SellmeierPoleSquared,
    wavelength_range_nm: tuple[float, float] | None,
    material_name: str,
) -> MaterialConfig:
    """Translate either MaterialCard Sellmeier convention to GDSFactory FDTD."""
    if model.offset != 0:
        raise _config_error(
            material_name,
            f"has Sellmeier offset {model.offset}; GDSFactory FDTD requires offset 0.",
        )
    if model.conductivity is not None:
        raise _config_error(
            material_name,
            "has Sellmeier conductivity, which GDSFactory FDTD cannot encode.",
        )
    fit_band = _require_fit_band(wavelength_range_nm, material_name)
    _validate_model_range(model, fit_band, material_name)
    if isinstance(model, Sellmeier):
        coefficients = [term.c_um**2 for term in model.terms]
    else:
        coefficients = [term.c_um2 for term in model.terms]
    return MaterialConfig(
        dispersion=DispersionConfig(
            wavelength_range_nm=fit_band,
            sellmeier=SellmeierConfig(
                b=[term.b for term in model.terms],
                c_um2=coefficients,
            ),
        )
    )


def _angular_frequency_to_ev(value: float) -> float:
    """Convert angular frequency in radians per second to energy in eV."""
    return float(value) * hbar / electron_volt


def _drude_lorentz_config(
    model: Drude | Lorentz,
    wavelength_range_nm: tuple[float, float] | None,
    material_name: str,
) -> MaterialConfig:
    """Translate one MaterialCard Drude or Lorentz model to GDSFactory FDTD."""
    fit_band = _require_fit_band(wavelength_range_nm, material_name)
    _validate_model_range(model, fit_band, material_name)
    if isinstance(model, Drude):
        if len(model.terms) != 1:
            raise _config_error(
                material_name,
                "has multiple Drude terms; GDSFactory FDTD accepts exactly one.",
            )
        term = model.terms[0]
        drude = DrudeConfig(
            plasma_energy_ev=_angular_frequency_to_ev(term.omega_p),
            damping_ev=_angular_frequency_to_ev(term.gamma),
        )
        lorentz = None
    else:
        drude = None
        lorentz = [
            LorentzConfig(
                delta_eps=term.delta_eps,
                resonance_ev=_angular_frequency_to_ev(term.omega_0),
                damping_ev=_angular_frequency_to_ev(term.gamma),
            )
            for term in model.terms
        ]
    return MaterialConfig(
        dispersion=DispersionConfig(
            wavelength_range_nm=fit_band,
            drude_lorentz=DrudeLorentzConfig(
                eps_inf=model.eps_inf,
                drude=drude,
                lorentz=lorentz,
            ),
        )
    )


def material_config_from_snapshot(
    snapshot: MaterialSnapshot,
    wavelength_range_nm: tuple[float, float] | None,
) -> MaterialConfig:
    """Convert one resolved MaterialCard to GDSFactory FDTD's material schema."""
    material_name = snapshot.material_name
    card: MaterialCard = snapshot.card
    if card.optical is None or card.optical.permittivity is None:
        raise _config_error(material_name, "has no optical permittivity model.")
    if card.optical.conductivity is not None:
        raise _config_error(
            material_name,
            "has regime conductivity, which GDSFactory FDTD cannot encode.",
        )
    if card.optical.permeability is not None:
        raise _config_error(
            material_name,
            "has optical permeability, which GDSFactory FDTD cannot encode.",
        )
    model = card.optical.permittivity
    if isinstance(model, Index):
        return _index_config(model, wavelength_range_nm, material_name)
    if isinstance(model, Permittivity):
        return _permittivity_config(model, wavelength_range_nm, material_name)
    if isinstance(model, (Sellmeier, SellmeierPoleSquared)):
        return _sellmeier_config(model, wavelength_range_nm, material_name)
    if isinstance(model, (Drude, Lorentz)):
        return _drude_lorentz_config(model, wavelength_range_nm, material_name)
    raise _config_error(
        material_name,
        f"uses unsupported optical model {type(model).__name__} for GDSFactory FDTD.",
    )


__all__ = ["MaterialConfig", "material_config_from_snapshot"]
