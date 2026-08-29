"""Strict GDSFactory FDTD material wire-schema models."""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    """Reject unknown fields and non-finite values at the GDSFactory FDTD boundary."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class IndexTableConfig(_StrictModel):
    """GDSFactory FDTD tabulated complex refractive index."""

    wavelength_nm: list[float] = Field(min_length=1)
    n: list[float] = Field(min_length=1)
    k: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        """Require physical samples on one unambiguous wavelength grid."""
        if len(self.wavelength_nm) != len(self.n) or len(self.n) != len(self.k):
            raise ValueError("wavelength_nm, n, and k must have equal lengths")
        if any(wavelength <= 0 for wavelength in self.wavelength_nm):
            raise ValueError("table wavelengths must be positive")
        if len(set(self.wavelength_nm)) != len(self.wavelength_nm):
            raise ValueError("table wavelengths must be unique")
        if any(value < 0 for value in self.n):
            raise ValueError("table refractive indices cannot be negative")
        if any(value < 0 for value in self.k):
            raise ValueError("table extinction coefficients cannot be negative")
        return self


class SellmeierConfig(_StrictModel):
    """GDSFactory FDTD Sellmeier coefficients."""

    b: list[float] = Field(min_length=1)
    c_um2: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_terms(self) -> Self:
        """Require one pole position for every oscillator strength."""
        if len(self.b) != len(self.c_um2):
            raise ValueError("Sellmeier b and c_um2 must have equal lengths")
        return self


class DrudeConfig(_StrictModel):
    """GDSFactory FDTD free-carrier term, with energies in electron-volts."""

    plasma_energy_ev: float = Field(gt=0)
    damping_ev: float = Field(ge=0)


class LorentzConfig(_StrictModel):
    """GDSFactory FDTD bound-resonance term, with energies in electron-volts."""

    delta_eps: float
    resonance_ev: float = Field(gt=0)
    damping_ev: float = Field(ge=0)


class DrudeLorentzConfig(_StrictModel):
    """GDSFactory FDTD combined Drude-Lorentz parameterization."""

    eps_inf: float = 1.0
    drude: DrudeConfig | None = None
    lorentz: list[LorentzConfig] | None = None

    @model_validator(mode="after")
    def validate_contributions(self) -> Self:
        """Require at least one actual dispersive contribution."""
        if self.drude is None and not self.lorentz:
            raise ValueError("Drude-Lorentz needs a Drude or Lorentz term")
        return self


class DispersionConfig(_StrictModel):
    """One of the three dispersion shapes accepted by GDSFactory FDTD."""

    wavelength_range_nm: tuple[float, float] | None = None
    table: IndexTableConfig | None = None
    sellmeier: SellmeierConfig | None = None
    drude_lorentz: DrudeLorentzConfig | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Enforce GDSFactory FDTD's exact-one shape and range rules."""
        shapes = (self.table, self.sellmeier, self.drude_lorentz)
        if sum(shape is not None for shape in shapes) != 1:
            raise ValueError(
                "dispersion must contain exactly one of table, sellmeier, or "
                "drude_lorentz"
            )
        if self.table is not None:
            if self.wavelength_range_nm is not None:
                raise ValueError("table dispersion cannot set wavelength_range_nm")
            return self
        if self.wavelength_range_nm is None:
            raise ValueError("closed-form dispersion needs wavelength_range_nm")
        lower, upper = self.wavelength_range_nm
        if lower <= 0 or upper <= lower:
            raise ValueError("wavelength_range_nm must satisfy 0 < lower < upper")
        return self


class MaterialConfig(_StrictModel):
    """One scalar or dispersive material entry accepted by GDSFactory FDTD."""

    refractive_index: float | None = Field(default=None, gt=0)
    dispersion: DispersionConfig | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Require exactly one scalar or dispersive representation."""
        if (self.refractive_index is None) == (self.dispersion is None):
            raise ValueError(
                "material needs exactly one of refractive_index or dispersion"
            )
        return self


__all__ = [
    "DispersionConfig",
    "DrudeConfig",
    "DrudeLorentzConfig",
    "IndexTableConfig",
    "LorentzConfig",
    "MaterialConfig",
    "SellmeierConfig",
]
