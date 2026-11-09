"""Optional complex-field requests for GDSFactory FDTD plane monitors."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from gsim.common.config import MutableModel

FieldComponent = Literal["ex", "ey", "ez", "hx", "hy", "hz"]


class _FieldRequest(MutableModel):
    """Shared component selection and strict finite-value validation."""

    model_config = ConfigDict(allow_inf_nan=False)
    components: list[FieldComponent] | None = Field(default=None, min_length=1)

    @field_validator("components", mode="before")
    @classmethod
    def normalize_components(cls, values: Any) -> Any:
        """Accept the solver's case-insensitive component names."""
        if isinstance(values, (list, tuple)):
            return [
                value.lower() if isinstance(value, str) else value for value in values
            ]
        return values

    @field_validator("components")
    @classmethod
    def unique_components(
        cls, values: list[FieldComponent] | None
    ) -> list[FieldComponent] | None:
        """Reject duplicate components after normalization."""
        if values is not None and len(values) != len(set(values)):
            raise ValueError("field components must be unique")
        return values


def _physical_wavelengths(values: list[float] | None) -> list[float] | None:
    """Reject nonpositive wavelength selections."""
    if values is not None and any(value <= 0 for value in values):
        raise ValueError("field wavelengths must be positive")
    return values


class Fields(_FieldRequest):
    """Save complex fields at selected components and wavelengths.

    Omitted selections use all six components and the monitor's wavelengths.
    """

    wavelengths_um: list[float] | None = Field(default=None, min_length=1)
    _validate_wavelengths = field_validator("wavelengths_um")(_physical_wavelengths)

    def to_config(self) -> FieldsConfig:
        """Translate public micrometers to the solver's nanometers."""
        return FieldsConfig(
            components=self.components,
            wavelengths=None
            if self.wavelengths_um is None
            else [value * 1000 for value in self.wavelengths_um],
        )


class FieldsConfig(_FieldRequest):
    """Complex-field selection in the GDSFactory FDTD runtime schema."""

    wavelengths: list[float] | None = Field(default=None, min_length=1)
    _validate_wavelengths = field_validator("wavelengths")(_physical_wavelengths)


__all__ = ["Fields", "FieldsConfig"]
