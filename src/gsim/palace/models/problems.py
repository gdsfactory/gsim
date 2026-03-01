"""Problem-specific configuration models for Palace simulations.

This module contains Pydantic models for different simulation types:
- DrivenConfig: Frequency-domain driven simulation (S-parameters)
- EigenmodeConfig: Eigenmode/resonance simulation
- ElectrostaticConfig: Electrostatic capacitance extraction
- MagnetostaticConfig: Magnetostatic inductance extraction
- TransientConfig: Time-domain simulation
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DrivenConfig(BaseModel):
    """Configuration for driven (frequency sweep) simulation.

    This is used for S-parameter extraction and frequency response analysis.

    Attributes:
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz
        num_points: Number of frequency points
        scale: Frequency spacing ("linear" or "log")
        adaptive_tol: Adaptive tolerance (0 disables adaptive)
        adaptive_max_samples: Maximum samples for adaptive refinement
        compute_s_params: Whether to compute S-parameters
        reference_impedance: Reference impedance for S-params (Ohms)
        excitation_port: Name of port to excite (None = first port)
    """

    model_config = ConfigDict(validate_assignment=True)

    fmin: float = Field(default=1e9, gt=0, description="Min frequency in Hz")
    fmax: float = Field(default=100e9, gt=0, description="Max frequency in Hz")
    num_points: int = Field(default=40, ge=1, description="Number of frequency points")
    scale: Literal["linear", "log"] = "linear"

    # Adaptive options
    adaptive_tol: float = Field(
        default=0.02, ge=0, description="Adaptive tolerance (0 disables adaptive)"
    )
    adaptive_max_samples: int = Field(default=20, ge=1)

    # S-parameter options
    compute_s_params: bool = True
    reference_impedance: float = Field(default=50.0, gt=0)

    # Excitation
    excitation_port: str | None = Field(
        default=None, description="Port to excite (None = first port)"
    )

    @model_validator(mode="after")
    def validate_frequency_range(self) -> Self:
        """Validate that fmin < fmax."""
        if self.fmin >= self.fmax:
            raise ValueError(f"fmin ({self.fmin}) must be less than fmax ({self.fmax})")
        return self

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        freq_step = (self.fmax - self.fmin) / max(1, self.num_points - 1) / 1e9
        config: dict = {
            "Samples": [
                {
                    "Type": "Linear" if self.scale == "linear" else "Log",
                    "MinFreq": self.fmin / 1e9,
                    "MaxFreq": self.fmax / 1e9,
                    "FreqStep": freq_step,
                    "SaveStep": 0,
                }
            ],
            "AdaptiveTol": max(0, self.adaptive_tol),
        }
        if self.adaptive_tol > 0:
            config["AdaptiveMaxSamples"] = self.adaptive_max_samples
        return config


class EigenmodeConfig(BaseModel):
    """Configuration for eigenmode (resonance) simulation.

    This is used for finding resonant frequencies and mode shapes.

    Attributes:
        num_modes: Number of modes to find
        target: Target frequency in Hz for mode search
        tolerance: Eigenvalue solver tolerance
    """

    model_config = ConfigDict(validate_assignment=True)

    num_modes: int = Field(
        default=10, ge=1, alias="N", description="Number of modes to find"
    )
    target: float | None = Field(default=None, description="Target frequency in Hz")
    tolerance: float = Field(
        default=1e-6, gt=0, description="Eigenvalue solver tolerance"
    )

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        config: dict = {
            "N": self.num_modes,
            "Tol": self.tolerance,
        }
        if self.target is not None:
            config["Target"] = self.target / 1e9  # Convert to GHz
        return config


class ElectrostaticConfig(BaseModel):
    """Configuration for electrostatic (capacitance matrix) simulation.

    Attributes:
        save_fields: Number of field solutions to save
    """

    model_config = ConfigDict(validate_assignment=True)

    save_fields: int = Field(default=0, ge=0, description="Number of fields to save")

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        return {
            "Save": self.save_fields,
        }


class MagnetostaticConfig(BaseModel):
    """Configuration for magnetostatic (inductance matrix) simulation.

    Attributes:
        save_fields: Number of field solutions to save
    """

    model_config = ConfigDict(validate_assignment=True)

    save_fields: int = Field(default=0, ge=0, description="Number of fields to save")

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        return {
            "Save": self.save_fields,
        }


class TransientConfig(BaseModel):
    """Configuration for transient (time-domain) simulation.

    Attributes:
        max_time: Maximum simulation time in ns
        excitation: Excitation waveform type
        excitation_freq: Excitation frequency in Hz (for sinusoidal)
        excitation_width: Pulse width in ns (for gaussian)
        time_step: Time step in ns (None = adaptive)
    """

    model_config = ConfigDict(validate_assignment=True)

    excitation: Literal["sinusoidal", "gaussian", "ramp", "smoothstep"] = "sinusoidal"
    excitation_freq: float | None = Field(
        default=None, description="Excitation frequency in Hz"
    )
    excitation_width: float | None = Field(
        default=None, description="Pulse width in ns (for gaussian)"
    )
    max_time: float = Field(description="Maximum simulation time in ns")
    time_step: float | None = Field(
        default=None, description="Time step in ns (None = adaptive)"
    )

    def to_palace_config(self) -> dict:
        """Convert to Palace JSON config format."""
        config: dict = {
            "Type": self.excitation.capitalize(),
            "MaxTime": self.max_time,
        }
        if self.excitation_freq is not None:
            config["ExcitationFreq"] = self.excitation_freq / 1e9  # Convert to GHz
        if self.excitation_width is not None:
            config["ExcitationWidth"] = self.excitation_width
        if self.time_step is not None:
            config["TimeStep"] = self.time_step
        return config


__all__ = [
    "DrivenConfig",
    "EigenmodeConfig",
    "ElectrostaticConfig",
    "MagnetostaticConfig",
    "TransientConfig",
]
