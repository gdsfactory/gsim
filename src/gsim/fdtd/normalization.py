"""Source-aware normalization for non-eigenmode FDTD results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gsim.fdtd.api import GaussianBeamSource
from gsim.fdtd.results import FDTDResult, PlaneMonitorResult


@dataclass(frozen=True)
class CouplingEfficiencyResult:
    """One dimensionless coupling-efficiency spectrum."""

    label: str
    wavelength_um: np.ndarray
    efficiency: np.ndarray
    valid: np.ndarray

    @property
    def efficiency_db(self) -> np.ndarray:
        """Return power coupling efficiency in decibels."""
        values = np.full_like(self.efficiency, np.nan, dtype=float)
        selected = self.valid & (self.efficiency > 0)
        values[selected] = 10 * np.log10(self.efficiency[selected])
        return values

    def plot_plotly(self, *, db: bool = True) -> Any:
        """Return an interactive coupling-efficiency plot."""
        import plotly.graph_objects as go

        values = self.efficiency_db if db else 100 * self.efficiency.copy()
        if not db:
            values[~self.valid] = np.nan
        figure = go.Figure()
        figure.add_scatter(
            x=self.wavelength_um,
            y=values,
            mode="lines",
            name=self.label,
        )
        figure.update_layout(
            xaxis_title="Wavelength (µm)",
            yaxis_title="Coupling efficiency (dB)" if db else "Coupling efficiency (%)",
        )
        finite = np.isfinite(values) & np.isfinite(self.wavelength_um)
        if np.count_nonzero(finite) > 1:
            figure.update_xaxes(
                range=[
                    float(np.min(self.wavelength_um[finite])),
                    float(np.max(self.wavelength_um[finite])),
                ]
            )
        return figure


def gaussian_coupling_efficiency(
    result: FDTDResult,
    source: GaussianBeamSource,
    *,
    port: str,
    noise_floor_db: float = 25.0,
    wave_impedance: float = 1.0,
) -> CouplingEfficiencyResult:
    """Normalize raw port power by the analytic Gaussian-beam source power.

    The pulse transform and power expression match GDSFactory FDTD and the
    FDTD-Bench focusing-grating-coupler reference flow. Samples whose analytic
    source power is more than ``noise_floor_db`` below its peak are masked.
    """
    if noise_floor_db < 0:
        raise ValueError("noise_floor_db must be non-negative")
    if wave_impedance <= 0:
        raise ValueError("wave_impedance must be positive")

    trace = result.port_outputs[port]
    if trace.modal_power is None:
        raise ValueError(f"port {port!r} has no modal-power output")
    if len(trace.modal_power) != len(result.wavelength_um):
        raise ValueError("port power and result wavelengths have different lengths")

    wavelength_nm = result.wavelength_um * 1000
    center_wavelength_nm = source.wavelength_um * 1000
    halfspan_nm = source.wavelength_halfspan_um * 1000
    waist_radius_nm = source.waist_radius_um * 1000
    frequency_center, frequency_width = _frequency_parameters(
        center_wavelength_nm, halfspan_nm
    )
    spectrum = _source_spectrum(
        1 / wavelength_nm,
        frequency_center,
        frequency_width,
        source.amplitude,
    )
    spectrum_power = np.abs(spectrum) ** 2
    incident_power = np.pi * waist_radius_nm**2 * spectrum_power / (4 * wave_impedance)

    spectrum_peak = float(np.max(spectrum_power)) if spectrum_power.size else 0.0
    above_noise_floor = (
        spectrum_power >= spectrum_peak * 10 ** (-noise_floor_db / 10)
        if spectrum_peak > 0
        else np.zeros_like(spectrum_power, dtype=bool)
    )
    valid = (
        result.valid
        & trace.valid
        & above_noise_floor
        & (incident_power > 0)
        & (trace.modal_power > 0)
    )
    efficiency = np.full_like(incident_power, np.nan, dtype=float)
    efficiency[valid] = trace.modal_power[valid] / incident_power[valid]
    return CouplingEfficiencyResult(
        label=port,
        wavelength_um=result.wavelength_um.copy(),
        efficiency=efficiency,
        valid=valid,
    )


def fiber_coupling_efficiency(
    monitor: PlaneMonitorResult,
) -> CouplingEfficiencyResult:
    """Convert an eigenmode-normalized fiber overlap into power efficiency."""
    trace = monitor.coupling_efficiency
    if trace is None:
        raise ValueError(
            f"monitor {monitor.name!r} has no normalized fiber-coupling output"
        )
    efficiency = trace.magnitude**2
    valid = trace.valid & np.isfinite(efficiency)
    efficiency = efficiency.astype(float, copy=True)
    efficiency[~valid] = np.nan
    return CouplingEfficiencyResult(
        label=monitor.name,
        wavelength_um=monitor.wavelength_um.copy(),
        efficiency=efficiency,
        valid=valid,
    )


def _frequency_parameters(
    center_wavelength: float, wavelength_halfspan: float
) -> tuple[float, float]:
    """Reproduce GDSFactory FDTD pulse center and width parameters."""
    frequency_center = 1 / center_wavelength
    if wavelength_halfspan <= 0:
        return frequency_center, frequency_center * 0.1
    wavelength_low_frequency = center_wavelength + wavelength_halfspan
    wavelength_high_frequency = max(
        center_wavelength - wavelength_halfspan, center_wavelength * 0.5
    )
    frequency_width = 0.5 * (
        1 / wavelength_high_frequency - 1 / wavelength_low_frequency
    )
    return frequency_center, frequency_width


def _source_spectrum(
    frequency: np.ndarray,
    frequency_center: float,
    frequency_width: float,
    amplitude: float,
) -> np.ndarray:
    """Return the GDSFactory FDTD closed-form Gaussian-pulse Fourier transform."""
    frequency_max_offset = frequency_width * 2 ** (-1.5)
    scale_factor = 2 * frequency_max_offset
    if scale_factor <= 0:
        return np.zeros_like(frequency, dtype=complex)
    tau = 1 / scale_factor
    time_offset = 4 / scale_factor
    positive_term = np.pi * tau * (frequency + frequency_center)
    negative_term = np.pi * tau * (frequency - frequency_center)
    gaussian_difference = np.exp(-(positive_term**2)) - np.exp(-(negative_term**2))
    phase = -2 * np.pi * frequency * time_offset
    prefactor = 0.5j * tau * np.sqrt(np.pi)
    return amplitude * prefactor * np.exp(1j * phase) * gaussian_difference


__all__ = [
    "CouplingEfficiencyResult",
    "fiber_coupling_efficiency",
    "gaussian_coupling_efficiency",
]
