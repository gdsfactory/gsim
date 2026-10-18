"""Spectral trace collections for FDTD results."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

_S_PARAMETER_PATTERN = re.compile(r"S\(([^,]+),([^\)]+)\)")


@dataclass(frozen=True)
class ComplexTrace:
    """A complex spectral trace plus optional backend-derived quantities."""

    values: np.ndarray
    valid: np.ndarray
    modal_power: np.ndarray | None = None
    power_fraction: np.ndarray | None = None

    @classmethod
    def from_samples(
        cls, samples: list[Mapping[str, Any]], valid: np.ndarray
    ) -> ComplexTrace:
        """Build a trace from the GDSFactory FDTD list-of-records representation."""
        values = np.asarray(
            [complex(sample.get("re", 0), sample.get("im", 0)) for sample in samples]
        )
        if len(values) != len(valid):
            raise ValueError("spectral trace length does not match frequencies")
        modal_power = _optional_array(samples, "modal_power")
        power_fraction = _optional_array(samples, "power_fraction")
        return cls(values, valid.copy(), modal_power, power_fraction)

    @property
    def magnitude(self) -> np.ndarray:
        """Return linear complex magnitude."""
        return np.abs(self.values)

    @property
    def magnitude_db(self) -> np.ndarray:
        """Return amplitude magnitude in decibels."""
        with np.errstate(divide="ignore"):
            return 20 * np.log10(self.magnitude)

    @property
    def phase_deg(self) -> np.ndarray:
        """Return complex phase in degrees."""
        return np.angle(self.values, deg=True)


def _optional_array(
    samples: list[Mapping[str, Any]], field_name: str
) -> np.ndarray | None:
    """Load an optional scalar field shared by every spectral sample."""
    if not samples or any(field_name not in sample for sample in samples):
        return None
    return np.asarray([sample[field_name] for sample in samples], dtype=float)


class _TraceCollection:
    """Shared dataframe and plotting behavior for spectral trace mappings."""

    def __init__(
        self,
        traces: Mapping[Any, ComplexTrace],
        wavelength_um: np.ndarray,
        frequency_hz: np.ndarray,
    ) -> None:
        """Store spectral traces and their shared wavelength coordinates."""
        self._traces = dict(traces)
        self.wavelength_um = wavelength_um
        self.frequency_hz = frequency_hz

    def __len__(self) -> int:
        """Return the number of spectral traces."""
        return len(self._traces)

    def __iter__(self) -> Iterator[Any]:
        """Iterate over spectral trace keys."""
        return iter(self._traces)

    def items(self):
        """Return trace key-value pairs."""
        return self._traces.items()

    def plot(
        self,
        *,
        quantity: str = "magnitude_db",
        mask_invalid: bool = True,
        ax: Any = None,
    ) -> tuple[Any, Any]:
        """Plot all traces against wavelength, gapping invalid samples."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        for key, trace in self.items():
            values = np.asarray(getattr(trace, quantity), dtype=float).copy()
            if mask_invalid:
                values[~trace.valid] = np.nan
            ax.plot(self.wavelength_um, values, label=self._label(key))
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel(_quantity_label(quantity))
        if self._traces:
            ax.legend()
        return ax.figure, ax

    def plot_plotly(
        self, *, quantity: str = "magnitude_db", mask_invalid: bool = True
    ) -> Any:
        """Return an interactive Plotly plot fitted to finite result samples."""
        import plotly.graph_objects as go

        figure = go.Figure()
        plotted_wavelengths = []
        for key, trace in self.items():
            values = np.asarray(getattr(trace, quantity), dtype=float).copy()
            if mask_invalid:
                values[~trace.valid] = np.nan
            finite = np.isfinite(values) & np.isfinite(self.wavelength_um)
            plotted_wavelengths.extend(self.wavelength_um[finite])
            figure.add_scatter(
                x=self.wavelength_um,
                y=values,
                mode="lines",
                name=self._label(key),
            )
        figure.update_layout(
            xaxis_title="Wavelength (µm)", yaxis_title=_quantity_label(quantity)
        )
        if plotted_wavelengths:
            x_min = float(np.min(plotted_wavelengths))
            x_max = float(np.max(plotted_wavelengths))
            if x_min < x_max:
                figure.update_xaxes(range=[x_min, x_max])
        return figure

    @staticmethod
    def _label(key: Any) -> str:
        """Format a generic trace key for plot legends."""
        return str(key)


def _quantity_label(quantity: str) -> str:
    """Return a human-readable axis label for a plotted quantity."""
    labels = {
        "magnitude": "Magnitude",
        "magnitude_db": "Magnitude (dB)",
        "phase_deg": "Phase (degrees)",
        "modal_power": "Modal power",
        "power_fraction": "Power fraction",
    }
    if quantity not in labels:
        raise ValueError(f"Unknown plot quantity {quantity!r}")
    return labels[quantity]


class SParameterResults(_TraceCollection):
    """One S-matrix column from an eigenmode-source simulation."""

    def __getitem__(self, key: str | tuple[str, str]) -> ComplexTrace:
        """Return one trace by tuple key or ``S(receive,source)`` string."""
        if isinstance(key, str):
            match = _S_PARAMETER_PATTERN.fullmatch(key)
            if match is None:
                raise KeyError(key)
            key = (match.group(1), match.group(2))
        return self._traces[key]

    @staticmethod
    def _label(key: tuple[str, str]) -> str:
        """Format an S-parameter tuple for plot legends."""
        return f"S({key[0]},{key[1]})"

    def to_dataframe(self) -> Any:
        """Return one tidy row per S-parameter and wavelength."""
        import pandas as pd

        records = []
        for (receive_port, source_port), trace in self.items():
            for index, wavelength_um in enumerate(self.wavelength_um):
                records.append(
                    {
                        "wavelength_um": wavelength_um,
                        "frequency_hz": self.frequency_hz[index],
                        "receive_port": receive_port,
                        "source_port": source_port,
                        "real": trace.values[index].real,  # noqa: PD011
                        "imaginary": trace.values[index].imag,  # noqa: PD011
                        "magnitude": trace.magnitude[index],
                        "magnitude_db": trace.magnitude_db[index],
                        "phase_deg": trace.phase_deg[index],
                        "valid": bool(trace.valid[index]),
                    }
                )
        return pd.DataFrame.from_records(records)


class PortOutputResults(_TraceCollection):
    """Outgoing port amplitudes from a non-eigenmode source."""

    def __getitem__(self, port: str) -> ComplexTrace:
        """Return the outgoing trace for one port."""
        return self._traces[port]

    def to_dataframe(self) -> Any:
        """Return one tidy row per port and wavelength."""
        import pandas as pd

        records = []
        for port, trace in self.items():
            for index, wavelength_um in enumerate(self.wavelength_um):
                records.append(
                    {
                        "wavelength_um": wavelength_um,
                        "frequency_hz": self.frequency_hz[index],
                        "port": port,
                        "real": trace.values[index].real,  # noqa: PD011
                        "imaginary": trace.values[index].imag,  # noqa: PD011
                        "modal_power": _array_value(trace.modal_power, index),
                        "power_fraction": _array_value(trace.power_fraction, index),
                        "valid": bool(trace.valid[index]),
                    }
                )
        return pd.DataFrame.from_records(records)


def _array_value(values: np.ndarray | None, index: int) -> float | None:
    """Return one optional array sample as a scalar."""
    return None if values is None else float(values[index])
