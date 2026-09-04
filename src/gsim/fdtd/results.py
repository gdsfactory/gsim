"""Typed loading, tabulation, and plotting of GDSFactory FDTD output."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class HeatmapResult:
    """Metadata and lazy file loading for one monitor heatmap."""

    file: Path
    quantity: str
    wavelength_um: float
    shape: tuple[int, int]

    def load(self) -> np.ndarray:
        """Load and validate the referenced NumPy field image."""
        values = np.load(self.file)
        if values.shape != self.shape:
            raise ValueError(
                f"heatmap {self.file.name!r} has shape {values.shape}, "
                f"expected {self.shape}"
            )
        return values


@dataclass
class PlaneMonitorResult:
    """Flux, heatmaps, and optional fiber response from one plane monitor."""

    name: str
    normal_axis: str
    normal_sign: int
    u_axis: str
    v_axis: str
    plane_position_um: float
    u_extent_um: tuple[float, float]
    v_extent_um: tuple[float, float]
    shape: tuple[int, int]
    wavelength_um: np.ndarray
    flux: np.ndarray | None = None
    heatmaps: list[HeatmapResult] = field(default_factory=list)
    coupling_efficiency: ComplexTrace | None = None
    fiber_amplitude: ComplexTrace | None = None

    def plot_flux(self, *, ax: Any = None) -> tuple[Any, Any]:
        """Plot signed outward flux against wavelength."""
        import matplotlib.pyplot as plt

        if self.flux is None:
            raise ValueError(f"monitor {self.name!r} has no flux data")
        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.wavelength_um, self.flux)
        ax.set_xlabel("Wavelength (µm)")
        ax.set_ylabel("Outward flux")
        ax.set_title(self.name)
        return ax.figure, ax

    def heatmap(
        self, *, wavelength_um: float, quantity: str | None = None
    ) -> HeatmapResult:
        """Select the closest available heatmap at a requested wavelength."""
        candidates = [
            result
            for result in self.heatmaps
            if quantity is None or result.quantity == quantity
        ]
        if not candidates:
            raise KeyError(f"monitor {self.name!r} has no matching heatmaps")
        return min(candidates, key=lambda item: abs(item.wavelength_um - wavelength_um))

    def plot_heatmap(
        self,
        *,
        wavelength_um: float,
        quantity: str | None = None,
        ax: Any = None,
        **imshow_kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot one lazy-loaded heatmap with its physical plane coordinates."""
        import matplotlib.pyplot as plt

        selected = self.heatmap(wavelength_um=wavelength_um, quantity=quantity)
        if ax is None:
            _, ax = plt.subplots()
        image = ax.imshow(
            selected.load(),
            origin="lower",
            aspect="auto",
            extent=(*self.u_extent_um, *self.v_extent_um),
            **imshow_kwargs,
        )
        ax.set_xlabel(f"{self.u_axis} (µm)")
        ax.set_ylabel(f"{self.v_axis} (µm)")
        ax.set_title(
            f"{self.name}: {selected.quantity} at {selected.wavelength_um:g} µm"
        )
        ax.figure.colorbar(image, ax=ax)
        return ax.figure, ax


class MonitorResults(dict[str, PlaneMonitorResult]):
    """Plane-monitor results keyed by the names configured on the simulation."""


@dataclass
class FDTDResult:
    """Complete typed result from one GDSFactory FDTD cloud run."""

    excitation_type: str
    excited_port: str | None
    ports: list[str]
    wavelength_um: np.ndarray
    frequency_hz: np.ndarray
    valid: np.ndarray
    s_parameters: SParameterResults
    port_outputs: PortOutputResults
    monitors: MonitorResults
    convergence: dict[str, Any]
    grid: dict[str, Any]
    timing: dict[str, Any]
    config_resolved: dict[str, Any]
    output_path: Path
    sim_dir: Path | None = None
    job_name: str = ""
    files: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        files: Mapping[str, Path] | None = None,
        sim_dir: str | Path | None = None,
        job_name: str = "",
    ) -> FDTDResult:
        """Load a result JSON and lazy references to any heatmap sidecars."""
        output_path = Path(path)
        document = json.loads(output_path.read_text(encoding="utf8"))
        wavelength_um = (
            np.asarray(document["frequencies"]["wavelength_nm"], dtype=float) / 1000
        )
        frequency_hz = np.asarray(document["frequencies"]["hz"], dtype=float)
        below_noise = document["frequencies"].get(
            "below_noise_floor", [False] * len(wavelength_um)
        )
        valid = ~np.asarray(below_noise, dtype=bool)
        s_traces = {}
        for name, samples in document.get("s_parameters", {}).items():
            match = _S_PARAMETER_PATTERN.fullmatch(name)
            if match is None:
                raise ValueError(f"Invalid S-parameter name {name!r}")
            s_traces[(match.group(1), match.group(2))] = ComplexTrace.from_samples(
                samples, valid
            )
        port_traces = {
            name: ComplexTrace.from_samples(samples, valid)
            for name, samples in document.get("port_outputs", {}).items()
        }
        file_map = {name: Path(value) for name, value in (files or {}).items()}
        monitors = MonitorResults(
            {
                name: _parse_monitor(name, data, output_path.parent, file_map, valid)
                for name, data in document.get("plane_monitors", {}).items()
            }
        )
        return cls(
            excitation_type=document["excitation_type"],
            excited_port=document.get("excited_port"),
            ports=list(document.get("ports", [])),
            wavelength_um=wavelength_um,
            frequency_hz=frequency_hz,
            valid=valid,
            s_parameters=SParameterResults(s_traces, wavelength_um, frequency_hz),
            port_outputs=PortOutputResults(port_traces, wavelength_um, frequency_hz),
            monitors=monitors,
            convergence=dict(document.get("convergence", {})),
            grid=dict(document.get("grid", {})),
            timing=dict(document.get("timing", {})),
            config_resolved=dict(document.get("config_resolved", {})),
            output_path=output_path,
            sim_dir=Path(sim_dir) if sim_dir is not None else None,
            job_name=job_name,
            files=file_map,
        )

    @classmethod
    def from_run_result(cls, run_result: Any) -> FDTDResult:
        """Locate and parse GDSFactory FDTD output downloaded by :mod:`gsim.gcloud`."""
        candidates = [
            Path(path)
            for name, path in run_result.files.items()
            if name.endswith(".json") and Path(name).stem.startswith("sparams")
        ]
        if not candidates:
            candidates = list(Path(run_result.sim_dir).rglob("sparams*.json"))
        if not candidates:
            raise FileNotFoundError("No sparams*.json file found in FDTD results")
        return cls.from_file(
            candidates[0],
            files=run_result.files,
            sim_dir=run_result.sim_dir,
            job_name=run_result.job_name,
        )

    def plot(self, **kwargs: Any) -> tuple[Any, Any]:
        """Plot S-parameters, or port power for a non-eigenmode source."""
        if self.s_parameters:
            return self.s_parameters.plot(**kwargs)
        kwargs.setdefault("quantity", "modal_power")
        return self.port_outputs.plot(**kwargs)

    def plot_plotly(self, **kwargs: Any) -> Any:
        """Interactively plot S-parameters, or non-eigenmode port power."""
        normalize_to = kwargs.pop("normalize_to", None)
        if self.s_parameters:
            if normalize_to is not None:
                raise ValueError("S-parameters are already source-normalized")
            return self.s_parameters.plot_plotly(**kwargs)
        kwargs.setdefault("quantity", "modal_power")
        if normalize_to is not None and kwargs["quantity"] != "modal_power":
            raise ValueError("monitor normalization requires modal_power")
        figure = self.port_outputs.plot_plotly(**kwargs)
        if normalize_to is None:
            return figure

        monitor = self.monitors[normalize_to]
        if monitor.flux is None:
            raise ValueError(f"monitor {normalize_to!r} has no flux data")
        order = np.argsort(monitor.wavelength_um)
        monitor_power = np.interp(
            self.wavelength_um,
            monitor.wavelength_um[order],
            np.abs(monitor.flux[order]),
            left=np.nan,
            right=np.nan,
        )
        for trace in figure.data:
            trace.y = np.divide(
                trace.y,
                monitor_power,
                out=np.full_like(trace.y, np.nan, dtype=float),
                where=monitor_power > 0,
            )
        figure.update_yaxes(title=f"Power / |{normalize_to} flux|")
        return figure


def _parse_monitor(
    name: str,
    data: Mapping[str, Any],
    output_dir: Path,
    files: Mapping[str, Path],
    valid: np.ndarray,
) -> PlaneMonitorResult:
    """Parse one plane-monitor record and resolve lazy heatmap paths."""
    wavelengths = np.asarray(data.get("wavelength_nm", []), dtype=float) / 1000
    monitor_valid = (
        valid
        if len(valid) == len(wavelengths)
        else np.ones(len(wavelengths), dtype=bool)
    )
    heatmaps = []
    for metadata in data.get("heatmaps", []):
        filename = metadata["file"]
        heatmaps.append(
            HeatmapResult(
                file=files.get(filename, output_dir / filename),
                quantity=metadata["quantity"],
                wavelength_um=float(metadata["wavelength_nm"]) / 1000,
                shape=tuple(metadata["shape"]),
            )
        )
    return PlaneMonitorResult(
        name=name,
        normal_axis=data["normal_axis"],
        normal_sign=int(data["normal_sign"]),
        u_axis=data["u_axis"],
        v_axis=data["v_axis"],
        plane_position_um=float(data["plane_position_nm"]) / 1000,
        u_extent_um=tuple(value / 1000 for value in data["u_extent_nm"]),
        v_extent_um=tuple(value / 1000 for value in data["v_extent_nm"]),
        shape=tuple(data["shape"]),
        wavelength_um=wavelengths,
        flux=np.asarray(data["flux"], dtype=float) if "flux" in data else None,
        heatmaps=heatmaps,
        coupling_efficiency=_optional_complex_trace(
            data.get("coupling_efficiency"), monitor_valid
        ),
        fiber_amplitude=_optional_complex_trace(
            data.get("fiber_amplitude"), monitor_valid
        ),
    )


def _optional_complex_trace(
    samples: list[Mapping[str, Any]] | None, valid: np.ndarray
) -> ComplexTrace | None:
    """Parse an optional complex spectral trace."""
    return None if samples is None else ComplexTrace.from_samples(samples, valid)


__all__ = [
    "ComplexTrace",
    "FDTDResult",
    "HeatmapResult",
    "MonitorResults",
    "PlaneMonitorResult",
    "PortOutputResults",
    "SParameterResults",
]
