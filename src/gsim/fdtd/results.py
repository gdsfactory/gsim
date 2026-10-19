"""Typed loading, tabulation, and plotting of GDSFactory FDTD output."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from gsim.fdtd.modes import SourceModeResult
from gsim.fdtd.traces import (
    _S_PARAMETER_PATTERN,
    ComplexTrace,
    PortOutputResults,
    SParameterResults,
)


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
    dry_run: bool = False
    source_mode: SourceModeResult | None = None
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
        dry_run = bool(document.get("dry_run", False))
        if dry_run and any(
            document.get(key)
            for key in ("s_parameters", "port_outputs", "plane_monitors")
        ):
            raise ValueError("A dry run cannot contain time-domain simulation results")
        valid = ~np.asarray(below_noise, dtype=bool)
        if dry_run:
            valid[:] = False
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
            dry_run=dry_run,
            source_mode=(
                SourceModeResult.from_metadata(
                    document["source_mode"], output_path.parent, file_map
                )
                if document.get("source_mode") is not None
                else None
            ),
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
