"""Conservative monitor-storage estimates for uniform-grid FDTD inputs.

This versioned estimate covers plane accumulators, reference profiles, retained
port-mode data and bounded sample buffers. It excludes the main field grid,
material/PML state and temporary eigensolver workspace.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from gsim.fdtd.config import FDTDConfig


def _span_cells(lower: float, upper: float, cell_size: float) -> int:
    """Bound cells intersecting an interval, including off-grid endpoints."""
    # An off-grid interval can intersect one more cell than its rounded length.
    return math.ceil((upper - lower) / cell_size) + 1


def _sweep(config: FDTDConfig) -> list[float]:
    """Resolve the implicit port-monitor wavelength samples."""
    excitation = config.excitation
    if excitation.num_wavelengths == 1 or excitation.wavelength_halfspan == 0:
        return [excitation.center_wavelength]
    lower = excitation.center_wavelength - excitation.wavelength_halfspan
    step = 2 * excitation.wavelength_halfspan / (excitation.num_wavelengths - 1)
    return [lower + index * step for index in range(excitation.num_wavelengths)]


def _sample_slots(config: FDTDConfig, wavelengths: Sequence[float]) -> int:
    """Bound raw sample-buffer slots from the configured source bandwidth."""
    excitation = config.excitation
    center = excitation.center_wavelength
    frequency_max = 1 / min(wavelengths)
    if excitation.waveform == "continuous_wave":
        frequency_max = max(frequency_max, 1 / center)
    else:
        halfspan = excitation.wavelength_halfspan
        if halfspan:
            halfwidth = 0.5 * (
                1 / max(center - halfspan, center * 0.5) - 1 / (center + halfspan)
            )
            # Gaussian source bandwidth at the supported spectral power floor.
            width = (
                math.pi * halfwidth / (2 * 2**-1.5 * math.sqrt(25 * math.log(10) / 20))
            )
        else:
            width = 0.1 / center
        frequency_max = max(frequency_max, 1 / center + width)
    # This is an upper bound on dt; smaller material-limited dt only reduces
    # the number of buffered samples. Round conservatively at integer strides.
    dt_upper = 0.95 * config.grid.nanometers_per_cell / math.sqrt(3) * (1 + 2**-23)
    stride = max(1, min(64, math.floor(1 / (2 * frequency_max * dt_upper))))
    return 256 // stride + 2


def _entry(
    config: FDTDConfig,
    *,
    name: str,
    kind: str,
    pixels: int,
    wavelengths: Sequence[float],
    reference: bool,
    port_cache: bool = False,
) -> dict[str, Any]:
    """Describe the storage arrays associated with one monitor."""
    frequency_count = len(wavelengths)
    # Planes retain two normal cells, all six complex128 field components.
    accumulators = pixels * 2 * 6 * 16 * frequency_count
    samples = pixels * 2 * 6 * 4 * _sample_slots(config, wavelengths)
    profile_point_bytes = 4 * 16 + int(port_cache)
    profiles = pixels * profile_point_bytes * frequency_count if reference else 0
    # Retained guided modes can include six complex fields, three real and
    # three complex permittivity arrays, and a guide mask. Reserve one extra
    # entry for the source mode when it does not share a sweep window/key.
    cache = pixels * (6 * 16 + 3 * 8 + 3 * 16 + 1) * (frequency_count + 1)
    return {
        "name": name,
        "kind": kind,
        "pixels_upper_bound": pixels,
        "frequency_count": frequency_count,
        "accumulator_bytes": accumulators,
        "sampling_buffer_bytes": samples,
        "reference_profile_bytes": profiles,
        "retained_port_mode_bytes": cache if port_cache else 0,
    }


def estimate_monitor_memory(
    config: FDTDConfig,
    port_bounds_nm: Mapping[str, Mapping[str, Sequence[float]]],
    cell_counts: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate all explicit planes and implicit guided-port monitors in bytes."""
    cell_size = config.grid.nanometers_per_cell
    sweep = _sweep(config)
    entries = []
    for monitor in config.monitors:
        axis = "xyz".index(monitor.normal[-1])
        pixels = math.prod(
            _span_cells(lower, upper, cell_size)
            for index, (lower, upper) in enumerate(
                zip(monitor.region_min, monitor.region_max, strict=True)
            )
            if index != axis
        )
        # Heatmap-only frequencies still need complete field accumulators.
        wavelengths = list(monitor.wavelengths or sweep)
        if monitor.heatmap is not None:
            for wavelength in monitor.heatmap.wavelengths:
                if wavelength not in wavelengths:
                    wavelengths.append(wavelength)
        entries.append(
            _entry(
                config,
                name=monitor.name,
                kind="plane",
                pixels=pixels,
                wavelengths=wavelengths,
                reference=monitor.fiber_mode is not None,
            )
        )
    for name, port in config.geometry.ports.items():
        bounds = port_bounds_nm[name]
        axis = next(index for index, normal in enumerate(port.normal) if normal)
        padding = min(64, max(8, math.ceil(1.5 * max(sweep) / cell_size)))
        pixels = 1
        for index in range(3):
            if index == axis:
                continue
            domain_cap = (
                cell_counts["total_cells_xyz"][index]
                - 2 * cell_counts["pml_cells_per_face_xyz"][index]
            )
            width = _span_cells(bounds["min"][index], bounds["max"][index], cell_size)
            pixels *= min(domain_cap, width + 2 * padding)
        entries.append(
            _entry(
                config,
                name=name,
                kind="guided_port",
                pixels=pixels,
                wavelengths=sweep,
                reference=True,
                port_cache=True,
            )
        )
    categories = (
        "accumulator_bytes",
        "sampling_buffer_bytes",
        "reference_profile_bytes",
        "retained_port_mode_bytes",
    )
    totals = {key: sum(entry[key] for entry in entries) for key in categories}
    return {
        "scope": "client_estimate",
        "estimator_version": 1,
        "estimated_monitor_memory_bytes": sum(totals.values()),
        **totals,
        "monitors": entries,
    }
