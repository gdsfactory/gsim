# Copyright 2026 GDSFactory
"""Reference-data utilities for reproducible Palace benchmarks."""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RemoteArtifact:
    """A remotely hosted benchmark artifact with an immutable checksum."""

    name: str
    url: str
    sha256: str
    size_bytes: int | None = None


@dataclass(frozen=True)
class SParameterData:
    """Frequency-indexed two-port scattering parameters."""

    frequency_hz: NDArray[np.float64]
    s: NDArray[np.complex128]
    reference_impedance_ohm: float = 50.0
    source: str | None = None

    def __post_init__(self) -> None:
        """Reject inconsistent arrays before they reach comparison code."""
        frequency_hz = np.asarray(self.frequency_hz, dtype=float)
        scattering = np.asarray(self.s, dtype=complex)
        if frequency_hz.ndim != 1:
            raise ValueError("frequency_hz must be one-dimensional")
        if scattering.shape != (frequency_hz.size, 2, 2):
            raise ValueError("s must have shape (number_of_frequencies, 2, 2)")
        if np.any(np.diff(frequency_hz) < 0):
            raise ValueError("frequency_hz must be monotonic")
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "s", scattering)


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_artifact(
    artifact: RemoteArtifact,
    destination: str | Path,
    *,
    timeout_seconds: float = 60.0,
) -> Path:
    """Download an artifact once and verify its checksum and optional size."""
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if destination_path.exists() and _artifact_matches(artifact, destination_path):
        return destination_path

    partial_path = destination_path.with_name(f".{destination_path.name}.part")
    try:
        with (
            urllib.request.urlopen(  # noqa: S310
                artifact.url, timeout=timeout_seconds
            ) as response,
            partial_path.open("wb") as output_handle,
        ):
            while block := response.read(1024 * 1024):
                output_handle.write(block)
        _validate_artifact(artifact, partial_path)
        partial_path.replace(destination_path)
    finally:
        partial_path.unlink(missing_ok=True)

    return destination_path


def load_touchstone_2port(path: str | Path) -> SParameterData:
    """Load a Touchstone 1.x two-port file in RI, MA, or DB form."""
    option_tokens: list[str] | None = None
    numeric_tokens: list[float] = []

    with Path(path).open(encoding="utf-8-sig") as file_handle:
        for raw_line in file_handle:
            line = raw_line.split("!", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                option_tokens = line[1:].upper().split()
                continue
            numeric_tokens.extend(float(token) for token in line.split())

    if option_tokens is None:
        raise ValueError("Touchstone option line is missing")
    unit, parameter, data_format, reference_impedance = _parse_touchstone_options(
        option_tokens
    )
    if parameter != "S":
        raise ValueError(f"Only S parameters are supported, received {parameter!r}")
    if len(numeric_tokens) % 9:
        raise ValueError("Two-port Touchstone data must contain groups of 9 values")

    rows = np.asarray(numeric_tokens, dtype=float).reshape((-1, 9))
    frequency_hz = rows[:, 0] * _frequency_scale(unit)
    complex_values = _touchstone_complex(rows[:, 1:].reshape((-1, 4, 2)), data_format)

    scattering = np.empty((rows.shape[0], 2, 2), dtype=complex)
    scattering[:, 0, 0] = complex_values[:, 0]  # S11
    scattering[:, 1, 0] = complex_values[:, 1]  # S21
    scattering[:, 0, 1] = complex_values[:, 2]  # S12
    scattering[:, 1, 1] = complex_values[:, 3]  # S22
    return SParameterData(
        frequency_hz=frequency_hz,
        s=scattering,
        reference_impedance_ohm=reference_impedance,
        source=str(path),
    )


def s_to_z(
    scattering: NDArray[np.complex128], reference_impedance_ohm: float = 50.0
) -> NDArray[np.complex128]:
    """Convert two-port S matrices to Z matrices at a real scalar impedance."""
    scattering_array = np.asarray(scattering, dtype=complex)
    if scattering_array.shape[-2:] != (2, 2):
        raise ValueError("scattering must end with shape (2, 2)")
    identity = np.eye(2, dtype=complex)
    return (
        reference_impedance_ohm
        * (identity + scattering_array)
        @ np.linalg.inv(identity - scattering_array)
    )


def differential_inductance_quality(
    data: SParameterData,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return differential inductance in henries and quality factor."""
    impedance = s_to_z(data.s, data.reference_impedance_ohm)
    differential_impedance = (
        impedance[:, 0, 0]
        - impedance[:, 0, 1]
        - impedance[:, 1, 0]
        + impedance[:, 1, 1]
    )
    angular_frequency = 2 * np.pi * data.frequency_hz
    inductance_h = differential_impedance.imag / angular_frequency
    quality_factor = differential_impedance.imag / differential_impedance.real
    return inductance_h, quality_factor


def reciprocity_error(data: SParameterData) -> float:
    """Return the maximum absolute S12-S21 mismatch."""
    return float(np.max(np.abs(data.s[:, 0, 1] - data.s[:, 1, 0])))


def maximum_singular_value(data: SParameterData) -> float:
    """Return the largest singular value across all two-port matrices."""
    return float(np.max(np.linalg.svd(data.s, compute_uv=False)))


def power_loss_fraction(data: SParameterData) -> NDArray[np.float64]:
    """Return incident power not scattered at either equal-impedance port."""
    scattered_power = np.sum(np.abs(data.s) ** 2, axis=1)
    return np.asarray(1.0 - scattered_power, dtype=np.float64)


def from_palace_sparams(results) -> SParameterData:
    """Convert a two-port Palace ``SParams`` result to the benchmark form."""
    port_names = results.port_names
    if len(port_names) != 2:
        raise ValueError(f"Expected two Palace ports, received {len(port_names)}")
    frequency_hz = np.asarray(results.freq, dtype=float) * 1e9
    scattering = np.empty((frequency_hz.size, 2, 2), dtype=complex)
    for to_index, to_port in enumerate(port_names):
        for from_index, from_port in enumerate(port_names):
            scattering[:, to_index, from_index] = results[to_port, from_port].complex
    return SParameterData(
        frequency_hz=frequency_hz,
        s=scattering,
        source="Palace",
    )


def interpolate_sparameters(
    data: SParameterData, frequency_hz: NDArray[np.float64]
) -> SParameterData:
    """Interpolate complex S entries onto an in-band frequency grid."""
    target_frequency = np.asarray(frequency_hz, dtype=float)
    if target_frequency.ndim != 1 or np.any(np.diff(target_frequency) <= 0):
        raise ValueError("Target frequency grid must be one-dimensional and increasing")
    source_frequency, unique_indices = np.unique(data.frequency_hz, return_index=True)
    if (
        target_frequency[0] < source_frequency[0]
        or target_frequency[-1] > source_frequency[-1]
    ):
        raise ValueError("Target frequency grid extends beyond the reference data")

    scattering = np.empty((target_frequency.size, 2, 2), dtype=complex)
    for row in range(2):
        for column in range(2):
            source_values = data.s[unique_indices, row, column]
            real = np.interp(target_frequency, source_frequency, source_values.real)
            imaginary = np.interp(
                target_frequency, source_frequency, source_values.imag
            )
            scattering[:, row, column] = real + 1j * imaginary
    return SParameterData(
        frequency_hz=target_frequency,
        s=scattering,
        reference_impedance_ohm=data.reference_impedance_ohm,
        source=data.source,
    )


def sparameter_error_summary(
    reference: SParameterData, candidate: SParameterData
) -> dict[str, dict[str, float]]:
    """Summarize complex, magnitude, and wrapped-phase error for all entries."""
    if reference.frequency_hz.shape != candidate.frequency_hz.shape or not np.allclose(
        reference.frequency_hz, candidate.frequency_hz, rtol=1e-12, atol=1e-3
    ):
        raise ValueError("Reference and candidate frequency grids do not match")

    labels = (("S11", 0, 0), ("S12", 0, 1), ("S21", 1, 0), ("S22", 1, 1))
    summary: dict[str, dict[str, float]] = {}
    for label, row, column in labels:
        reference_values = reference.s[:, row, column]
        candidate_values = candidate.s[:, row, column]
        absolute_delta = np.abs(candidate_values - reference_values)
        magnitude_delta_db = 20 * np.log10(
            np.maximum(np.abs(candidate_values), np.finfo(float).tiny)
        ) - 20 * np.log10(np.maximum(np.abs(reference_values), np.finfo(float).tiny))
        phase_delta_deg = np.rad2deg(
            np.angle(candidate_values * np.conjugate(reference_values))
        )
        summary[label] = {
            "max_abs_delta_s": float(np.max(absolute_delta)),
            "median_abs_delta_s": float(np.median(absolute_delta)),
            "max_abs_magnitude_delta_db": float(np.max(np.abs(magnitude_delta_db))),
            "median_abs_magnitude_delta_db": float(
                np.median(np.abs(magnitude_delta_db))
            ),
            "max_abs_phase_delta_deg": float(np.max(np.abs(phase_delta_deg))),
            "median_abs_phase_delta_deg": float(np.median(np.abs(phase_delta_deg))),
        }
    return summary


def _artifact_matches(artifact: RemoteArtifact, path: Path) -> bool:
    """Return whether an existing path matches the immutable manifest."""
    try:
        _validate_artifact(artifact, path)
    except ValueError:
        return False
    return True


def _validate_artifact(artifact: RemoteArtifact, path: Path) -> None:
    """Raise when an artifact does not match its manifest."""
    if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
        raise ValueError(
            f"{artifact.name} has size {path.stat().st_size}, expected "
            f"{artifact.size_bytes}"
        )
    actual_digest = sha256_file(path)
    if actual_digest != artifact.sha256:
        raise ValueError(
            f"{artifact.name} has SHA-256 {actual_digest}, expected {artifact.sha256}"
        )


def _parse_touchstone_options(tokens: list[str]) -> tuple[str, str, str, float]:
    """Parse a Touchstone option line, including its R value."""
    if len(tokens) < 3:
        raise ValueError("Touchstone option line must define unit, parameter, and form")
    reference_impedance = 50.0
    if "R" in tokens:
        resistance_index = tokens.index("R")
        try:
            reference_impedance = float(tokens[resistance_index + 1])
        except (IndexError, ValueError) as error:
            raise ValueError("Invalid Touchstone reference impedance") from error
    return tokens[0], tokens[1], tokens[2], reference_impedance


def _frequency_scale(unit: str) -> float:
    """Convert a Touchstone frequency unit to hertz."""
    scales = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}
    try:
        return scales[unit]
    except KeyError as error:
        raise ValueError(f"Unsupported Touchstone frequency unit {unit!r}") from error


def _touchstone_complex(values: NDArray[np.float64], data_format: str) -> NDArray:
    """Convert paired Touchstone columns to complex numbers."""
    first = values[..., 0]
    second = values[..., 1]
    if data_format == "RI":
        return first + 1j * second
    phase = np.deg2rad(second)
    if data_format == "MA":
        return first * np.exp(1j * phase)
    if data_format == "DB":
        return np.power(10.0, first / 20.0) * np.exp(1j * phase)
    raise ValueError(f"Unsupported Touchstone data format {data_format!r}")
