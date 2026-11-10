"""Lazy complex plane fields and actual sample coordinates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np

_COMPONENTS = {"ex", "ey", "ez", "hx", "hy", "hz"}
_DFT_CONVENTION = "exp(-i*omega*t)"


@dataclass(frozen=True)
class FieldCoordinates:
    """Actual colocated sample coordinates, expressed in micrometers."""

    u_um: np.ndarray
    v_um: np.ndarray
    normal_position_um: float
    axis_order: tuple[str, str]
    grid_location: str | None = None


@dataclass(frozen=True)
class ComplexFieldResult:
    """One complex component/frequency, with unmodified solver metadata.

    Loading combines the paired arrays without source normalization, phase
    correction or a noise mask. Inspect metadata before quantitative analysis.
    """

    component: str
    wavelength_um: float
    frequency_hz: float
    shape: tuple[int, int]
    real_file: Path
    imag_file: Path
    coordinates: FieldCoordinates
    metadata: dict[str, Any]

    def load(self) -> np.ndarray:
        """Load finite real/imaginary arrays and return their complex values."""
        parts = []
        declared_dtype = self.metadata.get("dtype")
        try:
            expected_dtype = (
                None if declared_dtype is None else np.dtype(declared_dtype)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid complex-field dtype metadata") from error
        for path in (self.real_file, self.imag_file):
            values = np.load(path, allow_pickle=False)
            if not isinstance(values, np.ndarray):
                values.close()
                raise TypeError(f"Complex field {path.name!r} must contain an array")
            if values.shape != self.shape:
                raise ValueError(
                    f"Complex field {path.name!r} has shape {values.shape}, "
                    f"expected {self.shape}"
                )
            if not np.issubdtype(values.dtype, np.floating):
                raise ValueError(
                    f"Complex field {path.name!r} must contain "
                    "real floating-point samples"
                )
            if expected_dtype is not None and values.dtype.newbyteorder(
                "="
            ) != expected_dtype.newbyteorder("="):
                raise ValueError(
                    f"Complex field {path.name!r} dtype differs from "
                    f"declared {declared_dtype!r}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Complex field {path.name!r} contains non-finite samples"
                )
            parts.append(values)
        return parts[0] + 1j * parts[1]


def _coordinates(data: Mapping[str, Any], shape: tuple[int, int]) -> FieldCoordinates:
    """Validate actual coordinates rather than inferring them from requested bounds."""
    if data.get("units") != "nm":
        raise ValueError("Complex-field coordinates must declare units='nm'")
    if data.get("axis_order") != ["v", "u"]:
        raise ValueError("Complex-field coordinates must declare axis_order=['v', 'u']")
    axes = []
    for key, count in (("u", shape[1]), ("v", shape[0])):
        values = np.asarray(data[key], dtype=float)
        if values.shape != (count,) or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Complex-field {key} coordinates must contain {count} finite samples"
            )
        if np.any(np.diff(values) <= 0):
            raise ValueError(
                f"Complex-field {key} coordinates must be strictly increasing"
            )
        axes.append(values / 1000)
    position = float(data["normal_position"])
    if not isfinite(position):
        raise ValueError("Complex-field normal position must be finite")
    return FieldCoordinates(
        u_um=axes[0],
        v_um=axes[1],
        normal_position_um=position / 1000,
        axis_order=("v", "u"),
        grid_location=data.get("grid_location"),
    )


def _sidecar(filename: str, output_dir: Path, files: Mapping[str, Path]) -> Path:
    """Keep result-provided filenames inside the result directory."""
    candidate = output_dir / filename
    if Path(filename).is_absolute() or not candidate.resolve().is_relative_to(
        output_dir.resolve()
    ):
        raise ValueError(
            "Complex-field sidecar paths must stay inside the result directory"
        )
    return files.get(filename, candidate)


def parse_complex_fields(
    data: Mapping[str, Any],
    output_dir: Path,
    files: Mapping[str, Path],
    plane_shape: tuple[int, int],
) -> tuple[list[ComplexFieldResult], FieldCoordinates | None]:
    """Parse optional complex-field metadata without opening NumPy sidecars."""
    entries = data.get("fields", [])
    coordinate_data = data.get("field_coordinates")
    if not entries and coordinate_data is None:
        return [], None
    if len(plane_shape) != 2 or any(
        not isinstance(size, int) or isinstance(size, bool) or size <= 0
        for size in plane_shape
    ):
        raise ValueError("Complex-field shape must contain two positive dimensions")
    # Plane dimensions are (u, v); array rows/columns and field shape are (v, u).
    shape = (plane_shape[1], plane_shape[0])
    coordinates = (
        None if coordinate_data is None else _coordinates(coordinate_data, shape)
    )
    if not entries:
        return [], coordinates
    if coordinates is None:
        raise ValueError("Complex fields require actual field_coordinates metadata")
    fields = []
    for entry in entries:
        if entry.get("dft_convention") != _DFT_CONVENTION:
            raise ValueError(
                "Complex fields must declare dft_convention="
                f"{_DFT_CONVENTION!r}; unsupported or missing DFT convention"
            )
        raw_component = entry["component"]
        if not isinstance(raw_component, str):
            raise TypeError("Complex-field component must be a string")
        component = raw_component.lower()
        if component not in _COMPONENTS:
            raise ValueError(f"Unknown complex field component {component!r}")
        if tuple(entry["shape"]) != shape:
            raise ValueError("Complex-field shape must match its plane coordinates")
        wavelength = float(entry["wavelength_nm"])
        frequency = float(entry["frequency_hz"])
        if not all(isfinite(value) and value > 0 for value in (wavelength, frequency)):
            raise ValueError(
                "Complex-field wavelength and frequency must be finite and positive"
            )
        fields.append(
            ComplexFieldResult(
                component=component,
                wavelength_um=wavelength / 1000,
                frequency_hz=frequency,
                shape=shape,
                real_file=_sidecar(entry["real_file"], output_dir, files),
                imag_file=_sidecar(entry["imag_file"], output_dir, files),
                coordinates=coordinates,
                metadata=dict(entry),
            )
        )
    return fields, coordinates


__all__ = ["ComplexFieldResult", "FieldCoordinates"]
