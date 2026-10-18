"""Inspection of the power-normalized port mode injected by ZAP."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SourceModeResult:
    """The actual source profile at its center wavelength, in world axes.

    Fields retain their native Yee staggering. ``abs_e`` and ``abs_h`` combine
    components at matching array indices for a preview of the mode envelope.
    Arrays have shape ``(v, u)`` and load only when requested.
    """

    port: str
    wavelength_um: float
    n_eff: float
    beta_per_um: float
    normal_axis: str
    normal_sign: int
    u_axis: str
    v_axis: str
    plane_position_um: float
    u_extent_um: tuple[float, float]
    v_extent_um: tuple[float, float]
    shape: tuple[int, int]
    field_files: Mapping[str, tuple[Path, Path]] = field(repr=False)

    @classmethod
    def from_metadata(
        cls,
        data: Mapping[str, Any],
        output_dir: Path,
        files: Mapping[str, Path],
    ) -> SourceModeResult:
        """Parse mode metadata and resolve its real/imaginary NumPy sidecars."""

        def resolve(filename: str) -> Path:
            return files.get(filename, output_dir / filename)

        return cls(
            port=data["port"],
            wavelength_um=float(data["wavelength_nm"]) / 1000,
            n_eff=float(data["n_eff"]),
            beta_per_um=float(data["beta_per_um"]),
            normal_axis=data["normal_axis"],
            normal_sign=int(data["normal_sign"]),
            u_axis=data["u_axis"],
            v_axis=data["v_axis"],
            plane_position_um=float(data["plane_position_nm"]) / 1000,
            u_extent_um=tuple(value / 1000 for value in data["u_extent_nm"]),
            v_extent_um=tuple(value / 1000 for value in data["v_extent_nm"]),
            shape=tuple(data["shape"]),
            field_files={
                component: (resolve(record["real_file"]), resolve(record["imag_file"]))
                for component, record in data["fields"].items()
            },
        )

    def load(self, component: str = "Ex") -> np.ndarray:
        """Load a complex world-axis component (Ex/Ey/Ez/Hx/Hy/Hz)."""
        if component not in self.field_files:
            raise ValueError(
                f"Unknown mode component {component!r}; "
                f"choose from {', '.join(self.field_files)}"
            )
        real_path, imag_path = self.field_files[component]
        real = np.load(real_path, allow_pickle=False)
        imaginary = np.load(imag_path, allow_pickle=False)
        if real.shape != self.shape or imaginary.shape != self.shape:
            raise ValueError(f"Mode component {component} must have shape {self.shape}")
        return real + 1j * imaginary

    def _plot_values(self, component: str, part: str) -> np.ndarray:
        """Select a field envelope or a complex component's display part."""
        if component in {"abs_e", "abs_h"}:
            prefix = "E" if component == "abs_e" else "H"
            return np.sqrt(sum(np.abs(self.load(prefix + axis)) ** 2 for axis in "xyz"))
        values = self.load(component)
        if part == "abs":
            return np.abs(values)
        if part == "real":
            return values.real
        if part == "imag":
            return values.imag
        raise ValueError("part must be 'abs', 'real', or 'imag'")

    def plot(
        self,
        component: str = "abs_e",
        *,
        part: str = "abs",
        ax: Any = None,
        **imshow_kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot the 2D mode with physical coordinates and effective index."""
        import matplotlib.pyplot as plt

        values = self._plot_values(component, part)
        if ax is None:
            _, ax = plt.subplots()
        image = ax.imshow(
            values,
            origin="lower",
            extent=(*self.u_extent_um, *self.v_extent_um),
            **imshow_kwargs,
        )
        ax.set_xlabel(f"{self.u_axis} (µm)")
        ax.set_ylabel(f"{self.v_axis} (µm)")
        ax.set_title(
            f"{self.port}: {component}, wavelength = {self.wavelength_um:g} µm, "
            f"n_eff = {self.n_eff:.5f}"
        )
        ax.figure.colorbar(image, ax=ax)
        return ax.figure, ax


__all__ = ["SourceModeResult"]
