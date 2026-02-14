"""Result models for MEEP simulation output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SParameterResult(BaseModel):
    """S-parameter results from MEEP simulation.

    Parses CSV output from the cloud runner and provides
    visualization via matplotlib.
    """

    model_config = ConfigDict(validate_assignment=True)

    wavelengths: list[float] = Field(default_factory=list)
    s_params: dict[str, list[complex]] = Field(
        default_factory=dict,
        description="S-param name -> list of complex values per wavelength",
    )
    port_names: list[str] = Field(default_factory=list)
    debug_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Eigenmode diagnostics from meep_debug.json (if available)",
    )

    @classmethod
    def from_csv(cls, path: str | Path) -> SParameterResult:
        """Parse S-parameter results from CSV file.

        Expected CSV format:
            wavelength,S11_mag,S11_phase,S21_mag,S21_phase,...
            1.5, 0.1, -30.0, 0.9, 45.0, ...

        Automatically loads ``meep_debug.json`` from the same directory
        if it exists, populating the ``debug_info`` field.

        Args:
            path: Path to CSV file

        Returns:
            SParameterResult instance
        """
        import cmath

        path = Path(path)
        wavelengths: list[float] = []
        s_params: dict[str, list[complex]] = {}
        port_names: set[str] = set()

        with open(path) as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return cls()

            # Discover S-param columns
            sparam_names: list[str] = []
            for col in reader.fieldnames:
                if col.endswith("_mag"):
                    name = col.removesuffix("_mag")
                    sparam_names.append(name)
                    s_params[name] = []

            for row in reader:
                wavelengths.append(float(row["wavelength"]))
                for name in sparam_names:
                    mag = float(row[f"{name}_mag"])
                    phase_deg = float(row[f"{name}_phase"])
                    phase_rad = cmath.pi * phase_deg / 180.0
                    s_params[name].append(cmath.rect(mag, phase_rad))

        # Extract port names from S-param names (e.g., "S11" -> port 1)
        for name in sparam_names:
            # S-param format: "Sij" where i,j are port indices
            if len(name) >= 3 and name[0] == "S":
                indices = name[1:]
                for idx_char in indices:
                    port_names.add(f"port_{idx_char}")

        # Auto-load debug log if present alongside CSV
        debug_info: dict[str, Any] = {}
        debug_path = path.parent / "meep_debug.json"
        if debug_path.exists():
            try:
                debug_info = json.loads(debug_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        return cls(
            wavelengths=wavelengths,
            s_params=s_params,
            port_names=sorted(port_names),
            debug_info=debug_info,
        )

    def plot(self, db: bool = True, **kwargs: Any) -> Any:
        """Plot S-parameters vs wavelength.

        Args:
            db: If True, plot in dB scale
            **kwargs: Passed to matplotlib plot()

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        for name, values in self.s_params.items():
            magnitudes = [abs(v) for v in values]
            if db:
                import math

                y_vals = [20 * math.log10(m) if m > 0 else -100 for m in magnitudes]
                ylabel = "|S| (dB)"
            else:
                y_vals = magnitudes
                ylabel = "|S|"

            ax.plot(self.wavelengths, y_vals, label=name, **kwargs)

        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("S-Parameters")
        fig.tight_layout()

        return fig
