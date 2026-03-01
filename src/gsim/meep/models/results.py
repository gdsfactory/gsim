"""Result models for MEEP simulation output."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


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
    diagnostic_images: dict[str, str] = Field(
        default_factory=dict,
        description="Diagnostic image paths: key -> filepath",
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
            with contextlib.suppress(json.JSONDecodeError, OSError):
                debug_info = json.loads(debug_path.read_text())

        # Auto-detect diagnostic PNGs
        diagnostic_images: dict[str, str] = {}
        for key, filename in [
            ("geometry_xy", "meep_geometry_xy.png"),
            ("geometry_xz", "meep_geometry_xz.png"),
            ("geometry_yz", "meep_geometry_yz.png"),
            ("fields_xy", "meep_fields_xy.png"),
            ("animation", "meep_animation.mp4"),
        ]:
            img_path = path.parent / filename
            if img_path.exists():
                diagnostic_images[key] = str(img_path)

        # Detect animation frame PNGs
        frame_pngs = sorted(path.parent.glob("meep_frame_*.png"))
        if frame_pngs:
            diagnostic_images["animation_frames"] = str(path.parent)

        return cls(
            wavelengths=wavelengths,
            s_params=s_params,
            port_names=sorted(port_names),
            debug_info=debug_info,
            diagnostic_images=diagnostic_images,
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> SParameterResult:
        """Load from directory — handles preview-only with no CSV.

        If ``s_parameters.csv`` exists, delegates to ``from_csv()``.
        Otherwise loads only debug info and diagnostic images (preview mode).

        Args:
            directory: Path to results directory

        Returns:
            SParameterResult instance
        """
        directory = Path(directory)
        csv_path = directory / "s_parameters.csv"
        if csv_path.exists():
            return cls.from_csv(csv_path)

        # Preview-only: load debug + images only
        debug_info: dict[str, Any] = {}
        debug_path = directory / "meep_debug.json"
        if debug_path.exists():
            with contextlib.suppress(json.JSONDecodeError, OSError):
                debug_info = json.loads(debug_path.read_text())

        diagnostic_images: dict[str, str] = {}
        for key, filename in [
            ("geometry_xy", "meep_geometry_xy.png"),
            ("geometry_xz", "meep_geometry_xz.png"),
            ("geometry_yz", "meep_geometry_yz.png"),
            ("fields_xy", "meep_fields_xy.png"),
            ("animation", "meep_animation.mp4"),
        ]:
            img_path = directory / filename
            if img_path.exists():
                diagnostic_images[key] = str(img_path)

        # Detect animation frame PNGs
        frame_pngs = sorted(directory.glob("meep_frame_*.png"))
        if frame_pngs:
            diagnostic_images["animation_frames"] = str(directory)

        return cls(
            debug_info=debug_info,
            diagnostic_images=diagnostic_images,
        )

    def show_diagnostics(self) -> None:
        """Display diagnostic images in Jupyter."""
        from IPython.display import Image, display

        if not self.diagnostic_images:
            logger.info("No diagnostic images available.")
            return

        for name, img_path in sorted(self.diagnostic_images.items()):
            if not img_path.endswith((".png", ".jpg", ".jpeg", ".gif")):
                continue  # skip video/directories; use show_animation() for MP4
            logger.info("--- %s ---", name)
            display(Image(filename=img_path))

    def show_animation(self) -> None:
        """Display field animation MP4 in Jupyter."""
        mp4_path = self.diagnostic_images.get("animation")
        if mp4_path is None:
            logger.info("No animation MP4 available.")
            return

        from IPython.display import Video, display

        display(Video(mp4_path, embed=True, mimetype="video/mp4"))

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
        plt.close(fig)  # prevent double display in notebooks

        ylabel = "|S| (dB)" if db else "|S|"
        for name, values in self.s_params.items():
            magnitudes = [abs(v) for v in values]
            if db:
                import math

                y_vals = [20 * math.log10(m) if m > 0 else -100 for m in magnitudes]
            else:
                y_vals = magnitudes

            ax.plot(self.wavelengths, y_vals, ".-", label=name, **kwargs)

        ax.set_xlabel("Wavelength (um)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title("S-Parameters")
        fig.tight_layout()

        return fig
