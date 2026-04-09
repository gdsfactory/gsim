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

    def plot(
        self,
        db: bool = True,
        keys: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Plot S-parameters vs wavelength.

        Args:
            db: If True, plot in dB scale
            keys: S-parameter names to plot (e.g. ["s21", "s31"]). Plots all if None.
            **kwargs: Passed to matplotlib plot()

        Returns:
            matplotlib Figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plt.close(fig)  # prevent double display in notebooks

        ylabel = "|S| (dB)" if db else "|S|"
        names = self._resolve_keys(keys)
        for name in names:
            values = self.s_params[name]
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

    def _resolve_keys(self, keys: list[str] | None) -> list[str]:
        """Resolve key names with case-insensitive lookup."""
        if keys is None:
            return list(self.s_params.keys())
        lower_map = {k.lower(): k for k in self.s_params}
        return [lower_map.get(k.lower(), k) for k in keys]

    def plot_plotly(self, keys: list[str] | None = None) -> Any:
        """Plot S-parameters with Plotly (interactive).

        Returns a ``plotly.graph_objects.Figure`` with magnitude (dB)
        and phase subplots.

        Args:
            keys: S-parameter names to plot (e.g. ["s21", "s31"]). Plots all if None.

        Returns:
            plotly Figure
        """
        import cmath
        import math

        from plotly.subplots import make_subplots  # type: ignore[import-untyped]

        names = self._resolve_keys(keys)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Magnitude (dB)", "Phase (deg)"),
        )
        for name in names:
            values = self.s_params[name]
            db = [20 * math.log10(abs(v)) if abs(v) > 0 else -100 for v in values]
            phase = [math.degrees(cmath.phase(v)) for v in values]

            fig.add_scatter(
                x=self.wavelengths,
                y=db,
                mode="lines+markers",
                name=name,
                legendgroup=name,
                row=1,
                col=1,
            )
            fig.add_scatter(
                x=self.wavelengths,
                y=phase,
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=False,
                row=2,
                col=1,
            )
        fig.update_xaxes(title_text="Wavelength (um)", row=2, col=1)
        fig.update_yaxes(title_text="dB", row=1, col=1)
        fig.update_yaxes(title_text="deg", row=2, col=1)
        fig.update_layout(title="S-Parameters", height=600)
        return fig

    def plot_interactive(self, phase: bool = False) -> Any:
        """Plot S-parameters with interactive legend toggling.

        Args:
            phase: If True, plot phase (deg). Default is magnitude (dB).

        Returns:
            plotly Figure
        """
        import cmath
        import math

        import plotly.graph_objects as go  # type: ignore[import-untyped]

        names = list(self.s_params.keys())

        # Hide reflections (Sii) and cap visible at 4
        def _is_reflection(name: str) -> bool:
            clean = name.upper().lstrip("S")
            return len(clean) >= 2 and clean[0] == clean[1]

        transmission = [n for n in names if not _is_reflection(n)]
        reflections = [n for n in names if _is_reflection(n)]
        ordered = transmission + reflections
        visible_set = set(transmission[:4])

        fig = go.Figure()

        for name in ordered:
            values = self.s_params[name]
            if phase:
                y = [math.degrees(cmath.phase(v)) for v in values]
            else:
                y = [20 * math.log10(abs(v)) if abs(v) > 0 else -100 for v in values]
            vis = True if name in visible_set else "legendonly"
            fig.add_scatter(
                x=self.wavelengths,
                y=y,
                mode="lines+markers",
                name=name,
                visible=vis,
            )

        ylabel = "Phase (deg)" if phase else "|S| (dB)"
        fig.update_layout(
            xaxis_title="Wavelength (µm)",
            yaxis_title=ylabel,
            width=700,
            height=400,
            margin=dict(t=40, b=40, l=60, r=10),
            legend=dict(
                groupclick="toggleitem",
                itemclick="toggle",
                itemdoubleclick="toggleothers",
                itemsizing="constant",
                bordercolor="#888",
                borderwidth=1,
                bgcolor="rgba(245,245,245,0.9)",
                entrywidthmode="pixels",
                entrywidth=70,
            ),
        )
        return fig
