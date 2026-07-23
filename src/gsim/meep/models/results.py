"""Result models for MEEP simulation output."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
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


class ModeResult(BaseModel):
    """Eigenmode solution from standalone mode solving.

    Returned by :func:`solve_slab_mode` and :func:`solve_cross_section_mode`.
    Contains the effective index, field profiles over the cross-section,
    the dominant wavevector, and metadata about the mode.

    Attributes:
        n_eff: Effective index (real part of propagation constant / k0).
        wavelength: Free-space wavelength in µm.
        frequency: Frequency in MEEP units (1/µm = 1/wavelength).
        fields: Complex field arrays keyed by component name
            (e.g. ``{"Ex": array, "Ey": array, "Ez": array, ...}``).
        kdom: Dominant wavevector [kx, ky, kz] in MPB k-units (``k*a/(2*pi)``,
            equivalent to ``n_eff/lambda`` in 1/um).  The true propagation
            constant is ``beta = 2*pi*|kdom|`` rad/um; ``n_eff = |kdom|*lambda``.
        n_group: Group index, if computable.
        band_num: Mode band index (1 = fundamental).
        parity: Parity of the mode (``"NO_PARITY"``, ``"EVEN_Y"``, etc.).
        x_grid: X-axis grid coordinates in µm (absolute frame).  Populated
            for XZ cross-section modes.  ``None`` when unavailable.
        y_grid: Y-axis grid coordinates in µm (absolute frame).  Populated
            for YZ cross-section modes.  ``None`` when unavailable.
        z_grid: Z-axis grid coordinates in µm (absolute frame).  Populated
            for both slab and cross-section modes.  ``None`` when
            unavailable.
        stack: :class:`LayerStack` used for mode solving.  Provides
            material boundary context for index profile reconstruction.
            ``None`` when unavailable.
        component: GDSFactory :class:`Component` for cross-section modes.
            ``None`` for slab modes.
        port_or_position: Port name (``str``) or position ``(x, y)`` used
            for cross-section mode extraction.  ``None`` for slab modes.
        cross_section_plane: ``"xz"`` or ``"yz"`` for cross-section modes.
            ``None`` for slab modes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    n_eff: float
    wavelength: float
    frequency: float
    fields: dict[str, np.ndarray] = Field(default_factory=dict)
    kdom: list[float] = Field(default_factory=list)
    n_group: float | None = None
    band_num: int = 1
    parity: str = "NO_PARITY"
    x_grid: np.ndarray | None = None
    y_grid: np.ndarray | None = None
    z_grid: np.ndarray | None = None
    stack: Any | None = Field(default=None, exclude=True)
    component: Any | None = Field(default=None, exclude=True)
    domain_config: Any | None = Field(default=None, exclude=True)
    port_or_position: str | tuple[float, float] | None = None
    cross_section_plane: str | None = None

    # ------------------------------------------------------------------
    # plot_mode
    # ------------------------------------------------------------------

    _VALID_NORMS = frozenset({"abs", "real", "imag", "phase"})

    def _auto_component(self) -> str:
        """Return the field component with the largest ``max(|field|)``."""
        best: str | None = None
        best_val = -1.0
        for name, arr in self.fields.items():
            v = float(np.max(np.abs(arr)))
            if v > best_val:
                best_val = v
                best = name
        if best is None:
            raise ValueError("No field components available.")
        return best

    @staticmethod
    def _apply_norm(field: np.ndarray, norm: str) -> np.ndarray:
        """Transform complex field array for display."""
        if norm == "abs":
            return np.abs(field)
        if norm == "real":
            return field.real
        if norm == "imag":
            return field.imag
        if norm == "phase":
            return np.angle(field)
        raise ValueError(
            f"Unknown norm {norm!r}. Must be one of: abs, real, imag, phase."
        )

    def _get_ndim(self) -> int:
        """Return dimensionality from the first available field component."""
        for arr in self.fields.values():
            return arr.ndim
        return 0

    def _get_horizontal_grid(self) -> tuple[np.ndarray, str]:
        """Return ``(horizontal_grid, axis_label)``.

        Raises ``ValueError`` if both ``x_grid`` and ``y_grid`` are
        populated (ambiguous geometry) or if no horizontal grid is set.
        """
        has_x = self.x_grid is not None
        has_y = self.y_grid is not None
        if has_x and has_y:
            raise ValueError(
                "Ambiguous geometry: both x_grid and y_grid are populated."
            )
        if has_y:
            return self.y_grid, "y (µm)"
        if has_x:
            return self.x_grid, "x (µm)"
        raise ValueError(
            "No horizontal grid available. Set x_grid or y_grid on ModeResult."
        )

    def plot_mode(
        self,
        components: str | list[str] = "auto",
        *,
        norm: str = "abs",
        index: bool = False,
        geometry: bool = False,
        geom_kwargs: dict[str, Any] | None = None,
        ax: Any | None = None,
        figsize: tuple[float, float] = (8, 6),
        cmap: str | None = None,
        aspect: str = "equal",
        title: str | None = "auto",
        suptitle: str | None = "auto",
        show: bool = True,
        shared_colorbar: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot eigenmode field profiles.

        Args:
            components: ``"auto"`` (dominant), ``"all"`` (every field), or
                list of component names e.g. ``["Ey", "Hx"]``.
            norm: ``"abs"`` (magnitude), ``"real"``, ``"imag"``, or
                ``"phase"`` (angle in radians).
            index: Overlay refractive index profile (twin-axis for 1D,
                greyscale underlay for 2D).
            geometry: Overlay structural geometry boundaries (material
                interfaces).  1D: horizontal lines at dielectric
                z-interfaces.  2D: prism outline rectangles at the
                cross-section plane.  Requires ``stack`` (and
                ``component`` + ``domain_config`` for 2D).
            geom_kwargs: Forwarded to the geometry overlay's matplotlib
                artist.  Defaults: ``color="white"``, ``linestyle="-"``,
                ``linewidth=1.0``, ``zorder=50``.  User values override
                defaults.  Only used when ``geometry=True``.
            ax: Existing matplotlib ``Axes``.  Only valid when
                ``components`` resolves to a single component.
            figsize: ``(width, height)`` tuple in inches.
            cmap: Matplotlib colormap name for field data.  When ``None``
                (default), auto-selected: ``"inferno"`` for ``abs``,
                ``"RdBu_r"`` for ``real``/``imag``, ``"twilight"`` for
                ``"phase"``.
            aspect: ``"equal"`` or ``"auto"`` for ``pcolormesh`` subplots.
            title: Per-subplot title. ``"auto"`` generates
                ``|comp|  n_eff=...`` for single-component, component name
                only for multi-component.  ``None`` suppresses.
            suptitle: Figure-level suptitle.  ``"auto"`` generates
                descriptive text for multi-component.  ``None`` suppresses.
            show: Call ``plt.show()`` (default ``True``).  Set ``False``
                for customisation before display.
            shared_colorbar: Use a single colour bar for all subplots
                instead of per-subplot colour bars.
            **kwargs: Forwarded to the underlying matplotlib call
                (``ax.plot`` for 1D, ``ax.pcolormesh`` for 2D).

        Returns:
            ``(fig, ax)`` for single-component, ``(fig, axes)`` for
            multi-component.

        Raises:
            ValueError: If ``fields`` is empty, ``ax`` is passed with
                multi-component, ``norm`` is invalid, or horizontal axis
                is ambiguous.
        """
        if not self.fields:
            raise ValueError(
                "ModeResult has no field data. "
                "Pass field grids to the solver to enable plotting."
            )
        if self.x_grid is not None and self.y_grid is not None:
            raise ValueError(
                "Ambiguous geometry: both x_grid and y_grid are populated."
            )
        if norm not in self._VALID_NORMS:
            raise ValueError(
                f"Unknown norm {norm!r}. Must be one of: "
                f"{', '.join(sorted(self._VALID_NORMS))}."
            )

        resolved = self._resolve_components(components)
        if cmap is None:
            cmap = self._colormap_for_norm(norm)  # type: ignore[assignment]
        is_single = len(resolved) == 1
        if ax is not None and not is_single:
            raise ValueError(
                "ax may only be passed when components resolves to a single component."
            )

        ndim = self._get_ndim()
        if ndim == 0:
            raise ValueError(
                "ModeResult has no field data. "
                "Pass field grids to the solver to enable plotting."
            )

        if ndim == 1:
            return self._plot_mode_1d(
                resolved,
                norm=norm,
                index=index,
                geometry=geometry,
                geom_kwargs=geom_kwargs,
                ax=ax,
                figsize=figsize,
                title=title,
                suptitle=suptitle,
                show=show,
                **kwargs,
            )
        return self._plot_mode_2d(
            resolved,
            norm=norm,
            index=index,
            geometry=geometry,
            geom_kwargs=geom_kwargs,
            ax=ax,
            figsize=figsize,
            cmap=cmap,
            aspect=aspect,
            title=title,
            suptitle=suptitle,
            show=show,
            shared_colorbar=shared_colorbar,
            **kwargs,
        )

    def _resolve_components(self, components: str | list[str]) -> list[str]:
        """Resolve ``components`` to a concrete list of field names."""
        if isinstance(components, str):
            if components == "auto":
                return [self._auto_component()]
            if components == "all":
                return [
                    c for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") if c in self.fields
                ]
            if components not in self.fields:
                available = sorted(self.fields)
                raise ValueError(
                    f"Unknown component: {components!r}. Available: {available}."
                )
            return [components]
        missing = [c for c in components if c not in self.fields]
        if missing:
            available = sorted(self.fields)
            raise ValueError(
                f"Unknown component(s): {missing}. Available: {available}."
            )
        return list(components)

    @staticmethod
    def _colormap_for_norm(norm: str) -> str:
        """Return a sensible colormap for a given field *norm*."""
        if norm == "abs":
            return "inferno"
        if norm in ("real", "imag"):
            return "RdBu_r"
        if norm == "phase":
            return "twilight"
        return "inferno"

    # -- 1D helpers -------------------------------------------------------

    def _make_title(
        self, comp: str, user_title: str | None, is_single: bool
    ) -> str | None:
        """Build a per-subplot title from *user_title* and component info."""
        if user_title is None:
            return None
        if user_title == "auto":
            if is_single:
                return f"|{comp}|  n_eff={self.n_eff:.4f}"
            return comp
        return user_title

    def _plot_mode_1d(
        self,
        comps: list[str],
        *,
        norm: str,
        index: bool,
        geometry: bool,
        geom_kwargs: dict[str, Any] | None,
        ax: Any | None,
        figsize: tuple[float, float],
        title: str | None,
        suptitle: str | None,
        show: bool,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """1D line-plot path: one subplot per component stacked vertically."""
        import matplotlib.pyplot as plt

        is_single = len(comps) == 1
        z = self.z_grid
        if z is None:
            raise ValueError("z_grid must be set on ModeResult for 1D plotting.")

        if ax is not None:
            fig = ax.figure
            axes = [ax]
        else:
            ncols = 1
            fig, axes_arr = plt.subplots(
                nrows=len(comps),
                ncols=ncols,
                figsize=figsize,
                squeeze=False,
            )
            axes = [axes_arr[i, 0] for i in range(len(comps))]

        for i, comp in enumerate(comps):
            ax_i = axes[i]
            data = self._apply_norm(self.fields[comp], norm)

            ax_i.plot(z, data, **kwargs)
            if geometry and self.stack is not None:
                self._draw_geometry_overlay_1d(ax_i, **(geom_kwargs or {}))
            if index:
                ax_i_twin = ax_i.twinx()
                n_prof = self._compute_index_profile_1d()
                ax_i_twin.plot(z, n_prof, color="gray", linestyle="--", alpha=0.7)
                ax_i_twin.set_ylabel("n", color="gray")
            t = self._make_title(comp, title, is_single)
            if t is not None:
                ax_i.set_title(t)
            ax_i.set_xlabel("z (µm)")

        if suptitle == "auto" and not is_single:
            norm_label = "|field|" if norm == "abs" else norm
            fig.suptitle(f"Mode fields ({norm_label})", fontsize=12)
        elif suptitle is not None and suptitle != "auto":
            fig.suptitle(suptitle, fontsize=12)

        fig.tight_layout()
        if show:
            plt.show()

        if is_single:
            return (fig, axes[0])
        return (fig, axes_arr)  # Always return array for multi-component

    def _compute_index_profile_1d(self) -> np.ndarray:
        """Compute 1D refractive index profile from stored context."""
        if self.stack is None:
            raise ValueError(
                "No stack stored on ModeResult. Cannot compute index profile."
            )
        if self.z_grid is None:
            raise ValueError("z_grid must be set on ModeResult for index profile.")
        from gsim.meep.mode_solver import refractive_index_profile

        return refractive_index_profile(
            self.stack,
            self.wavelength,
            z_grid=self.z_grid,
        )

    # -- 2D helpers -------------------------------------------------------

    def _plot_mode_2d(
        self,
        comps: list[str],
        *,
        norm: str,
        index: bool,
        geometry: bool,
        geom_kwargs: dict[str, Any] | None,
        ax: Any | None,
        figsize: tuple[float, float],
        cmap: str,
        aspect: str,
        title: str | None,
        suptitle: str | None,
        show: bool,
        shared_colorbar: bool,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """2D pcolormesh path: subplot grid with optional index overlay."""
        import matplotlib.pyplot as plt

        is_single = len(comps) == 1
        h_grid, h_label = self._get_horizontal_grid()
        z = self.z_grid
        if z is None:
            raise ValueError("z_grid must be set on ModeResult for 2D plotting.")

        if ax is not None:
            fig = ax.figure
            axes = [ax]
            nrows = ncols = 1
        else:
            ncols = min(len(comps), 3)
            nrows = (len(comps) + ncols - 1) // ncols
            fig, axes_arr = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                squeeze=False,
            )
            axes = [axes_arr.flat[i] for i in range(len(comps))]

        index_overlay: np.ndarray | None = None
        if index:
            index_overlay = self._compute_index_profile_2d()

        images: list[Any] = []
        vmin = vmax = None
        if shared_colorbar:
            vmin = min(
                float(self._apply_norm(self.fields[c], norm).min()) for c in comps
            )
            vmax = max(
                float(self._apply_norm(self.fields[c], norm).max()) for c in comps
            )

        for i, comp in enumerate(comps):
            ax_i = axes[i]
            data = self._apply_norm(self.fields[comp], norm)
            im = ax_i.pcolormesh(
                h_grid,
                z,
                data,
                cmap=cmap,
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                **kwargs,
            )
            if index_overlay is not None:
                ax_i.pcolormesh(
                    h_grid,
                    z,
                    index_overlay,
                    cmap="Greys",
                    alpha=0.25,
                    shading="auto",
                )
            if geometry:
                self._draw_geometry_overlay_2d(ax_i, **(geom_kwargs or {}))
            images.append(im)
            ax_i.set_aspect(aspect)
            t = self._make_title(comp, title, is_single)
            if t is not None:
                ax_i.set_title(t)
            ax_i.set_xlabel(h_label)
            ax_i.set_ylabel("z (µm)")
            if not shared_colorbar:
                fig.colorbar(im, ax=ax_i)

        if shared_colorbar and images:
            for ax_i in axes[len(comps) :]:
                ax_i.set_visible(False)
            fig.colorbar(images[-1], ax=axes, shrink=0.6)

        if suptitle == "auto" and not is_single:
            norm_label = "|field|" if norm == "abs" else norm
            fig.suptitle(
                f"Mode fields ({norm_label})  n_eff={self.n_eff:.4f}",
                fontsize=12,
            )
        elif suptitle is not None and suptitle != "auto":
            fig.suptitle(suptitle, fontsize=12)

        fig.tight_layout()
        if show:
            plt.show()

        if is_single:
            return (fig, axes[0])
        return (fig, axes_arr)

    def _compute_index_profile_2d(self) -> np.ndarray:
        """Compute 2D refractive index profile from stored context."""
        if self.stack is None:
            raise ValueError(
                "No stack stored on ModeResult. Cannot compute index profile."
            )
        if self.z_grid is None:
            raise ValueError("z_grid must be set on ModeResult.")
        from gsim.meep.mode_solver import refractive_index_profile

        h_grid, _ = self._get_horizontal_grid()
        if self.y_grid is not None:
            return refractive_index_profile(
                self.stack,
                self.wavelength,
                z_grid=self.z_grid,
                y_grid=h_grid,
                component=self.component,
                port=self.port_or_position
                if isinstance(self.port_or_position, str)
                else None,
            )
        return refractive_index_profile(
            self.stack,
            self.wavelength,
            z_grid=self.z_grid,
            x_grid=h_grid,
            component=self.component,
            port=self.port_or_position
            if isinstance(self.port_or_position, str)
            else None,
        )

    # ------------------------------------------------------------------
    # geometry overlay helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _geom_default_kwargs() -> dict[str, Any]:
        """Return default matplotlib kwargs for geometry overlay lines."""
        return {"color": "white", "linestyle": "-", "linewidth": 1.0, "zorder": 50}

    def _draw_geometry_overlay_1d(self, ax: Any, **geom_kwargs: Any) -> None:
        """Draw horizontal lines at dielectric z-interfaces.

        Each unique z-boundary from ``stack.dielectrics`` gets a line
        across the full x-span.
        """
        if self.stack is None or not self.stack.dielectrics:
            return

        cfg = self._geom_default_kwargs()
        cfg.update(geom_kwargs)

        z_vals: set[float] = set()
        for diel in self.stack.dielectrics:
            z_vals.add(diel["zmin"])
            z_vals.add(diel["zmax"])

        for z in sorted(z_vals):
            ax.axhline(y=z, **cfg)

    def _draw_geometry_overlay_2d(self, ax: Any, **geom_kwargs: Any) -> None:
        """Draw prism outline rectangles at the cross-section slice plane.

        Requires ``component`` and ``stack`` on this result.
        For an XZ view (``cross_section_plane == "xz"``) the slice is at
        the port y-coordinate; for YZ it is at the port x-coordinate.

        Uses :func:`gsim.meep.viz.build_cross_section_rectangles` which
        extracts polygon intersections from the *gsim* :class:`LayerStack`
        directly — no PDK layer stack dependency.
        """
        import math

        from matplotlib.patches import Polygon, Rectangle

        import gsim.meep.viz as meep_viz

        if self.component is None or self.stack is None:
            logger.info(
                "Geometry overlay requires component and stack "
                "on ModeResult — skipping geometry boundaries."
            )
            return

        if self.cross_section_plane == "xz":
            slice_axis = "y"
            if isinstance(self.port_or_position, tuple):
                slice_coord = self.port_or_position[1]
            elif isinstance(self.port_or_position, str):
                port = next(
                    (
                        p
                        for p in self.component.ports
                        if p.name == self.port_or_position
                    ),
                    None,
                )
                if port is None:
                    logger.info(
                        "Port %r not found on component — "
                        "skipping geometry boundaries.",
                        self.port_or_position,
                    )
                    return
                slice_coord = port.center[1]
            else:
                logger.info(
                    "Cannot determine slice coordinate — skipping geometry boundaries."
                )
                return
        elif self.cross_section_plane == "yz":
            slice_axis = "x"
            if isinstance(self.port_or_position, tuple):
                slice_coord = self.port_or_position[0]
            elif isinstance(self.port_or_position, str):
                port = next(
                    (
                        p
                        for p in self.component.ports
                        if p.name == self.port_or_position
                    ),
                    None,
                )
                if port is None:
                    return
                slice_coord = port.center[0]
            else:
                return
        else:
            return

        try:
            rects = meep_viz.build_cross_section_rectangles(
                self.component, self.stack, slice_axis, slice_coord
            )
        except Exception:
            logger.info(
                "Could not build cross-section geometry — "
                "skipping geometry boundaries.",
                exc_info=True,
            )
            return

        if not rects:
            return

        h_min_geom: float = float("inf")
        h_max_geom: float = -float("inf")
        z_min_geom: float = float("inf")
        z_max_geom: float = -float("inf")

        cfg = self._geom_default_kwargs()
        cfg.update(geom_kwargs)
        rect_kwargs: dict[str, Any] = {k: v for k, v in cfg.items() if k != "color"}
        rect_kwargs.setdefault("edgecolor", cfg["color"])
        rect_kwargs.setdefault("facecolor", "none")

        layer_drawn: set[str] = set()
        for rect in rects:
            h_min = rect["h_min"]
            h_max = rect["h_max"]
            z_min = rect["z_min"]
            z_max = rect["z_max"]
            layer_name = rect["layer_name"]
            sw_deg = float(rect.get("sidewall_angle", 0.0) or 0.0)

            if math.isinf(h_min) or math.isinf(h_max):
                ax_h_min, ax_h_max = ax.get_xlim()
                h_min = ax_h_min
                h_max = ax_h_max

            if h_max - h_min < 1e-12:
                continue

            label = layer_name if layer_name not in layer_drawn else None
            layer_drawn.add(layer_name)

            if sw_deg and not math.isinf(z_min) and not math.isinf(z_max):
                sw_rad = math.radians(sw_deg)
                dx = math.tan(sw_rad) * (z_max - z_min)
                xy_vertices = [
                    (h_min, z_min),
                    (h_max, z_min),
                    (h_max - dx, z_max),
                    (h_min + dx, z_max),
                ]
                ax.add_patch(
                    Polygon(
                        xy_vertices,
                        closed=True,
                        label=label,
                        **rect_kwargs,
                    )
                )
            else:
                ax.add_patch(
                    Rectangle(
                        (h_min, z_min),
                        h_max - h_min,
                        z_max - z_min,
                        label=label,
                        **rect_kwargs,
                    )
                )
            h_min_geom = min(h_min_geom, h_min)
            h_max_geom = max(h_max_geom, h_max)
            z_min_geom = min(z_min_geom, z_min)
            z_max_geom = max(z_max_geom, z_max)

        if h_min_geom < float("inf"):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xlim(min(xlim[0], h_min_geom), max(xlim[1], h_max_geom))
            ax.set_ylim(min(ylim[0], z_min_geom), max(ylim[1], z_max_geom))

    # ------------------------------------------------------------------
    # plot_index
    # ------------------------------------------------------------------

    def plot_index(
        self,
        *,
        ax: Any | None = None,
        figsize: tuple[float, float] = (7, 5),
        cmap: str = "RdYlBu",
        show: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """Plot the refractive index profile from stored context.

        Zero-argument when grid and stack fields are populated.
        Auto-detects 1D vs 2D from the available grid arrays.

        Args:
            ax: Existing matplotlib ``Axes`` (optional).
            figsize: ``(width, height)`` in inches.
            cmap: Colormap name for 2D ``pcolormesh`` (default
                ``"RdYlBu"``).
            show: Call ``plt.show()`` (default ``True``).
            **kwargs: Forwarded to ``ax.plot`` (1D) or ``ax.pcolormesh``
                (2D).

        Returns:
            ``(fig, ax)``.

        Raises:
            ValueError: If ``stack``, ``wavelength``, or ``z_grid`` are
                missing.
        """
        if self.stack is None:
            raise ValueError(
                "No stack stored on ModeResult. Cannot compute index profile."
            )
        if self.z_grid is None:
            raise ValueError(
                "No z_grid stored on ModeResult. Cannot compute index profile."
            )

        ndim = self._get_ndim()
        if ndim <= 1 and self.y_grid is None and self.x_grid is None:
            return self._plot_index_1d(
                ax=ax,
                figsize=figsize,
                show=show,
                cmap=cmap,
                **kwargs,
            )

        return self._plot_index_2d(
            ax=ax,
            figsize=figsize,
            cmap=cmap,
            show=show,
            **kwargs,
        )

    def _plot_index_1d(
        self,
        *,
        ax: Any | None,
        figsize: tuple[float, float],
        show: bool,
        cmap: str,  # noqa: ARG002
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """1D line-plot of n(z) from stored context."""
        if self.z_grid is None:
            raise ValueError("z_grid must be set on ModeResult.")
        import matplotlib.pyplot as plt

        n_prof = self._compute_index_profile_1d()
        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self.z_grid, n_prof, **kwargs)
        ax.set_xlabel("z (µm)")
        ax.set_ylabel("n")
        ax.set_title(f"Refractive index  (lambda={self.wavelength:.2f} um)")

        fig.tight_layout()
        if show:
            plt.show()

        return (fig, ax)

    def _plot_index_2d(
        self,
        *,
        ax: Any | None,
        figsize: tuple[float, float],
        cmap: str,
        show: bool,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        """2D pcolormesh of n(y,z) or n(x,z) from stored context."""
        if self.z_grid is None:
            raise ValueError("z_grid must be set on ModeResult.")
        import matplotlib.pyplot as plt

        n_prof = self._compute_index_profile_2d()
        h_grid, h_label = self._get_horizontal_grid()

        if ax is not None:
            fig = ax.figure
        else:
            fig, ax = plt.subplots(figsize=figsize)

        im = ax.pcolormesh(
            h_grid,
            self.z_grid,
            n_prof,
            cmap=cmap,
            shading="auto",
            **kwargs,
        )
        ax.set_aspect("equal")
        ax.set_xlabel(h_label)
        ax.set_ylabel("z (µm)")
        ax.set_title(f"Refractive index  (lambda={self.wavelength:.2f} um)")
        fig.colorbar(im, ax=ax)

        fig.tight_layout()
        if show:
            plt.show()

        return (fig, ax)
