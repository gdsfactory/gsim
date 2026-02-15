"""Configuration models for MEEP photonic simulation.

Defines serializable Pydantic models for the complete simulation config
that gets written as JSON and consumed by the cloud MEEP runner script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class SymmetryEntry(BaseModel):
    """One MEEP mirror symmetry plane."""

    model_config = ConfigDict(validate_assignment=True)

    direction: Literal["X", "Y", "Z"]
    phase: Literal[1, -1] = Field(default=1)


class DomainConfig(BaseModel):
    """Simulation domain sizing: margins around geometry + PML thickness.

    Margins control how much material (from the layer stack) is kept around
    the waveguide core.  ``set_z_crop()`` uses ``margin_z_above`` /
    ``margin_z_below`` to determine the crop window.  Along XY the margin
    is the gap between the geometry bounding-box and the PML inner edge.

    Cell size formula:
        cell_x = bbox_width  + 2*(margin_xy + dpml)
        cell_y = bbox_height + 2*(margin_xy + dpml)
        cell_z = z_extent + 2*dpml          (z-margins baked into z_extent via set_z_crop)
    """

    model_config = ConfigDict(validate_assignment=True)

    dpml: float = Field(default=1.0, ge=0, description="PML thickness in um")
    margin_xy: float = Field(
        default=0.5, ge=0, description="XY margin between geometry and PML in um"
    )
    margin_z_above: float = Field(
        default=0.5, ge=0, description="Z margin above core kept by set_z_crop in um"
    )
    margin_z_below: float = Field(
        default=0.5, ge=0, description="Z margin below core kept by set_z_crop in um"
    )
    port_margin: float = Field(
        default=0.5,
        ge=0,
        description="Margin on each side of port waveguide width for mode monitors in um",
    )
    extend_ports: float = Field(
        default=0.0,
        ge=0,
        description="Length to extend waveguide ports into PML in um. "
        "0 = auto (margin_xy + dpml).",
    )


class StoppingConfig(BaseModel):
    """Controls when the MEEP simulation stops.

    ``fixed`` mode runs for a fixed time after sources turn off.
    ``decay`` mode monitors field decay at a point and stops when the
    fields have decayed by ``decay_by``, with ``run_after_sources`` as
    a numeric time cap (whichever fires first).
    ``dft_decay`` mode monitors convergence of all DFT monitors and
    stops when they stabilize, with built-in min/max time bounds.
    Best for S-parameter extraction.
    """

    model_config = ConfigDict(validate_assignment=True)

    mode: Literal["fixed", "decay", "dft_decay"] = Field(default="fixed")
    run_after_sources: float = Field(default=100.0, gt=0)
    decay_dt: float = Field(default=50.0, gt=0)
    decay_component: str = Field(default="Ey")
    decay_by: float = Field(default=1e-3, gt=0, lt=1)
    decay_monitor_port: str | None = Field(default=None)
    dft_min_run_time: float = Field(
        default=100,
        ge=0,
        description="Minimum run time after sources for dft_decay mode. "
        "Must exceed pulse transit time through the device to avoid "
        "false convergence on near-zero fields at output ports.",
    )


class SourceConfig(BaseModel):
    """Source excitation configuration.

    Controls the Gaussian source bandwidth and which port is excited.
    When ``bandwidth`` is ``None`` (auto), ``compute_fwidth`` returns a
    bandwidth ~3x wider than the monitor frequency span (matching
    gplugins' ``dfcen=0.2`` convention) so edge frequencies receive
    adequate spectral power.
    """

    model_config = ConfigDict(validate_assignment=True)

    bandwidth: float | None = Field(
        default=None,
        description="Source Gaussian bandwidth in wavelength um. None = auto (~3x monitor bw).",
    )
    port: str | None = Field(
        default=None,
        description="Source port name. None = auto-select first port.",
    )
    fwidth: float = Field(
        default=0.0,
        ge=0,
        description="Computed source fwidth in frequency units. "
        "Set automatically by write_config(); 0 = not yet computed.",
    )

    def compute_fwidth(self, fcen: float, monitor_df: float) -> float:
        """Compute Gaussian source fwidth in frequency units.

        When auto (bandwidth=None), returns ``max(3 * monitor_df, 0.2 * fcen)``
        to ensure edge frequencies have enough spectral power.

        Args:
            fcen: Center frequency (1/um).
            monitor_df: Monitor frequency span (1/um).

        Returns:
            Source fwidth in frequency units (1/um).
        """
        if self.bandwidth is not None:
            # Convert wavelength bandwidth to frequency bandwidth
            wl_center = 1.0 / fcen
            wl_min = wl_center - self.bandwidth / 2
            wl_max = wl_center + self.bandwidth / 2
            return 1.0 / wl_min - 1.0 / wl_max
        return max(3 * monitor_df, 0.2 * fcen)


class FDTDConfig(BaseModel):
    """Wavelength and frequency settings for MEEP FDTD simulation.

    MEEP uses normalized units where c = 1 and lengths are in um.
    Frequency f = 1/wavelength (in 1/um).
    """

    model_config = ConfigDict(validate_assignment=True)

    wavelength: float = Field(default=1.55, gt=0, description="Center wavelength in um")
    bandwidth: float = Field(
        default=0.1, ge=0, description="Wavelength bandwidth in um"
    )
    num_freqs: int = Field(default=11, ge=1, description="Number of frequency points")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fcen(self) -> float:
        """Center frequency in MEEP units (1/um, since c=1)."""
        return 1.0 / self.wavelength

    @computed_field  # type: ignore[prop-decorator]
    @property
    def df(self) -> float:
        """Frequency width in MEEP units."""
        wl_min = self.wavelength - self.bandwidth / 2
        wl_max = self.wavelength + self.bandwidth / 2
        f_max = 1.0 / wl_min
        f_min = 1.0 / wl_max
        return f_max - f_min


class ResolutionConfig(BaseModel):
    """MEEP grid resolution configuration."""

    model_config = ConfigDict(validate_assignment=True)

    pixels_per_um: int = Field(
        default=32, ge=4, description="Grid pixels per micrometer"
    )

    @classmethod
    def coarse(cls) -> ResolutionConfig:
        """Coarse resolution (16 pixels/um) for quick tests."""
        return cls(pixels_per_um=16)

    @classmethod
    def default(cls) -> ResolutionConfig:
        """Default resolution (32 pixels/um)."""
        return cls(pixels_per_um=32)

    @classmethod
    def fine(cls) -> ResolutionConfig:
        """Fine resolution (64 pixels/um) for production runs."""
        return cls(pixels_per_um=64)


class AccuracyConfig(BaseModel):
    """Controls MEEP subpixel averaging and polygon simplification.

    ``eps_averaging`` toggles MEEP's subpixel smoothing (expensive for
    complex polygons).  ``subpixel_maxeval`` / ``subpixel_tol`` tune the
    integration that implements it.  ``simplify_tol`` applies Shapely
    polygon simplification before extrusion, dramatically reducing vertex
    counts on dense GDS curves.
    """

    model_config = ConfigDict(validate_assignment=True)

    eps_averaging: bool = Field(default=True, description="Toggle subpixel averaging")
    subpixel_maxeval: int = Field(
        default=0, ge=0, description="Cap on integration evaluations (0=unlimited)"
    )
    subpixel_tol: float = Field(
        default=1e-4, gt=0, description="Subpixel integration tolerance"
    )
    simplify_tol: float = Field(
        default=0.0, ge=0, description="Shapely simplification tolerance in um (0=off)"
    )


class DiagnosticsConfig(BaseModel):
    """Controls diagnostic outputs from the MEEP runner."""

    model_config = ConfigDict(validate_assignment=True)

    save_geometry: bool = Field(default=True, description="Pre-run geometry cross-section plots")
    save_fields: bool = Field(default=True, description="Post-run field snapshot")
    save_epsilon_raw: bool = Field(
        default=False, description="Raw epsilon .npy (advanced)"
    )
    save_animation: bool = Field(
        default=False, description="Field animation MP4 (heavy, needs ffmpeg)"
    )
    animation_interval: float = Field(
        default=0.5, gt=0,
        description="MEEP time units between animation frames",
    )
    preview_only: bool = Field(
        default=False,
        description="Init sim and save geometry diagnostics, skip FDTD run",
    )


class PortData(BaseModel):
    """Serializable port data for the config JSON."""

    model_config = ConfigDict(validate_assignment=True)

    name: str
    center: list[float] = Field(description="[x, y, z] center coordinates")
    orientation: float = Field(description="Port orientation in degrees")
    width: float = Field(gt=0)
    normal_axis: int = Field(ge=0, le=1, description="0=x, 1=y")
    direction: Literal["+", "-"] = Field(description="Direction along normal axis")
    is_source: bool = False


class LayerStackEntry(BaseModel):
    """One layer in the stack config sent to the cloud runner.

    The cloud runner reads these entries alongside layout.gds to
    extract polygons per layer and extrude them to 3D prisms.
    """

    model_config = ConfigDict(validate_assignment=True)

    layer_name: str
    gds_layer: list[int] = Field(description="[layer_number, datatype]")
    zmin: float
    zmax: float
    material: str
    sidewall_angle: float = Field(default=0.0, description="Sidewall angle in degrees")


class MaterialData(BaseModel):
    """Optical material data for config JSON."""

    model_config = ConfigDict(validate_assignment=True)

    refractive_index: float = Field(gt=0)
    extinction_coeff: float = Field(default=0.0, ge=0)


class SimConfig(BaseModel):
    """Complete serializable simulation config written as JSON.

    This is the top-level config that the cloud MEEP runner reads.
    The geometry is NOT included here — it's in the GDS file.
    The layer_stack tells the runner how to extrude each GDS layer.
    """

    model_config = ConfigDict(validate_assignment=True)

    gds_filename: str = Field(
        default="layout.gds", description="GDS file with 2D layout"
    )
    component_bbox: list[float] | None = Field(
        default=None,
        description="Original component bbox [xmin, ymin, xmax, ymax] before port extension.",
    )
    layer_stack: list[LayerStackEntry] = Field(default_factory=list)
    dielectrics: list[dict[str, Any]] = Field(default_factory=list)
    ports: list[PortData] = Field(default_factory=list)
    materials: dict[str, MaterialData] = Field(default_factory=dict)
    fdtd: FDTDConfig = Field(default_factory=FDTDConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    stopping: StoppingConfig = Field(default_factory=StoppingConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    domain: DomainConfig = Field(default_factory=DomainConfig)
    accuracy: AccuracyConfig = Field(default_factory=AccuracyConfig)
    verbose_interval: float = Field(
        default=0, ge=0, description="MEEP time units between progress prints (0=off)"
    )
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    symmetries: list[SymmetryEntry] = Field(default_factory=list)
    split_chunks_evenly: bool = Field(default=False)
    meep_np: int = Field(
        default=1, ge=1, description="Recommended MPI process count"
    )

    def to_json(self, path: str | Path) -> Path:
        """Write config to JSON file.

        Args:
            path: Output file path

        Returns:
            Path to the written file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.model_dump(), indent=2))
        return path
